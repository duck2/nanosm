#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <stdint.h>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace nanosm {

// The instruction set's revision: the opcodes, each form's fields in the
// instruction word (encoding below) and what they mean. It is folded into the
// stamp a kernel carries (src/profile.hpp device_profile_fingerprint), so a
// reader built for another revision refuses the kernel. It changes with any
// change to what a word means, a reserved bit given a meaning included.
inline constexpr uint32_t isa_revision = 2;

// ===========================================================================
// The instruction set is two tables: the forms and the opcodes. Every fact
// about an opcode below (its encoding, its text, its registers, the unit that
// takes it) is read from them.
//
// A FORM is how a word's payload holds an instruction's operands. For each
// register field of Instruction -- dst, src0, src1, src2 -- one token says
// which register file it names and where in the word it sits:
//   no       the form has no such operand
//   V0 .. V3 a vector register in register field 0 .. 3 (encoding below)
//   U0 .. U2 a uniform register in register field 0 .. 2
//   P0       a predicate (p0..p6) in register field 0
//   X2       a vector register, or a uniform one when the U bit is set, in field 2
//   V0w V1w  a vector register and the ones after it, register_span of them
//            (eight for a v8 form, one otherwise)
//   U1n      a tma's coordinates: tma_coordinate_count uniform registers from field 1
//   U6       a uniform register in umov's 6-bit field (bits 37..32)
// dst is written and the sources are read, in every form. Then the
// immediate's kind (ImmediateKind below); whether the vector sources are read at
// issue or after it (by the shared-memory collect stage, the load / store
// unit's command slot and data beats, an acc.st's beats); and the operands as
// written: d a b c are dst src0 src1 src2 in their files, i the immediate, m
// the address [src0 + i], j a branch displacement, s a special register's
// name, x umov's 32-bit word, A and B an mma.async operand with its row spacing.
// ===========================================================================
#define NANOSM_FORMS(X) \
    /* form                 dst  src0 src1 src2 immediate   vector reads  operands */ \
    X(no_operands,          no,  no,  no,  no,  none,       at_issue,    "") \
    X(special,              V0,  no,  no,  no,  s22,        at_issue,    "d, s") \
    X(uniform_special,      U0,  no,  no,  no,  s22,        at_issue,    "d, s") \
    X(vector_imm,           V0,  no,  no,  no,  s22,        at_issue,    "d, i") \
    X(uniform_word,         U6,  no,  no,  no,  word,       at_issue,    "d, x") \
    X(vector_unary,         V0,  V1,  no,  no,  none,       at_issue,    "d, a") \
    X(vector_to_uniform,    U0,  V1,  no,  no,  none,       at_issue,    "d, a") \
    X(uniform_to_vector,    V0,  U1,  no,  no,  none,       at_issue,    "d, a") \
    X(uniform_unary,        U0,  U1,  no,  no,  none,       at_issue,    "d, a") \
    X(vector_binary,        V0,  V1,  X2,  no,  none,       at_issue,    "d, a, b") \
    X(vector_ternary,       V0,  V1,  X2,  V3,  none,       at_issue,    "d, a, b, c") \
    X(vector_binary_imm,    V0,  V1,  no,  no,  s22,        at_issue,    "d, a, i") \
    X(vector_compare,       P0,  V1,  X2,  no,  none,       at_issue,    "d, a, b") \
    X(vector_compare_imm,   P0,  V1,  no,  no,  s22,        at_issue,    "d, a, i") \
    X(uniform_binary,       U0,  U1,  U2,  no,  none,       at_issue,    "d, a, b") \
    X(uniform_binary_imm,   U0,  U1,  no,  no,  s22,        at_issue,    "d, a, i") \
    X(uniform_compare,      P0,  U1,  U2,  no,  none,       at_issue,    "d, a, b") \
    X(uniform_compare_imm,  P0,  U1,  no,  no,  s22,        at_issue,    "d, a, i") \
    X(load,                 V0w, V1,  no,  no,  memory,     after_issue, "d, m") \
    X(store,                no,  V0,  V1w, no,  memory,     after_issue, "m, b") \
    X(uniform_load,         V0,  U1,  no,  no,  s22,        at_issue,    "d, m") \
    X(atomic,               V0,  U1,  U2,  no,  s14,        at_issue,    "d, m, b") \
    X(acc_load,             V0w, U1,  no,  no,  s22,        at_issue,    "d, [a, i]") \
    X(acc_store,            no,  U0,  V1w, no,  s22,        after_issue, "[a, i], b") \
    X(barrier,              no,  U0,  no,  no,  none,       at_issue,    "[a]") \
    X(barrier_imm,          no,  U0,  no,  no,  s22,        at_issue,    "[a], i") \
    X(barrier_reg,          no,  U0,  U1,  no,  none,       at_issue,    "[a], b") \
    X(probe_imm,            P0,  U1,  no,  no,  s22,        at_issue,    "d, [a], i") \
    X(probe_reg,            P0,  U1,  U2,  no,  none,       at_issue,    "d, [a], b") \
    X(tma_load,             no,  U0,  U1n, U2,  descriptor, at_issue,    "[a], [i, b], [c]") \
    X(tma_store,            no,  U0,  U1n, U2,  descriptor, at_issue,    "[i, b], [a], [c]") \
    X(mma,                  no,  U1,  U2,  U0,  spacings,   at_issue,    "[c], A, B") \
    X(branch,               no,  no,  no,  no,  s22,        at_issue,    "j")

// One row per OPCODE: its value, its mnemonic, its form, the unit that takes
// it at issue (IssueUnit below), its register span (8: a v8 form), whether it sets
// a depbar (set: it must name one, sb0..sb5; no: the field must say none) and,
// for setp and usetp, its comparison. The enum Opcode is made from these
// rows, so every opcode has exactly one. The mma.async block (mma_block_first
// below) is sixteen more opcodes of one row, named by the profile.
#define NANOSM_OPCODES(X) \
    /* opcode                    value  mnemonic                 form                 unit          span depbar compare */ \
    X(sread,                     0x01, "sread",                  special,             alu,          1, no,  none) \
    X(mov,                       0x02, "mov",                    vector_imm,          alu,          1, no,  none) \
    X(add,                       0x03, "add",                    vector_binary,       alu,          1, no,  none) \
    X(addi,                      0x04, "addi",                   vector_binary_imm,   alu,          1, no,  none) \
    X(shli,                      0x05, "shli",                   vector_binary_imm,   alu,          1, no,  none) \
    X(fadd,                      0x06, "fadd",                   vector_binary,       fpu,          1, no,  none) \
    X(xori,                      0x07, "xori",                 vector_binary_imm,   alu,          1, no,  none) \
    X(setp_eq,                   0x08, "setp.eq.u32",            vector_compare,      alu,          1, no,  eq) \
    X(setp_ne,                   0x09, "setp.ne.u32",            vector_compare,      alu,          1, no,  ne) \
    X(setp_lt,                   0x0a, "setp.lt.u32",            vector_compare,      alu,          1, no,  lt) \
    X(setp_le,                   0x0b, "setp.le.u32",            vector_compare,      alu,          1, no,  le) \
    X(setp_gt,                   0x0c, "setp.gt.u32",            vector_compare,      alu,          1, no,  gt) \
    X(setp_ge,                   0x0d, "setp.ge.u32",            vector_compare,      alu,          1, no,  ge) \
    X(fmul,                      0x0e, "fmul",                   vector_binary,       fpu,          1, no,  none) \
    X(fsub,                      0x0f, "fsub",                   vector_binary,       fpu,          1, no,  none) \
    X(lds,                       0x10, "lds",                    load,                shared_read,  1, set, none) \
    X(sts,                       0x11, "sts",                    store,               shared_write, 1, set, none) \
    X(ldg,                       0x12, "ldg",                    load,                lsu,          1, set, none) \
    X(stg,                       0x13, "stg",                    store,               lsu,          1, set, none) \
    X(fmax,                      0x14, "fmax",                   vector_binary,       fpu,          1, no,  none) \
    X(fma,                       0x15, "fma",                    vector_ternary,      fpu,          1, no,  none) \
    /* rd = half(ra) | half(rb) << 16 */ \
    X(cvt_pack,                  0x16, "cvt.pack.f16x2",         vector_binary,       fpu,          1, no,  none) \
    X(ex2,                       0x17, "ex2",                    vector_unary,        sfu,          1, no,  none) \
    X(rcp,                       0x18, "rcp",                    vector_unary,        sfu,          1, no,  none) \
    /* (the trailing underscores dodge the C++ operator tokens; a register */ \
    /* shift uses the low 5 bits: a 5-bit shifter; shr and shri are logical) */ \
    X(and_,                      0x19, "and",                    vector_binary,       alu,          1, no,  none) \
    X(or_,                       0x1a, "or",                     vector_binary,       alu,          1, no,  none) \
    X(xor_,                      0x1b, "xor",                    vector_binary,       alu,          1, no,  none) \
    X(shl,                       0x1c, "shl",                    vector_binary,       alu,          1, no,  none) \
    X(shr,                       0x1d, "shr",                    vector_binary,       alu,          1, no,  none) \
    X(andi,                      0x1e, "andi",                   vector_binary_imm,   alu,          1, no,  none) \
    X(ori,                       0x1f, "ori",                    vector_binary_imm,   alu,          1, no,  none) \
    X(bra,                       0x20, "bra",                    branch,              control,      1, no,  none) \
    X(bra_uni,                   0x21, "bra.uni",                branch,              control,      1, no,  none) \
    X(shri,                      0x22, "shri",                   vector_binary_imm,   alu,          1, no,  none) \
    /* Per lane: each active lane's rA is its own sector-aligned address; */ \
    /* the sector's eight words are registers rD .. rD+7 at that lane. */ \
    X(ldg_v8,                    0x23, "ldg.v8",                 load,                lsu,          8, set, none) \
    X(stg_v8,                    0x24, "stg.v8",                 store,               lsu,          8, set, none) \
    /* The base in a uniform register: one sector request (one shmem bank */ \
    /* access), the word fans out to every lane. */ \
    X(ldg_uni,                   0x25, "ldg.uni",                uniform_load,        lsu,          1, set, none) \
    X(lds_uni,                   0x26, "lds.uni",                uniform_load,        shared_read,  1, set, none) \
    /* Admitted only once every earlier shmem access of the warp has reached */ \
    /* the RAM (stores written, loads sampled), so nothing after it issues */ \
    /* before that. Never auto-inserted. */ \
    X(fence_sh,                  0x27, "fence.sh",               no_operands,         fence,        1, no,  none) \
    X(setpi_eq,                  0x28, "setp.eq.u32",            vector_compare_imm,  alu,          1, no,  eq) \
    X(setpi_ne,                  0x29, "setp.ne.u32",            vector_compare_imm,  alu,          1, no,  ne) \
    X(setpi_lt,                  0x2a, "setp.lt.u32",            vector_compare_imm,  alu,          1, no,  lt) \
    X(setpi_le,                  0x2b, "setp.le.u32",            vector_compare_imm,  alu,          1, no,  le) \
    X(setpi_gt,                  0x2c, "setp.gt.u32",            vector_compare_imm,  alu,          1, no,  gt) \
    X(setpi_ge,                  0x2d, "setp.ge.u32",            vector_compare_imm,  alu,          1, no,  ge) \
    /* Divergence, written by the kernel author, never inferred. Every */ \
    /* divergence stack entry is a (mask, pc) pair. ssy LABEL (any */ \
    /* instruction) pushes (the warp's active mask, LABEL). A predicated bra */ \
    /* is the divergent branch: taken = active AND p, fall = active AND NOT */ \
    /* p; one side empty moves the whole mask, both non-empty push (taken, */ \
    /* target) and run the fall-through arm under fall. sync pops the top */ \
    /* entry and switches to it, whoever pushed it: the mask becomes the */ \
    /* entry's, the warp goes to the entry's pc. */ \
    X(ssy,                       0x2e, "ssy",                    branch,              control,      1, no,  none) \
    X(sync,                      0x2f, "sync",                   no_operands,         control,      1, no,  none) \
    /* The barriers: the address in a uniform register; an immediate and a */ \
    /* register operand are distinct opcodes. */ \
    X(mbar_init_i,               0x30, "mbar.init",              barrier_imm,         shared_write, 1, no,  none) \
    X(mbar_init_r,               0x31, "mbar.init",              barrier_reg,         shared_write, 1, no,  none) \
    X(mbar_arrive,               0x32, "mbar.arrive",            barrier,             shared_write, 1, no,  none) \
    X(mbar_arrive_expect_tx_i,   0x33, "mbar.arrive.expect_tx",  barrier_imm,         shared_write, 1, no,  none) \
    X(mbar_arrive_expect_tx_r,   0x34, "mbar.arrive.expect_tx",  barrier_reg,         shared_write, 1, no,  none) \
    X(mbar_test_wait_i,          0x35, "mbar.test_wait",         probe_imm,           shared_read,  1, no,  none) \
    X(mbar_test_wait_r,          0x36, "mbar.test_wait",         probe_reg,           shared_read,  1, no,  none) \
    X(mbar_try_wait_i,           0x37, "mbar.try_wait",          probe_imm,           shared_read,  1, no,  none) \
    X(mbar_try_wait_r,           0x38, "mbar.try_wait",          probe_reg,           shared_read,  1, no,  none) \
    /* Uniform shared-memory atomics on signed int32: atom.op.uni rD, */ \
    /* [urA + imm], urB. ONE 32-bit word per warp instruction: the address */ \
    /* and the operand are uniform registers, the update is applied once, */ \
    /* and every active lane's rD receives the word's old value. add wraps */ \
    /* (32-bit two's complement); exch writes the operand. An mbar-engine */ \
    /* operation: the global store pause covers its one read, the ALU's C */ \
    /* stages and its one write (D + Lb + C + 1 cycles canonical: 7). */ \
    X(atom_add_uni,              0x39, "atom.add.uni",           atomic,              shared_write, 1, set, none) \
    X(atom_and_uni,              0x3a, "atom.and.uni",           atomic,              shared_write, 1, set, none) \
    X(atom_xor_uni,              0x3b, "atom.xor.uni",           atomic,              shared_write, 1, set, none) \
    X(atom_exch_uni,             0x3c, "atom.exch.uni",          atomic,              shared_write, 1, set, none) \
    /* tma.load.Nd [urS], [DESC, urC], [urM] and tma.store.Nd [DESC, urC], */ \
    /* [urS], [urM], N = 1..4 (src/tma_descriptor.hpp): urS the buffer's */ \
    /* shared address, urC .. urC + N - 1 the tile's coordinates, urM the */ \
    /* barrier's address, DESC the descriptor's shared address (a constant). */ \
    X(tma_load_1d,               0x40, "tma.load.1d",            tma_load,            tma,          1, no,  none) \
    X(tma_load_2d,               0x41, "tma.load.2d",            tma_load,            tma,          1, no,  none) \
    X(tma_load_3d,               0x42, "tma.load.3d",            tma_load,            tma,          1, no,  none) \
    X(tma_load_4d,               0x43, "tma.load.4d",            tma_load,            tma,          1, no,  none) \
    X(tma_store_1d,              0x44, "tma.store.1d",           tma_store,           tma,          1, no,  none) \
    X(tma_store_2d,              0x45, "tma.store.2d",           tma_store,           tma,          1, no,  none) \
    X(tma_store_3d,              0x46, "tma.store.3d",           tma_store,           tma,          1, no,  none) \
    X(tma_store_4d,              0x47, "tma.store.4d",           tma_store,           tma,          1, no,  none) \
    /* The signed vector compare: lt, le, gt, ge as signed 32-bit; eq and ne */ \
    /* need no signed form. An immediate is sign-extended either way. */ \
    X(setp_lt_s32,               0x48, "setp.lt.s32",            vector_compare,      alu,          1, no,  lt_s) \
    X(setp_le_s32,               0x49, "setp.le.s32",            vector_compare,      alu,          1, no,  le_s) \
    X(setp_gt_s32,               0x4a, "setp.gt.s32",            vector_compare,      alu,          1, no,  gt_s) \
    X(setp_ge_s32,               0x4b, "setp.ge.s32",            vector_compare,      alu,          1, no,  ge_s) \
    X(setpi_lt_s32,              0x4c, "setp.lt.s32",            vector_compare_imm,  alu,          1, no,  lt_s) \
    X(setpi_le_s32,              0x4d, "setp.le.s32",            vector_compare_imm,  alu,          1, no,  le_s) \
    X(setpi_gt_s32,              0x4e, "setp.gt.s32",            vector_compare_imm,  alu,          1, no,  gt_s) \
    X(setpi_ge_s32,              0x4f, "setp.ge.s32",            vector_compare_imm,  alu,          1, no,  ge_s) \
    /* acc.ld rD, [urT, S] / acc.st [urT, S], rS: S the region's words per */ \
    /* row (src/layout.hpp: a power of two, 4 or more, at most tmem's words). */ \
    X(acc_ld,                    0x50, "acc.ld",                 acc_load,            acc_load,     1, set, none) \
    X(acc_ld_v8,                 0x51, "acc.ld.v8",              acc_load,            acc_load,     8, set, none) \
    X(acc_st,                    0x52, "acc.st",                 acc_store,           acc_store,    1, set, none) \
    X(acc_st_v8,                 0x53, "acc.st.v8",              acc_store,           acc_store,    8, set, none) \
    X(mma_commit,                0x62, "mma.commit",             barrier,             mma,          1, no,  none) \
    /* 0x70..0x7f: the mma.async block. */ \
    /* The uniform register file: urf.registers 32-bit registers per warp */ \
    /* (canonical 32). Uniform arithmetic runs on the SM's one integer */ \
    /* uniform ALU, one value per warp instruction. umov urD, imm (umovi) */ \
    /* takes any 32-bit word: a uniform register number needs at most 6 */ \
    /* bits, which leaves the whole word, a float's bits included. */ \
    X(umov,                      0x80, "umov",                   uniform_unary,       uniform,      1, no,  none) \
    X(umovi,                     0x81, "umov",                   uniform_word,        uniform,      1, no,  none) \
    X(uadd,                      0x82, "uadd",                   uniform_binary,      uniform,      1, no,  none) \
    X(uaddi,                     0x83, "uaddi",                  uniform_binary_imm,  uniform,      1, no,  none) \
    X(usub,                      0x84, "usub",                   uniform_binary,      uniform,      1, no,  none) \
    X(ushl,                      0x85, "ushl",                   uniform_binary,      uniform,      1, no,  none) \
    X(ushr,                      0x86, "ushr",                   uniform_binary,      uniform,      1, no,  none) \
    X(ushli,                     0x87, "ushli",                  uniform_binary_imm,  uniform,      1, no,  none) \
    X(ushri,                     0x88, "ushri",                  uniform_binary_imm,  uniform,      1, no,  none) \
    X(uand,                      0x89, "uand",                   uniform_binary,      uniform,      1, no,  none) \
    X(uor,                       0x8a, "uor",                    uniform_binary,      uniform,      1, no,  none) \
    X(uxor,                      0x8b, "uxor",                   uniform_binary,      uniform,      1, no,  none) \
    X(uandi,                     0x8c, "uandi",                  uniform_binary_imm,  uniform,      1, no,  none) \
    X(uori,                      0x8d, "uori",                   uniform_binary_imm,  uniform,      1, no,  none) \
    X(uxori,                     0x8e, "uxori",                  uniform_binary_imm,  uniform,      1, no,  none) \
    /* usread urD, WARP_ID / N_WARPS: a warp-uniform special register */ \
    X(usread,                    0x8f, "usread",                 uniform_special,     uniform,      1, no,  none) \
    /* Moves between the files: r2ur takes lane 0 of a vector register; ur2r */ \
    /* writes a uniform register into every lane (a fixed-latency vector */ \
    /* result, like mov). */ \
    X(r2ur,                      0x90, "r2ur",                   vector_to_uniform,   uniform,      1, no,  none) \
    X(ur2r,                      0x91, "ur2r",                   uniform_to_vector,   alu,          1, no,  none) \
    /* The uniform compare: setp's comparisons on uniform registers, on the */ \
    /* uniform ALU (a compare is a subtract); it writes pD with the same */ \
    /* value in every lane (src/timing.hpp issue_to_uniform_predicate_visible). */ \
    X(usetp_eq,                  0x92, "usetp.eq.u32",           uniform_compare,     uniform,      1, no,  eq) \
    X(usetp_ne,                  0x93, "usetp.ne.u32",           uniform_compare,     uniform,      1, no,  ne) \
    X(usetp_lt,                  0x94, "usetp.lt.u32",           uniform_compare,     uniform,      1, no,  lt) \
    X(usetp_le,                  0x95, "usetp.le.u32",           uniform_compare,     uniform,      1, no,  le) \
    X(usetp_gt,                  0x96, "usetp.gt.u32",           uniform_compare,     uniform,      1, no,  gt) \
    X(usetp_ge,                  0x97, "usetp.ge.u32",           uniform_compare,     uniform,      1, no,  ge) \
    X(usetpi_eq,                 0x98, "usetp.eq.u32",           uniform_compare_imm, uniform,      1, no,  eq) \
    X(usetpi_ne,                 0x99, "usetp.ne.u32",           uniform_compare_imm, uniform,      1, no,  ne) \
    X(usetpi_lt,                 0x9a, "usetp.lt.u32",           uniform_compare_imm, uniform,      1, no,  lt) \
    X(usetpi_le,                 0x9b, "usetp.le.u32",           uniform_compare_imm, uniform,      1, no,  le) \
    X(usetpi_gt,                 0x9c, "usetp.gt.u32",           uniform_compare_imm, uniform,      1, no,  gt) \
    X(usetpi_ge,                 0x9d, "usetp.ge.u32",           uniform_compare_imm, uniform,      1, no,  ge) \
    X(usetp_lt_s32,              0x9e, "usetp.lt.s32",           uniform_compare,     uniform,      1, no,  lt_s) \
    X(usetp_le_s32,              0x9f, "usetp.le.s32",           uniform_compare,     uniform,      1, no,  le_s) \
    X(usetp_gt_s32,              0xa0, "usetp.gt.s32",           uniform_compare,     uniform,      1, no,  gt_s) \
    X(usetp_ge_s32,              0xa1, "usetp.ge.s32",           uniform_compare,     uniform,      1, no,  ge_s) \
    X(usetpi_lt_s32,             0xa2, "usetp.lt.s32",           uniform_compare_imm, uniform,      1, no,  lt_s) \
    X(usetpi_le_s32,             0xa3, "usetp.le.s32",           uniform_compare_imm, uniform,      1, no,  le_s) \
    X(usetpi_gt_s32,             0xa4, "usetp.gt.s32",           uniform_compare_imm, uniform,      1, no,  gt_s) \
    X(usetpi_ge_s32,             0xa5, "usetp.ge.s32",           uniform_compare_imm, uniform,      1, no,  ge_s) \
    /* No operands, no result, no unit: only its control fields act (a wait */ \
    /* mask holds it, a stall delays the next issue). qtx's `wait` is a nop */ \
    /* that waits for every live depbar. Not 0x00: a fetch past the */ \
    /* kernel's end fills its packet with zero words. */ \
    X(nop,                       0xfe, "nop",                    no_operands,         nop,          1, no,  none) \
    X(halt,                      0xff, "halt",                   no_operands,         halt,         1, no,  none)

enum class Opcode : uint8_t {
#define NANOSM_OPCODE_ENUMERATOR(name, value, ...) name = value,
    NANOSM_OPCODES(NANOSM_OPCODE_ENUMERATOR)
#undef NANOSM_OPCODE_ENUMERATOR
};

// The mma.async block (shapes are not hardcoded).
// One opcode per shape and per init variant -- the shape is never an
// instruction field -- but which shape an opcode means is the profile's:
// shape i of its mma.shapes list is mma_block_first + 2i (accumulate:
// C += A x B) and mma_block_first + 2i + 1 (init: C = A x B). The block holds
// mma_block_shapes shapes.
inline constexpr uint8_t mma_block_first = 0x70;
inline constexpr uint32_t mma_block_shapes = 8;

// The unit that takes an instruction at issue: the machine's path through
// issue (src/machine/machine.hpp IssuePlan; the ALU, the FPU and the
// special-function unit share the arithmetic pipeline's).
enum class IssueUnit : uint8_t {
    none,                     // no instruction: an opcode the set does not have
    alu, fpu, sfu,             // the fixed-latency arithmetic pipeline (timing.alu, .fadd, .sfu)
    uniform,                   // the URF write pipeline: uniform arithmetic, r2ur, usetp
    control,                   // bra, bra.uni, ssy, sync: the issue register's branch
    fence, nop, halt,
    tma,                       // the TMA's queue
    mma,                       // the tensor core's queue: mma.async, mma.commit
    acc_load, acc_store,       // the acc.ld and acc.st engines
    shared_read, shared_write, // shared memory's read and write queues
    lsu,                       // the load / store unit's queue
};

// setp's and usetp's comparisons: unsigned 32-bit, the _s ones signed.
enum class Comparison : uint8_t { none, eq, ne, lt, le, gt, ge, lt_s, le_s, gt_s, ge_s };

// An operand's register file (either: vector, or uniform when the U bit is
// set) and how many registers it names from its field's (span:
// register_span; coordinates: tma_coordinate_count).
enum class RegisterFile : uint8_t { none, vector, uniform, predicate, either };
enum class OperandWidth : uint8_t { one, span, coordinates };

// Where the immediate sits in the payload and what it holds.
enum class ImmediateKind : uint8_t {
    none,
    s22,        // simm[22], bits 21..0
    memory,     // the per-lane memory forms: read depbar[3] (bits 21..19,
                // Control::read_depbar), simm[19] (bits 18..0)
    s14,        // simm[14], bits 13..0 (the third register field takes the high bits)
    descriptor, // a tma descriptor's shared address in 64-byte units, unsigned
                // 14 bits (1 MiB of reach); Instruction::imm holds the byte address
    spacings,   // mma.async's two row spacings, 7 bits each (mma_row_spacing below)
    word,       // umov's 32-bit word, bits 31..0, held as written (its sign is not meaning)
};

enum class VectorReads : uint8_t { at_issue, after_issue };

struct Operand {
    RegisterFile file = RegisterFile::none;
    uint8_t field = 0;  // register field 0..3; 4 is umov's 6-bit field
    OperandWidth width = OperandWidth::one;
};

enum class Form : uint8_t {
#define NANOSM_FORM_ENUMERATOR(name, ...) name,
    NANOSM_FORMS(NANOSM_FORM_ENUMERATOR)
#undef NANOSM_FORM_ENUMERATOR
};

struct FormInfo {
    std::array<Operand, 4> operands{}; // dst, src0, src1, src2
    ImmediateKind immediate = ImmediateKind::none;
    VectorReads reads = VectorReads::at_issue;
    const char* text = "";
};

struct OpcodeInfo {
    const char* mnemonic = nullptr; // none: not in the set, or the mma.async block (the profile names it)
    Form form = Form::no_operands;
    IssueUnit unit = IssueUnit::none;
    uint8_t span = 1;
    bool sets_depbar = false;
    Comparison compare = Comparison::none;
};

namespace isa_tables {

inline constexpr Operand op_no{};
inline constexpr Operand op_V0{RegisterFile::vector, 0}, op_V1{RegisterFile::vector, 1}, op_V3{RegisterFile::vector, 3};
inline constexpr Operand op_V0w{RegisterFile::vector, 0, OperandWidth::span}, op_V1w{RegisterFile::vector, 1, OperandWidth::span};
inline constexpr Operand op_U0{RegisterFile::uniform, 0}, op_U1{RegisterFile::uniform, 1}, op_U2{RegisterFile::uniform, 2};
inline constexpr Operand op_U1n{RegisterFile::uniform, 1, OperandWidth::coordinates}, op_U6{RegisterFile::uniform, 4};
inline constexpr Operand op_P0{RegisterFile::predicate, 0}, op_X2{RegisterFile::either, 2};
inline constexpr bool depbar_set = true, depbar_no = false;

#define NANOSM_COUNT_ROW(...) +1
inline constexpr std::size_t form_count = 0 NANOSM_FORMS(NANOSM_COUNT_ROW);
#undef NANOSM_COUNT_ROW

inline constexpr std::array<FormInfo, form_count> forms = [] {
    std::array<FormInfo, form_count> table{};
#define NANOSM_FORM_ROW(name, dst, src0, src1, src2, immediate, reads, text) \
    table[static_cast<std::size_t>(Form::name)] = \
        {{op_##dst, op_##src0, op_##src1, op_##src2}, ImmediateKind::immediate, VectorReads::reads, text};
    NANOSM_FORMS(NANOSM_FORM_ROW)
#undef NANOSM_FORM_ROW
    return table;
}();

// Indexed by the opcode's value. Two rows with one value do not compile
// (the throw is reached while the table is built at compile time).
inline constexpr std::array<OpcodeInfo, 256> opcodes = [] {
    std::array<OpcodeInfo, 256> table{};
    const auto row = [&table](Opcode opcode, const OpcodeInfo& info) {
        auto& entry = table[static_cast<uint8_t>(opcode)];
        if (entry.unit != IssueUnit::none) throw "two rows for one opcode value";
        entry = info;
    };
#define NANOSM_OPCODE_ROW(name, value, mnemonic, form, unit, span, depbar, compare) \
    row(Opcode::name, {mnemonic, Form::form, IssueUnit::unit, span, depbar_##depbar, Comparison::compare});
    NANOSM_OPCODES(NANOSM_OPCODE_ROW)
#undef NANOSM_OPCODE_ROW
    for (uint32_t value = mma_block_first; value < mma_block_first + 2 * mma_block_shapes; ++value) {
        row(static_cast<Opcode>(value), {nullptr, Form::mma, IssueUnit::mma, 1, true, Comparison::none});
    }
    return table;
}();

} // namespace isa_tables

inline constexpr const OpcodeInfo& opcode_info(Opcode opcode) {
    return isa_tables::opcodes[static_cast<uint8_t>(opcode)];
}

inline constexpr const FormInfo& form_info(Opcode opcode) {
    return isa_tables::forms[static_cast<std::size_t>(opcode_info(opcode).form)];
}

// An opcode the instruction set has (kernel load refuses a word with any other).
inline constexpr bool is_known_opcode(Opcode opcode) {
    return opcode_info(opcode).unit != IssueUnit::none;
}

enum class SpecialReg : uint8_t {
    lane_id = 0,
    warp_id = 1,
    thread_id = 2,
    n_warps = 3,
};

struct Timing {
    uint8_t alu = 0;
    uint8_t fadd = 0;
    // Effective SFU result latency from execution start: the unit serves
    // sfu.lanes lanes per cycle, so this is (occupancy - 1) + sfu latency.
    // Derived from the profile's sfu section at parse time.
    uint8_t sfu = 0;
};

struct Control {
    // Predicate id 7 is PT. Bit 3 of the encoded predicate field is inversion.
    uint8_t predicate = 7;
    bool predicate_invert = false;
    uint8_t stall = 0;
    uint8_t set_depbar = 7; // 0..5 are real depbars, 7 means none
    // The read depbar (has_read_depbar_field: the six vector memory forms
    // only): raised at issue with set_depbar, lowered when the instruction's
    // last vector source register has been read. Same encoding; never the
    // counter set_depbar names.
    uint8_t read_depbar = 7;
    uint8_t wait_mask = 0;   // bits 0..5
    bool operator==(const Control&) const = default;
};

struct Instruction {
    Opcode opcode{};
    Control control{};
    uint8_t dst = 0;
    uint8_t src0 = 0;
    uint8_t src1 = 0;
    uint8_t src2 = 0; // third source: fma's addend, a tma's barrier, an mma.async's result pointer
    // src1 names a uniform register, not a vector register: the one uniform
    // source of a per-lane ALU or FPU instruction (takes_uniform_source).
    bool src1_uniform = false;
    int32_t imm = 0;
    uint32_t line = 0;
    std::string text;
};

// An instruction as the machine holds it: the fields of the decoded word, a
// fixed bundle of bits. (Instruction above is the toolchain's: it adds the
// source line and text, which no register carries.) It converts to and from
// an Instruction, so the helpers that take one take this too.
struct DecodedInstruction {
    Opcode opcode{};
    Control control{};
    uint8_t dst = 0;
    uint8_t src0 = 0;
    uint8_t src1 = 0;
    uint8_t src2 = 0;
    bool src1_uniform = false;
    int32_t imm = 0;

    DecodedInstruction() = default;
    DecodedInstruction(const Instruction& inst)
        : opcode(inst.opcode), control(inst.control), dst(inst.dst), src0(inst.src0), src1(inst.src1),
          src2(inst.src2), src1_uniform(inst.src1_uniform), imm(inst.imm) {}
    operator Instruction() const {
        Instruction inst;
        inst.opcode = opcode;
        inst.control = control;
        inst.dst = dst;
        inst.src0 = src0;
        inst.src1 = src1;
        inst.src2 = src2;
        inst.src1_uniform = src1_uniform;
        inst.imm = imm;
        return inst;
    }
    bool operator==(const DecodedInstruction&) const = default;
};

// A kernel's signature (docs/qtx.md "Launch"): the `.param` lines of its qtx, in
// order. The launch passes one argument per parameter; nothing else crosses
// between the host and the kernel.
//   ptr   8 bytes: a 64-bit global byte address
//   u32   4 bytes
//   f32   4 bytes
//   desc  64 bytes: a TMA descriptor (src/tma_descriptor.hpp), built by the host
enum class ParamType : uint8_t { ptr, u32, f32, desc };

inline uint32_t param_bytes(ParamType type) {
    switch (type) {
        case ParamType::ptr: return 8;
        case ParamType::u32: return 4;
        case ParamType::f32: return 4;
        case ParamType::desc: return 64;
    }
    return 0;
}

inline const char* param_type_name(ParamType type) {
    switch (type) {
        case ParamType::ptr: return "ptr";
        case ParamType::u32: return "u32";
        case ParamType::f32: return "f32";
        case ParamType::desc: return "desc";
    }
    return "?";
}

// One named piece of what a kernel declares in shared memory beside its code
// (docs/qtx.md "Launch"), in the order the qtx declares it:
//   .param TYPE NAME           an argument of the launch (the signature)
//   .mbar  NAME, expect N      a barrier's initial state: phase 0, N arrivals
//                              expected, none seen (8 bytes)
//   .mbar  NAME[C], expect N   C such barriers side by side, ONE section:
//                              element i is at NAME + 8 i, and the whole
//                              array arrives (and is waited for) together
//   .data  NAME, f32|u32 ...   initialized words
// Together, in that order, they are the LAUNCH BLOCK (launch_block below):
// the bytes the launch copies from global memory into shared memory, the
// first declared arriving first.
enum class SectionKind : uint8_t { param, mbar, data };

// The most barriers one `.mbar NAME[C]` array may hold.
inline constexpr uint32_t max_mbar_array = 1024;

struct KernelSection {
    std::string name;
    SectionKind kind = SectionKind::param;
    ParamType type = ParamType::ptr;  // param: its type
    uint16_t expected = 0;            // mbar: arrivals expected
    uint32_t count = 1;               // mbar: how many barriers (NAME[C]); 1 otherwise
    std::vector<uint8_t> bytes;       // data: its words, little-endian
    uint32_t address = 0;             // where it sits in shared memory (launch_block)
};

inline uint32_t section_bytes(const KernelSection& section) {
    switch (section.kind) {
        case SectionKind::param: return param_bytes(section.type);
        case SectionKind::mbar: return 8 * section.count;
        case SectionKind::data: return static_cast<uint32_t>(section.bytes.size());
    }
    return 0;
}

// ptr 8, u32 / f32 4, desc 64, an mbar 8, data 4.
inline uint32_t section_alignment(const KernelSection& section) {
    switch (section.kind) {
        case SectionKind::param: return param_bytes(section.type);
        case SectionKind::mbar: return 8;
        case SectionKind::data: return 4;
    }
    return 4;
}

// The launch block's place, one fixed rule from the sections and the size of
// shared memory: the sections sit in declaration order, each aligned to its
// own alignment; the block starts at the top of shared memory minus its
// bytes, rounded down to a multiple of 64. `addresses` has one per section;
// `bytes` is the block's size (a multiple of 4; 0 with no sections). nullopt
// when it does not fit.
struct LaunchBlock {
    uint32_t start = 0;
    uint32_t bytes = 0;
    std::vector<uint32_t> addresses;
};

inline std::optional<LaunchBlock> launch_block(
    const std::vector<KernelSection>& sections, uint32_t shmem_bytes) {
    LaunchBlock block;
    uint64_t total = 0;
    for (const auto& section : sections) {
        const uint64_t alignment = section_alignment(section);
        total = (total + alignment - 1) / alignment * alignment;
        block.addresses.push_back(static_cast<uint32_t>(total));
        total += section_bytes(section);
    }
    total = (total + 3) / 4 * 4;
    if (total > shmem_bytes) return std::nullopt;
    block.bytes = static_cast<uint32_t>(total);
    block.start = (shmem_bytes - block.bytes) / 64 * 64;
    for (auto& address : block.addresses) address += block.start;
    return block;
}

// A launch flag (docs/qtx.md "Launch"): depbar `bar` starts counted once in every
// warp and is decremented in every warp when the first `bytes` bytes of the
// launch block have landed in shared memory. A kernel's flags are written by
// qtxc (the policy: default_launch_flags) and travel to the launch record;
// the machine only applies them.
struct LaunchFlag {
    uint8_t bar = 0;
    uint32_t bytes = 0;
};

// The first policy for the cuts: with `count` flags (the profile's
// core.launch_flags), one flag for each of the first count - 1 sections and
// the last flag for all the rest; fewer sections than flags leaves flags
// unused. Flags take depbars from sb5 downward. `block` is launch_block's.
inline std::vector<LaunchFlag> default_launch_flags(
    const std::vector<KernelSection>& sections, const LaunchBlock& block, uint32_t count) {
    std::vector<LaunchFlag> flags;
    if (count == 0 || sections.empty()) return flags;
    const auto used = std::min<std::size_t>(count, sections.size());
    for (std::size_t index = 0; index < used; ++index) {
        const bool last = index + 1 == used;
        const uint32_t cut = last ? block.bytes
            : block.addresses[index] - block.start + section_bytes(sections[index]);
        flags.push_back({static_cast<uint8_t>(5 - index), cut});
    }
    return flags;
}

// The flag a byte range of the block ending at offset `end` (from the block's
// start) waits for: the first one, in cut order, whose cut covers it.
inline std::optional<uint8_t> launch_flag_covering(const std::vector<LaunchFlag>& flags, uint32_t end) {
    std::optional<LaunchFlag> best;
    for (const auto& flag : flags) {
        if (flag.bytes >= end && (!best.has_value() || flag.bytes < best->bytes)) best = flag;
    }
    if (!best.has_value()) return std::nullopt;
    return best->bar;
}

// A kernel is its words, the profile they were built for, and what its qtx
// declares beside the code: how many warps it is written for (0: it does not
// say), its sections (the signature and its initialized shared memory: the
// launch block) and the launch flags over them. An accumulator is a tmem
// pointer in a uniform register (src/layout.hpp), not part of this.
// `symbols` are the qtx's names (.equ constants and section addresses) for
// a debugger or a test that looks inside; a hex kernel and a saved launch
// image do not carry them.
// `register_names`: the qtx's register names (written %name there) and the
// number qtxc gave each, for a reader, a debugger or a test; `warnings`: what
// qtxc said while compiling (a copy it inserted, say). Assembly and hex carry
// the names (`.name %x, r5` lines); a saved launch image does not, and
// nothing but the compile itself carries the warnings.
struct RegisterName {
    std::string name;       // without the %
    bool uniform = false;   // a uniform register (urN) or a vector register (rN)
    uint8_t number = 0;
    bool predicate = false; // a predicate with a name (pN); then not a register
};

// A named piece of the kernel's own memory (docs/qtx.md "Memory"; qtx `.shmem` /
// `.tmem`, placed by qtxc: src/qtxc/objects.cpp). It is the author's
// bookkeeping and nothing else: no hardware reads it, the machine never sees
// an object, only the addresses the instructions carry. Shared memory is in
// bytes, tmem in words (`tmem`). An array NAME[N] has `count` elements,
// `stride` apart; `size` is one element's. A tile has rows x cols (elements
// of `elem` bytes in shared memory, words in tmem); a plain buffer has rows
// 0. `over`: the objects it was declared to lie on, as written.
struct MemoryObject {
    std::string name;
    bool tmem = false;
    uint32_t address = 0;   // element 0
    uint32_t count = 1;
    uint32_t size = 0;      // one element
    uint32_t stride = 0;    // element i is at address + i x stride
    uint32_t alignment = 0;
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t elem = 0;
    std::vector<std::string> over;
    bool placed = false;    // its address is known (false only while a source is being read)
    uint32_t line = 0;
};

// One element's row in its own unit: cols x elem bytes (shared), cols words (tmem).
inline uint32_t object_row(const MemoryObject& object) {
    return object.tmem ? object.cols : object.cols * object.elem;
}

// From its first unit to its last, all elements. Wider than an address: a
// large array's extent is found here before anyone has checked that it fits.
inline uint64_t object_extent(const MemoryObject& object) {
    return uint64_t{object.count - 1} * object.stride + object.size;
}

struct Kernel {
    std::string profile_name;
    uint64_t profile_fingerprint = 0;
    uint16_t register_count = 0;
    std::vector<uint64_t> words;
    uint32_t warps = 0;
    std::vector<KernelSection> sections;
    std::vector<LaunchFlag> flags;
    std::vector<std::pair<std::string, int32_t>> symbols;
    std::vector<RegisterName> register_names;
    std::vector<std::string> warnings;
    // The qtx's `.shmem` / `.tmem` objects, for a debugger or a test (a hex
    // kernel and a saved launch image do not carry them).
    std::vector<MemoryObject> objects;
};

namespace encoding {

// 63                            38 37                                      0
// +-------------------------------+-----------------------------------------+
// | opcode/predicate/stall/depbars | the form's 38-bit payload               |
// +-------------------------------+-----------------------------------------+
// The payload (the forms above say what each holds): register fields 0..3
// at bits 37..30, 29..22, 21..14 and 13..6, 8 bits each (a vector, uniform or
// predicate register number: the form says which file); the U bit, bit 0
// (an X2 operand: src1 is a uniform register); the immediate in the low bits
// (ImmediateKind above). A payload bit its form does not name is reserved and
// must be zero (reserved_bit below; kernel load refuses one that is set).
// Branch immediates (bra, bra.uni, and ssy's label) are signed byte
// displacements relative to the following PC.
// stall is 5 bits so a slow FPGA-priced SFU (20+ cycles) can still be a stall count.
inline constexpr unsigned opcode_shift = 56;
inline constexpr unsigned predicate_shift = 52;
inline constexpr unsigned stall_shift = 47;
inline constexpr unsigned set_depbar_shift = 44;
inline constexpr unsigned wait_mask_shift = 38;
inline constexpr unsigned immediate_shift = 0;

inline constexpr uint64_t opcode_mask = 0xff;
inline constexpr uint64_t predicate_mask = 0x0f;
inline constexpr uint64_t stall_mask = 0x1f;
inline constexpr uint64_t set_depbar_mask = 0x07;
inline constexpr uint64_t wait_mask_mask = 0x3f;
inline constexpr uint64_t register_mask = 0xff;
inline constexpr uint64_t immediate_mask = 0x3fffff;
inline constexpr int32_t immediate_min = -(1 << 21);
inline constexpr int32_t immediate_max = (1 << 21) - 1;
inline constexpr uint64_t atomic_immediate_mask = 0x3fff;
inline constexpr uint64_t tma_descriptor_field_mask = 0x3fff;
inline constexpr uint32_t tma_descriptor_field_unit = 64;
inline constexpr int32_t atomic_immediate_min = -(1 << 13);
inline constexpr int32_t atomic_immediate_max = (1 << 13) - 1;
inline constexpr unsigned read_depbar_shift = 19;
inline constexpr uint64_t read_depbar_mask = 0x07;
inline constexpr uint64_t vector_memory_immediate_mask = 0x7ffff;
inline constexpr int32_t vector_memory_immediate_min = -(1 << 18);
inline constexpr int32_t vector_memory_immediate_max = (1 << 18) - 1;

inline constexpr unsigned umovi_dst_shift = 32;
inline constexpr uint64_t umovi_dst_mask = 0x3f;

inline constexpr unsigned reg0_shift = 30;
inline constexpr unsigned reg1_shift = 22;
inline constexpr unsigned reg2_shift = 14;
inline constexpr unsigned reg3_shift = 6; // fma's rc, in RRR reserved bits
inline constexpr unsigned uniform_source_shift = 0; // U: src1 is a uniform register

} // namespace encoding

// The per-lane ALU and FPU instructions that may take ONE source from a
// uniform register: always the second, src1 (the instruction's U bit), the
// one slot the uniform value enters the datapath by. add, and, or, xor, shl,
// shr (a uniform shift count), the register setp forms, fadd, fsub (x minus
// the uniform value), fmul, fmax, cvt.pack (the high half) and fma (a
// multiplicand: a x U + c). The uniform register is read in the instruction's
// one round beside its vector registers (src/timing.hpp); the vector sources
// are the others, and only they count for the RF banks.
inline bool takes_uniform_source(Opcode opcode) {
    return form_info(opcode).operands[2].file == RegisterFile::either;
}

// A logical MMA shape: an m x k A times a k x n B into an m x n accumulator.
struct MmaShape {
    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t k = 0;
    bool operator==(const MmaShape&) const = default;
};

inline bool is_tma_opcode(Opcode opcode) {
    return opcode >= Opcode::tma_load_1d && opcode <= Opcode::tma_store_4d;
}

inline bool is_tma_store(Opcode opcode) {
    return opcode >= Opcode::tma_store_1d && opcode <= Opcode::tma_store_4d;
}

// A tma opcode's coordinate count N (1..4).
inline uint32_t tma_coordinate_count(Opcode opcode) {
    return (static_cast<uint32_t>(opcode) - static_cast<uint32_t>(Opcode::tma_load_1d)) % 4 + 1;
}

inline Opcode tma_opcode(bool store, uint32_t coordinates) {
    return static_cast<Opcode>(static_cast<uint32_t>(Opcode::tma_load_1d) + (store ? 4 : 0) + coordinates - 1);
}

inline bool is_mma_async(Opcode opcode) {
    const auto value = static_cast<uint8_t>(opcode);
    return value >= mma_block_first && value < mma_block_first + 2 * mma_block_shapes;
}

inline bool is_mma_init(Opcode opcode) {
    return is_mma_async(opcode) && ((static_cast<uint8_t>(opcode) - mma_block_first) & 1u) != 0;
}

// Which entry of the profile's shape list an mma.async opcode names.
inline uint32_t mma_shape_index(Opcode opcode) {
    return (static_cast<uint8_t>(opcode) - mma_block_first) / 2u;
}

inline Opcode mma_opcode(uint32_t shape_index, bool init) {
    return static_cast<Opcode>(mma_block_first + 2 * shape_index + (init ? 1 : 0));
}

// An mma.async operand is a uniform register holding the shared address of
// the 16-byte piece it starts at in its row 0 (any row of a taller tile),
// and a row spacing, an instruction constant (src/layout.hpp).
// The spacings sit in the 14 bits beside the three register fields: A's in
// bits 7..13, B's in bits 0..6, each in 16-byte units, so 16 .. 2032 bytes
// (swizzle64 takes the powers of two among them: 16 .. 1024).
// The assembly operand is [urA] (spacing 16: one 16-byte piece per row, a
// K = 8 tile) or [urA, spacing].
inline constexpr uint32_t mma_spacing_unit = 16;
inline constexpr uint32_t mma_spacing_field_bits = 7;
inline constexpr uint32_t mma_spacing_field_mask = (1u << mma_spacing_field_bits) - 1;
inline constexpr uint32_t mma_default_spacing = 16;
inline constexpr uint32_t mma_max_spacing = mma_spacing_field_mask * mma_spacing_unit;

inline uint32_t mma_row_spacing(const Instruction& instruction, bool a_operand) {
    const auto fields = static_cast<uint32_t>(instruction.imm);
    return ((fields >> (a_operand ? mma_spacing_field_bits : 0)) & mma_spacing_field_mask) * mma_spacing_unit;
}

// The immediate holding spacings `a` and `b` (bytes, multiples of 16 up to
// mma_max_spacing; the caller checks).
inline int32_t mma_spacing_immediate(uint32_t a, uint32_t b) {
    return static_cast<int32_t>((a / mma_spacing_unit) << mma_spacing_field_bits | (b / mma_spacing_unit));
}

// "m64n64k8": the mnemonic suffix, also the profile's spelling of a shape.
std::string mma_shape_name(const MmaShape& shape);
// The inverse: "m32n64k8" -> {32, 64, 8}; nullopt unless the text is exactly
// m<digits>n<digits>k<digits> with nonzero values.
std::optional<MmaShape> parse_mma_shape_name(const std::string& name);

// A bra with a predicate (anything but plain PT) is the divergent branch;
// plain bra moves every active lane and bra.uni is a promise about the whole
// warp.
inline bool is_divergent_branch(const Instruction& instruction) {
    return instruction.opcode == Opcode::bra &&
        (instruction.control.predicate != 7 || instruction.control.predicate_invert);
}

// The instruction word's predicate field names something (anything but plain
// PT). On bra it makes the divergent branch, on bra.uni it is the branch's
// condition; on every other instruction it is the instruction's write mask,
// which makes it a masked instruction.
inline bool has_write_mask(const Instruction& instruction) {
    return instruction.control.predicate != 7 || instruction.control.predicate_invert;
}

// The opcodes each fixed-latency unit executes: the ALU (timing.alu), the FPU
// (timing.fadd), the special-function unit (timing.sfu); is_arithmetic, any
// of the three: the fixed-latency arithmetic pipeline's instructions.
inline bool is_alu_operation(Opcode opcode) { return opcode_info(opcode).unit == IssueUnit::alu; }
inline bool is_fpu_operation(Opcode opcode) { return opcode_info(opcode).unit == IssueUnit::fpu; }
inline bool is_sfu_operation(Opcode opcode) { return opcode_info(opcode).unit == IssueUnit::sfu; }
inline bool is_arithmetic(Opcode opcode) {
    return is_alu_operation(opcode) || is_fpu_operation(opcode) || is_sfu_operation(opcode);
}

// The forms whose word has the read depbar field and the 19-bit immediate:
// the per-lane memory instructions lds, sts, ldg, stg, ldg.v8, stg.v8.
inline bool has_read_depbar_field(Opcode opcode) {
    return form_info(opcode).immediate == ImmediateKind::memory;
}

// The instructions that can be masked (@p / @!p): the per-lane ones, which
// already write only the divergence stack's active lanes -- every instruction
// of the fixed arithmetic pipeline (ALU, FPU, SFU, cvt.pack, mov, sread, ur2r,
// setp) and the per-lane memory instructions lds, sts, ldg, stg, ldg.v8, stg.v8. The
// operation runs as usual and its result is written with a mask: the lanes
// the instruction writes (or, for sts / stg, sends) are the active mask AND
// its write mask, the predicate's lanes (inverted for @!p), sampled with the
// operands at issue + 1. @!PT turns every lane off. Fixed timing ignores the
// write mask. Everything else (uniform arithmetic, usetp, r2ur, the
// warp-wide memory and engine operations, mbar, atomics, fence, nop, halt,
// ssy, sync) refuses to be masked at qtxc, the assembler and kernel load; bra
// and bra.uni keep their own meaning of the predicate.
inline bool can_be_masked(Opcode opcode) {
    return is_arithmetic(opcode) || has_read_depbar_field(opcode);
}

// The instructions that set a depbar: the shared and global memory
// accesses, the atomics, the accumulator transfers and mma.async.
inline bool sets_depbar(Opcode opcode) {
    return opcode_info(opcode).sets_depbar;
}

// The instructions that use the divergence stack: refused by qtxc, the
// assembler and kernel load when the profile has none
// (core.divergence_stack_entries = 0).
inline bool uses_divergence_stack(const Instruction& instruction) {
    return instruction.opcode == Opcode::ssy || instruction.opcode == Opcode::sync ||
        is_divergent_branch(instruction);
}

// The uniform shared atomics: atom.add/and/xor/exch.uni.
inline bool is_atomic(Opcode opcode) {
    return form_info(opcode).immediate == ImmediateKind::s14;
}

// The barrier instructions by what they do: a test_wait or try_wait probes a
// barrier and writes a predicate; init, arrive and arrive.expect_tx change
// the barrier through the read-modify-write engine.
inline bool is_mbar_test_wait(Opcode opcode) {
    return opcode == Opcode::mbar_test_wait_i || opcode == Opcode::mbar_test_wait_r;
}

inline bool is_mbar_try_wait(Opcode opcode) {
    return opcode == Opcode::mbar_try_wait_i || opcode == Opcode::mbar_try_wait_r;
}

inline bool is_mbar_predicate_writer(Opcode opcode) {
    return is_mbar_test_wait(opcode) || is_mbar_try_wait(opcode);
}

inline bool is_mbar_rmw(Opcode opcode) {
    return opcode >= Opcode::mbar_init_i && opcode <= Opcode::mbar_arrive_expect_tx_r;
}

// The loads that write a vector register from shared memory (lds, lds.uni)
// and from tensor memory (acc.ld, acc.ld.v8).
inline bool is_shared_load(Opcode opcode) {
    return opcode == Opcode::lds || opcode == Opcode::lds_uni;
}

inline bool is_acc_load(Opcode opcode) {
    return opcode_info(opcode).unit == IssueUnit::acc_load;
}

// Uniform arithmetic: the uniform ALU's opcodes (umov .. usread).
inline bool is_uniform_arithmetic(Opcode opcode) {
    return opcode >= Opcode::umov && opcode <= Opcode::usread;
}

// The writers of the uniform register file: uniform arithmetic and r2ur, and
// nothing else (no loads into it). Every one presents its result at the URF
// write port urf.issue_to_port_cycles after its issue (src/timing.hpp).
inline bool writes_uniform_register(Opcode opcode) {
    return form_info(opcode).operands[0].file == RegisterFile::uniform;
}

// The vector compare, register and immediate forms, unsigned and signed.
inline bool is_setp(Opcode opcode) {
    const auto form = opcode_info(opcode).form;
    return form == Form::vector_compare || form == Form::vector_compare_imm;
}

inline bool is_setp_immediate(Opcode opcode) {
    return opcode_info(opcode).form == Form::vector_compare_imm;
}

// The uniform compare, register and immediate forms, unsigned and signed.
inline bool is_usetp(Opcode opcode) {
    const auto form = opcode_info(opcode).form;
    return form == Form::uniform_compare || form == Form::uniform_compare_imm;
}

inline bool is_usetp_immediate(Opcode opcode) {
    return opcode_info(opcode).form == Form::uniform_compare_imm;
}

// What the URF write pipeline carries (src/timing.hpp): the uniform-register
// writers, and usetp, whose uniform-ALU result leaves it at the write port's
// stage for the predicate file instead of the URF.
inline bool uses_urf_pipeline(Opcode opcode) {
    return opcode_info(opcode).unit == IssueUnit::uniform;
}

// How many registers a register operand of this opcode names from the one
// its field holds: eight for the v8 forms' data registers (ldg.v8's and
// acc.ld.v8's destination, stg.v8's and acc.st.v8's data), one otherwise.
inline uint32_t register_span(Opcode opcode) {
    return opcode_info(opcode).span;
}

// The uniform registers an instruction reads, in operand order (a register
// named twice is listed twice). Most read at most two, all in one read round
// at issue + 1 (src/timing.hpp; the round reads a repeated register once): the
// URF has two read ports. acc.ld / acc.st read their tmem pointer, an ALU or
// FPU instruction with its U bit its src1 (takes_uniform_source). Two read
// more over several rounds, two per round, listed in read order and read by
// operand position (a register named twice is read twice;
// uniform_read_rounds in src/timing.hpp): a tma, N + 2 registers (urS, urM, then
// urC .. urC + N - 1), and an mma.async, three (urA, urB, then urC).
std::vector<uint8_t> uniform_sources(const Instruction& instruction);
// The uniform register an instruction writes, if any.
std::optional<uint8_t> uniform_destination(const Instruction& instruction);
// The vector registers an instruction reads, in operand order, a v8 form's
// data registers each (its uniform registers: uniform_sources). A uniform
// second source (the U bit) is not one of them.
std::vector<uint8_t> vector_sources(const Instruction& instruction);
// Those its issue round reads: all of them but the ones read after issue --
// the shared and global memory forms' (lds, sts, ldg, stg, ldg.v8, stg.v8:
// their collect stage or the load / store unit's command slot and data beats)
// and acc.st's data (beat by beat).
std::vector<uint8_t> issue_sources(const Instruction& instruction);
// The vector register an instruction writes, if any (the first of a v8
// form's eight: register_span).
std::optional<uint8_t> vector_destination(const Instruction& instruction);
// The predicate an instruction writes, if any: setp, usetp, the mbar probes.
std::optional<uint8_t> predicate_destination(const Instruction& instruction);

uint64_t encode(const Instruction& instruction);
Instruction decode(uint64_t word);
// The lowest bit of `word` that its form does not define, if any: a bit that
// encoding what decode reads from the word does not write back (so a form
// defines exactly what encode writes for it; an opcode the set does not have
// defines only the opcode and control fields). A reserved bit must be zero:
// kernel load refuses a word with one set (src/qtxc/checks.cpp), so a word
// that gives one a meaning is never read as if it were not there.
std::optional<unsigned> reserved_bit(uint64_t word);
// Names. An mma.async opcode's shape is the profile's: given its shape list
// (`mma_shapes`), it is named "mma.async[.init].m64n64k8"; without one, or
// past the list's end, by its place in the block ("mma.async.slot0").
std::string format_instruction(const Instruction& instruction, bool include_control = true,
    const std::vector<MmaShape>* mma_shapes = nullptr);
std::string format_control(const Control& control);
std::string opcode_name(Opcode opcode, const std::vector<MmaShape>* mma_shapes = nullptr);
// The first opcode (by value) whose mnemonic this is; one that shares it has
// another form (setp's register and immediate forms, say). Not the mma.async
// block's, which the profile names.
std::optional<Opcode> opcode_for_mnemonic(std::string_view mnemonic);
// The fixed latency of an instruction of the arithmetic pipeline (its unit's)
// and of mbar.test_wait (one cycle: the probe's predicate); 0 for the rest.
uint8_t fixed_latency(const Instruction& instruction, const Timing& timing);

} // namespace nanosm
