#pragma once

#include <algorithm>
#include <optional>
#include <stdint.h>
#include <string>
#include <utility>
#include <vector>

namespace nanosm {

enum class Opcode : uint8_t {
    sread = 0x01,
    mov   = 0x02,
    add   = 0x03,
    addi  = 0x04,
    shli  = 0x05,
    fadd  = 0x06,
    xori  = 0x07,
    setp_eq = 0x08,
    setp_ne = 0x09,
    setp_lt = 0x0a,
    setp_le = 0x0b,
    setp_gt = 0x0c,
    setp_ge = 0x0d,
    fmul  = 0x0e,
    fsub  = 0x0f,
    lds   = 0x10,
    sts   = 0x11,
    ldg   = 0x12,
    stg   = 0x13,
    fmax  = 0x14,
    fma   = 0x15,
    cvt_pack = 0x16, // cvt.pack.f16x2: rd = half(ra) | half(rb) << 16
    ex2   = 0x17,    // SFU: 2^x
    rcp   = 0x18,    // SFU: 1/x
    and_  = 0x19,    // trailing underscores dodge the C++ operator tokens
    or_   = 0x1a,
    xor_  = 0x1b,
    shl   = 0x1c,    // register shifts use the low 5 bits: a 5-bit shifter
    shr   = 0x1d,    // logical
    andi  = 0x1e,
    ori   = 0x1f,
    bra   = 0x20,
    bra_uni = 0x21,
    shri  = 0x22,    // logical; 0x19..0x1f were full
    // Interleaved vector (rfc09): rA warp-uniform, 64 B aligned, 512 contiguous
    // bytes; register rD+j, lane l <-> word 16*j + l.
    ldg_v8 = 0x23,
    stg_v8 = 0x24,
    ldg_uni = 0x25,  // base in a uniform register: one sector request, the word fans out to every lane
    lds_uni = 0x26,  // base in a uniform register: one shmem bank access, the word fans out to every lane
    // No operands (trash/rfc12.md section 10): admitted only once every earlier
    // shmem access of the warp has reached the RAM (stores written, loads
    // sampled), so nothing after it issues before that. Never auto-inserted.
    fence_sh = 0x27,

    setpi_eq = 0x28,
    setpi_ne = 0x29,
    setpi_lt = 0x2a,
    setpi_le = 0x2b,
    setpi_gt = 0x2c,
    setpi_ge = 0x2d,
    // Divergence (trash/divergence-2026-11-21.md), written by the kernel
    // author, never inferred. Every divergence stack entry is a (mask, pc)
    // pair. ssy LABEL (a branch displacement like bra's; any instruction)
    // pushes (the warp's active mask, LABEL). A predicated bra (0x20 with a
    // predicate) is the divergent branch: taken = active AND p, fall =
    // active AND NOT p; one side empty moves the whole mask, both non-empty
    // push (taken, target) and run the fall-through arm under fall. sync
    // (no operands) pops the top entry and switches to it, whoever pushed
    // it: the mask becomes the entry's, the warp goes to the entry's pc.
    ssy = 0x2e,
    sync = 0x2f,
    mbar_init_i = 0x30,
    mbar_init_r = 0x31,
    mbar_arrive = 0x32,
    mbar_arrive_expect_tx_i = 0x33,
    mbar_arrive_expect_tx_r = 0x34,
    mbar_test_wait_i = 0x35,
    mbar_test_wait_r = 0x36,
    mbar_try_wait_i = 0x37,
    mbar_try_wait_r = 0x38,
    // Uniform shared-memory atomics on signed int32 (trash/rfc13.md):
    // atom.op.uni rD, [urA + imm], urB. ONE 32-bit word per warp instruction:
    // the address (urA + imm) and the operand (urB) are uniform registers, the
    // update is applied once, and every active lane's rD receives the word's old
    // value. add wraps (32-bit two's complement); exch writes the operand. An
    // mbar-engine operation: the global store pause covers its one read, the
    // ALU's C stages and its one write (D + Lb + C + 1 cycles canonical: 7).
    atom_add_uni = 0x39,
    atom_and_uni = 0x3a,
    atom_xor_uni = 0x3b,
    atom_exch_uni = 0x3c,
    // tma.load.Nd [urS], [DESC, urC], [urM] and tma.store.Nd [DESC, urC],
    // [urS], [urM], N = 1..4 (src/tma.hpp, trash/rfc17.md): the direction and
    // the coordinate count are the opcode; urS the buffer's shared address,
    // urC .. urC + N - 1 the tile's coordinates, urM the barrier's address,
    // DESC the descriptor's shared address (a constant).
    tma_load_1d = 0x40,
    tma_load_2d = 0x41,
    tma_load_3d = 0x42,
    tma_load_4d = 0x43,
    tma_store_1d = 0x44,
    tma_store_2d = 0x45,
    tma_store_3d = 0x46,
    tma_store_4d = 0x47,
    // The signed vector compare (trash/signed-compare-2026-11-18.md):
    // setp.xx.s32 pD, rA, rB | imm for lt, le, gt, ge compares as signed
    // 32-bit; everything else is setp.xx.u32's (encoding, timing, predicate
    // write). eq and ne need no signed form. The immediate is the signed
    // 22-bit field, sign-extended to 32 bits, as for setp.xx.u32 (whose
    // compare is then unsigned).
    setp_lt_s32 = 0x48,
    setp_le_s32 = 0x49,
    setp_gt_s32 = 0x4a,
    setp_ge_s32 = 0x4b,
    setpi_lt_s32 = 0x4c,
    setpi_le_s32 = 0x4d,
    setpi_gt_s32 = 0x4e,
    setpi_ge_s32 = 0x4f,
    acc_ld = 0x50,
    acc_ld_v8 = 0x51,
    acc_st = 0x52,
    acc_st_v8 = 0x53,
    mma_commit = 0x62,
    // 0x70..0x7f: the mma.async block (mma_block_first below). The profile's
    // shape list says what each of these opcodes means.

    // The uniform register file (trash/rfc16.md, trash/urf-slice-2026-11-08.md):
    // urf.registers 32-bit registers per warp (canonical 32, ur0..ur31). The
    // uniform arithmetic below runs on the SM's one integer uniform ALU, one
    // value per warp instruction; register operands are uniform registers.
    // Shifts by a register use its low 5 bits; immediates are the signed
    // 22-bit field.
    umov   = 0x80,   // umov urD, urA
    umovi  = 0x81,   // umov urD, imm: any 32-bit word (encoding below)
    uadd   = 0x82,
    uaddi  = 0x83,
    usub   = 0x84,
    ushl   = 0x85,
    ushr   = 0x86,   // logical
    ushli  = 0x87,
    ushri  = 0x88,   // logical
    uand   = 0x89,
    uor    = 0x8a,
    uxor   = 0x8b,
    uandi  = 0x8c,
    uori   = 0x8d,
    uxori  = 0x8e,
    usread = 0x8f,   // usread urD, WARP_ID / N_WARPS: a warp-uniform special register
    // Moves between the files: r2ur urD, rS takes lane 0 of a vector register;
    // ur2r rD, urS writes a uniform register into every lane of a vector
    // register (a fixed-latency vector result, like mov).
    r2ur   = 0x90,
    ur2r   = 0x91,
    // The uniform compare (trash/uniform-compare-2026-11-18.md): usetp.xx.u32
    // pD, urA, urB and usetp.xx.u32 pD, urA, imm, the vector setp's six
    // comparisons (unsigned 32-bit) on uniform registers, and their signed lt,
    // le, gt, ge (usetp.xx.s32, 0x9e..0xa5). It runs on the uniform ALU (a
    // compare is a subtract) and writes predicate pD with the same value in
    // every lane; register and immediate forms have distinct opcodes, as
    // setp's do. Its timing: src/rf.hpp issue_to_uniform_predicate_visible.
    usetp_eq = 0x92,
    usetp_ne = 0x93,
    usetp_lt = 0x94,
    usetp_le = 0x95,
    usetp_gt = 0x96,
    usetp_ge = 0x97,
    usetpi_eq = 0x98,
    usetpi_ne = 0x99,
    usetpi_lt = 0x9a,
    usetpi_le = 0x9b,
    usetpi_gt = 0x9c,
    usetpi_ge = 0x9d,
    usetp_lt_s32 = 0x9e,
    usetp_le_s32 = 0x9f,
    usetp_gt_s32 = 0xa0,
    usetp_ge_s32 = 0xa1,
    usetpi_lt_s32 = 0xa2,
    usetpi_le_s32 = 0xa3,
    usetpi_gt_s32 = 0xa4,
    usetpi_ge_s32 = 0xa5,
    // No operands, no result, no unit (trash/typed-compare-and-wait-2026-11-18.md):
    // it issues like any instruction and only its control fields act -- a wait
    // mask holds it until those depbars are zero, a stall delays the next
    // issue. qtx's `wait` is a nop that waits for every live depbar. Not 0x00:
    // a fetch past the kernel's end fills its packet with zero words.
    nop   = 0xfe,
    halt  = 0xff,
};

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
    uint8_t wait_mask = 0;   // bits 0..5
};

struct Instruction {
    Opcode opcode{};
    Control control{};
    uint8_t dst = 0;
    uint8_t src0 = 0;
    uint8_t src1 = 0;
    uint8_t src2 = 0; // third source; only fma uses it (RRR reserved bits)
    int32_t imm = 0;
    uint32_t line = 0;
    std::string text;
};

// A kernel's signature (README "Launch"): the `.param` lines of its qtx, in
// order. The launch passes one argument per parameter; nothing else crosses
// between the host and the kernel.
//   ptr   8 bytes: a 64-bit global byte address
//   u32   4 bytes
//   f32   4 bytes
//   desc  64 bytes: a TMA descriptor (src/tma.hpp), built by the host
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
// (README "Launch"), in the order the qtx declares it:
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

// A launch flag (README "Launch"): depbar `bar` starts counted once in every
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

// A named piece of the kernel's own memory (README "Memory"; qtx `.shmem` /
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
// | opcode/predicate/stall/depbars | opcode-selected 38-bit payload          |
// +-------------------------------+-----------------------------------------+
// RRR payload: dst[8], src0[8], src1[8], reserved[14]
// RRI payload: dst[8], src0[8], simm[22]
// store payload: base[8], value[8], simm[22]
// RI payload: dst[8], reserved[8], simm[22]
// umov urD, imm (umovi) payload: dst[6], imm[32] -- a uniform register number
// needs at most 6 bits, which leaves a whole 32-bit word: any constant, a
// float's bits included, in one instruction and no memory. Instruction::imm
// holds the 32 bits as written (its sign is not meaning).
// atomic payload: dst[8], base[8], operand[8], simm[14] (the offset is 14 bits:
// the third register takes the immediate's high bits)
// setp and usetp use dst as p0..p6; immediate and register RHS forms have
// distinct opcodes (usetp's registers are uniform registers).
// Branch immediates (bra, bra.uni, and ssy's label) are signed byte
// displacements relative to the following PC; sync has no payload.
// mbar addresses are held in a uniform register; mbar immediates and register
// operands have distinct opcodes, as do setp immediates and register operands.
// A uniform register (ur0..) sits in the same 8-bit register fields: which
// field names which file is the opcode's (uniform_sources / uniform_destination;
// trash/urf-slice-2026-11-08.md lists every operand form).
// tma payload: buffer[8] (urS), coordinates[8] (urC, the first of N),
// barrier[8] (urM), descriptor[14] (its shared address in 64-byte units,
// unsigned: 1 MiB of reach). Instruction: src0 urS, src1 urC, src2 urM, imm
// the descriptor's byte address.
// mma.async payload: C[8] (urC, the result's tmem pointer), A[8] (urA),
// B[8] (urB), A's row spacing[7], B's row spacing[7] (16-byte units).
// Instruction: src2 urC, src0 urA, src1 urB, imm the two spacing fields as
// encoded (mma_row_spacing below reads them).
// acc.ld / acc.ld.v8 rD, [urT, S]: the RRI payload, dst rD, src0 urT, simm
// the region's words per row S. acc.st / acc.st.v8 [urT, S], rS: the store
// payload, base urT, value rS, simm S (src/layout.hpp: a power of two, 4 or
// more, at most tmem's words).
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

inline constexpr unsigned umovi_dst_shift = 32;
inline constexpr uint64_t umovi_dst_mask = 0x3f;

inline constexpr unsigned reg0_shift = 30;
inline constexpr unsigned reg1_shift = 22;
inline constexpr unsigned reg2_shift = 14;
inline constexpr unsigned reg3_shift = 6; // fma's rc, in RRR reserved bits

} // namespace encoding

// A logical MMA shape: an m x k A times a k x n B into an m x n accumulator.
struct MmaShape {
    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t k = 0;
    bool operator==(const MmaShape&) const = default;
};

// The mma.async block (trash/chan.txt 2026-10-22: shapes are not hardcoded).
// One opcode per shape and per init variant -- the shape is never an
// instruction field -- but which shape an opcode means is the profile's:
// shape i of its mma.shapes list is mma_block_first + 2i (accumulate:
// C += A x B) and mma_block_first + 2i + 1 (init: C = A x B). The block holds
// mma_block_shapes shapes.
inline constexpr uint8_t mma_block_first = 0x70;
inline constexpr uint32_t mma_block_shapes = 8;

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
// and a row spacing, an instruction constant (src/layout.hpp;
// trash/whole-rows-2026-11-18.md, trash/address-keyed-2026-11-18.md).
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
// which makes it a masked instruction. (The note that built this is
// trash/guards-2026-11-21.md; it uses an older word for the same thing.)
inline bool has_write_mask(const Instruction& instruction) {
    return instruction.control.predicate != 7 || instruction.control.predicate_invert;
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
bool can_be_masked(Opcode opcode);

// The instructions that use the divergence stack: refused by qtxc, the
// assembler and kernel load when the profile has none
// (core.divergence_stack_entries = 0).
inline bool uses_divergence_stack(const Instruction& instruction) {
    return instruction.opcode == Opcode::ssy || instruction.opcode == Opcode::sync ||
        is_divergent_branch(instruction);
}

// The uniform shared atomics (trash/rfc13.md): atom.add/and/xor/exch.uni.
inline bool is_atomic(Opcode opcode) {
    return opcode == Opcode::atom_add_uni || opcode == Opcode::atom_and_uni ||
        opcode == Opcode::atom_xor_uni || opcode == Opcode::atom_exch_uni;
}

// Uniform arithmetic: the uniform ALU's opcodes (umov .. usread).
inline bool is_uniform_arithmetic(Opcode opcode) {
    return opcode >= Opcode::umov && opcode <= Opcode::usread;
}

// The writers of the uniform register file: uniform arithmetic and r2ur, and
// nothing else (no loads into it). Every one presents its result at the URF
// write port urf.issue_to_port_cycles after its issue (src/rf.hpp).
inline bool writes_uniform_register(Opcode opcode) {
    return is_uniform_arithmetic(opcode) || opcode == Opcode::r2ur;
}

// The vector compare, register and immediate forms, unsigned and signed.
inline bool is_setp(Opcode opcode) {
    return (opcode >= Opcode::setp_eq && opcode <= Opcode::setp_ge) ||
        (opcode >= Opcode::setpi_eq && opcode <= Opcode::setpi_ge) ||
        (opcode >= Opcode::setp_lt_s32 && opcode <= Opcode::setpi_ge_s32);
}

inline bool is_setp_immediate(Opcode opcode) {
    return (opcode >= Opcode::setpi_eq && opcode <= Opcode::setpi_ge) ||
        (opcode >= Opcode::setpi_lt_s32 && opcode <= Opcode::setpi_ge_s32);
}

// The uniform compare, register and immediate forms, unsigned and signed.
inline bool is_usetp(Opcode opcode) {
    return (opcode >= Opcode::usetp_eq && opcode <= Opcode::usetpi_ge) ||
        (opcode >= Opcode::usetp_lt_s32 && opcode <= Opcode::usetpi_ge_s32);
}

inline bool is_usetp_immediate(Opcode opcode) {
    return (opcode >= Opcode::usetpi_eq && opcode <= Opcode::usetpi_ge) ||
        (opcode >= Opcode::usetpi_lt_s32 && opcode <= Opcode::usetpi_ge_s32);
}

// What the URF write pipeline carries (src/rf.hpp): the uniform-register
// writers, and usetp, whose uniform-ALU result leaves it at the write port's
// stage for the predicate file instead of the URF.
inline bool uses_urf_pipeline(Opcode opcode) {
    return writes_uniform_register(opcode) || is_usetp(opcode);
}

// The uniform registers an instruction reads, in operand order (a register
// named twice is listed twice). Most read at most two, all in one read round
// at issue + 1 (src/rf.hpp; the round reads a repeated register once): the
// URF has two read ports. acc.ld / acc.st read their tmem pointer. Two read
// more over several rounds, two per round, listed in read order and read by
// operand position (a register named twice is read twice;
// uniform_read_rounds in src/rf.hpp): a tma, N + 2 registers (urS, urM, then
// urC .. urC + N - 1), and an mma.async, three (urA, urB, then urC).
std::vector<uint8_t> uniform_sources(const Instruction& instruction);
// The uniform register an instruction writes, if any.
std::optional<uint8_t> uniform_destination(const Instruction& instruction);

uint64_t encode(const Instruction& instruction);
Instruction decode(uint64_t word);
// Names. An mma.async opcode's shape is the profile's: given its shape list
// (`mma_shapes`), it is named "mma.async[.init].m64n64k8"; without one, or
// past the list's end, by its place in the block ("mma.async.slot0").
std::string format_instruction(const Instruction& instruction, bool include_control = true,
    const std::vector<MmaShape>* mma_shapes = nullptr);
std::string format_control(const Control& control);
std::string opcode_name(Opcode opcode, const std::vector<MmaShape>* mma_shapes = nullptr);
uint8_t fixed_latency(const Instruction& instruction, const Timing& timing);

} // namespace nanosm
