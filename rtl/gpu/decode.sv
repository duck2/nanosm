`timescale 1ns / 1ps

// Instruction decoder: combinational decode of 32-bit instruction word.

module decode (
    input  logic [31:0] instr,

    // Format
    output logic [2:0] fmt,

    // Predicate
    output logic [2:0] pred_reg,      // 0-6 = p0-p6, 7 = always
    output logic       pred_neg,

    // Register addresses
    output logic [4:0] rd,
    output logic [4:0] rs1,
    output logic [4:0] rs2,
    output logic [4:0] rs3,

    // Opcodes (active depends on fmt)
    output logic [4:0] op_r4,         // R4 format
    output logic [2:0] op_i,          // I format
    output logic [2:0] op_m,          // M format
    output logic [3:0] op_p,          // P format
    output logic [2:0] op_j,          // J format
    output logic [3:0] op_x,          // X format

    // Immediates
    output logic [11:0] imm12,        // I/M format
    output logic [19:0] imm20,        // U format
    output logic [21:0] imm22,        // J format (branch target)

    // P-format extras
    output logic [2:0] cmp,           // setp comparison
    output logic       cmp_unsigned,  // setp type (0=signed, 1=unsigned)
    output logic [2:0] pd,            // dest predicate

    // Decoded instruction class
    output logic is_alu,
    output logic is_alu_imm,
    output logic is_shift,
    output logic is_shift_imm,
    output logic is_fpu,
    output logic is_mem,
    output logic is_load,
    output logic is_store,
    output logic is_shmem,            // lds/sts vs ldg/stg
    output logic is_lui,
    output logic is_setp,
    output logic is_pred_logic,
    output logic is_branch,
    output logic is_ssy,
    output logic is_sync,
    output logic is_sread,
    output logic is_halt,
    output logic is_nop,

    output logic writes_rd            // instruction writes to rd
);

    // ========================================================================
    // Format IDs
    // ========================================================================
    localparam [2:0] FMT_R4 = 3'd0;
    localparam [2:0] FMT_I  = 3'd1;
    localparam [2:0] FMT_M  = 3'd2;
    // 3'd3 reserved (was memory 2D)
    localparam [2:0] FMT_U  = 3'd4;
    localparam [2:0] FMT_P  = 3'd5;
    localparam [2:0] FMT_J  = 3'd6;
    localparam [2:0] FMT_X  = 3'd7;

    // ========================================================================
    // Field extraction
    // ========================================================================
    assign fmt      = instr[31:29];
    assign pred_reg = instr[27:25];
    assign pred_neg = instr[28];

    assign rd  = instr[24:20];
    assign rs1 = instr[19:15];
    assign rs2 = instr[14:10];
    assign rs3 = instr[9:5];

    assign op_r4 = instr[4:0];
    assign op_i  = instr[2:0];
    assign op_m  = instr[2:0];
    assign op_p  = instr[3:0];
    assign op_j  = instr[2:0];
    assign op_x  = instr[3:0];

    assign imm12 = instr[14:3];
    assign imm20 = instr[19:0];
    assign imm22 = instr[24:3];

    // P-format
    assign pd           = instr[24:22];
    assign cmp          = instr[11:9];
    assign cmp_unsigned = instr[8];

    // ========================================================================
    // Instruction class decode
    // ========================================================================
    wire fmt_r4 = (fmt == FMT_R4);
    wire fmt_i  = (fmt == FMT_I);
    wire fmt_m  = (fmt == FMT_M);
    wire fmt_u  = (fmt == FMT_U);
    wire fmt_p  = (fmt == FMT_P);
    wire fmt_j  = (fmt == FMT_J);
    wire fmt_x  = (fmt == FMT_X);

    // R4 opcode ranges
    wire r4_is_alu   = fmt_r4 && (op_r4 <= 5'd6);
    wire r4_is_shift = fmt_r4 && (op_r4 >= 5'd8) && (op_r4 <= 5'd10);
    wire r4_is_fpu   = fmt_r4 && (op_r4 >= 5'd16);

    // I opcode ranges
    wire i_is_alu   = fmt_i && (op_i <= 3'd4);
    wire i_is_shift = fmt_i && (op_i >= 3'd5);

    // M opcodes
    wire m_is_ldg = fmt_m && (op_m == 3'd0);
    wire m_is_lds = fmt_m && (op_m == 3'd1);
    wire m_is_stg = fmt_m && (op_m == 3'd2);
    wire m_is_sts = fmt_m && (op_m == 3'd3);

    // P opcodes
    wire p_is_setp = fmt_p && (op_p == 4'd0);
    wire p_is_logic = fmt_p && (op_p >= 4'd1) && (op_p <= 4'd3);

    // J opcodes
    wire j_is_bra  = fmt_j && (op_j == 3'd0);
    wire j_is_ssy  = fmt_j && (op_j == 3'd1);
    wire j_is_sync = fmt_j && (op_j == 3'd2);

    // X opcodes
    wire x_is_sread = fmt_x && (op_x == 4'd0);
    wire x_is_nop   = fmt_x && (op_x == 4'd1);
    wire x_is_halt  = fmt_x && (op_x == 4'd2);

    // ========================================================================
    // Output assignments
    // ========================================================================
    assign is_alu        = r4_is_alu;
    assign is_alu_imm    = i_is_alu;
    assign is_shift      = r4_is_shift;
    assign is_shift_imm  = i_is_shift;
    assign is_fpu        = r4_is_fpu;
    assign is_mem        = fmt_m;
    assign is_load       = m_is_ldg || m_is_lds;
    assign is_store      = m_is_stg || m_is_sts;
    assign is_shmem      = m_is_lds || m_is_sts;
    assign is_lui        = fmt_u;
    assign is_setp       = p_is_setp;
    assign is_pred_logic = p_is_logic;
    assign is_branch     = j_is_bra;
    assign is_ssy        = j_is_ssy;
    assign is_sync       = j_is_sync;
    assign is_sread      = x_is_sread;
    assign is_halt       = x_is_halt;
    assign is_nop        = x_is_nop;

    // Does this instruction write to rd?
    assign writes_rd = is_alu || is_alu_imm || is_shift || is_shift_imm ||
                       is_fpu || is_load || is_lui || is_sread;

    // ========================================================================
    // Simulation-only: fatal on unimplemented instructions
    // ========================================================================
    // synthesis translate_off
    wire is_global_mem = m_is_ldg || m_is_stg;

    always_comb begin
        if (is_global_mem) $fatal(1, "decode: global memory (ldg/stg) not implemented");
    end
    // synthesis translate_on

endmodule
