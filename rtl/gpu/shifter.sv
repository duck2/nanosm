/**
 * 32-bit Barrel Shifter
 * Simple 5-stage combinational barrel shifter with input and output registers.
 * Latency: 2 cycles
 */
`timescale  1 ps / 1 ps

module shifter (
    input  logic        clk,
    input  logic        rst,
    input  logic [1:0]  op,         // Operation
    input  logic [4:0]  shift_amt,  // Shift amount (0-31)
    input  logic [31:0] in1,        // Input data
    output logic [31:0] out         // Result
);
    localparam OP_SHL = 2'b00;  // Shift left logical
    localparam OP_SHR = 2'b01;  // Shift right logical
    localparam OP_SRA = 2'b11;  // Shift right arithmetic

    // Input registers
    logic [1:0]  op_reg;
    logic [4:0]  shift_amt_reg;
    logic [31:0] in1_reg;

    always_ff @(posedge clk) begin
        if (rst) begin
            op_reg <= 2'b0;
            shift_amt_reg <= 5'b0;
            in1_reg <= 32'b0;
        end else begin
            op_reg <= op;
            shift_amt_reg <= shift_amt;
            in1_reg <= in1;
        end
    end

    // Barrel shifter stages (all combinational)
    logic [31:0] stage0, stage1, stage2, stage3, stage4;

    wire sign_bit = (op_reg == OP_SRA) ? in1_reg[31] : 0;

    // Stage 0: shift by 0 or 1
    assign stage0 = (op_reg == OP_SHR || op_reg == OP_SRA)
        ? (shift_amt_reg[0] ? {sign_bit, in1_reg[31:1]} : in1_reg)
        : (shift_amt_reg[0] ? {in1_reg[30:0], 1'b0} : in1_reg);

    // Stage 1: shift by 0 or 2
    assign stage1 = (op_reg == OP_SHR || op_reg == OP_SRA)
        ? (shift_amt_reg[1] ? {{2{sign_bit}}, stage0[31:2]} : stage0)
        : (shift_amt_reg[1] ? {stage0[29:0], 2'b0} : stage0);

    // Stage 2: shift by 0 or 4
    assign stage2 = (op_reg == OP_SHR || op_reg == OP_SRA)
        ? (shift_amt_reg[2] ? {{4{sign_bit}}, stage1[31:4]} : stage1)
        : (shift_amt_reg[2] ? {stage1[27:0], 4'b0} : stage1);

    // Stage 3: shift by 0 or 8
    assign stage3 = (op_reg == OP_SHR || op_reg == OP_SRA)
        ? (shift_amt_reg[3] ? {{8{sign_bit}}, stage2[31:8]} : stage2)
        : (shift_amt_reg[3] ? {stage2[23:0], 8'b0} : stage2);

    // Stage 4: shift by 0 or 16
    assign stage4 = (op_reg == OP_SHR || op_reg == OP_SRA)
        ? (shift_amt_reg[4] ? {{16{sign_bit}}, stage3[31:16]} : stage3)
        : (shift_amt_reg[4] ? {stage3[15:0], 16'b0} : stage3);

    // Output register
    always_ff @(posedge clk) begin
        if (rst)
            out <= 32'b0;
        else
            out <= stage4;
    end

endmodule
