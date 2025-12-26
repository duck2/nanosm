/**
 * FPU 2x clock wrapper
 * Takes 2-wide inputs/outputs on clk domain (125 MHz)
 * Internally runs a single FPU at clk_2x (250 MHz)
 * clk_2x must be derived from clk (synchronous, no FIFO needed)
 */
module fpu_2x (
    input wire clk,
    input wire clk_2x,
    input wire rst,

    // 2-wide inputs (clk domain)
    input wire [31:0] in1_0,
    input wire [31:0] in1_1,
    input wire [31:0] in2_0,
    input wire [31:0] in2_1,
    input wire [31:0] in3_0,
    input wire [31:0] in3_1,
    input wire valid_in_0,
    input wire valid_in_1,

    // 2-wide outputs (clk domain)
    output wire [31:0] out_0,
    output wire [31:0] out_1,
    output wire valid_out_0,
    output wire valid_out_1
);

    // Phase toggle in clk_2x domain
    // phase=0: process slot 0, phase=1: process slot 1
    logic phase;
    always_ff @(posedge clk_2x) begin
        if (rst)
            phase <= 0;
        else
            phase <= ~phase;
    end

    // Mux inputs based on phase (in clk_2x domain) - no input registers, sample directly from RF
    wire [31:0] fpu_in1 = phase ? in1_1 : in1_0;
    wire [31:0] fpu_in2 = phase ? in2_1 : in2_0;
    wire [31:0] fpu_in3 = phase ? in3_1 : in3_0;
    wire fpu_valid_in = phase ? valid_in_1 : valid_in_0;

    // FPU instance running at clk_2x
    wire [31:0] fpu_out;
    wire fpu_valid_out;

    fpu_DSP48E1 fpu_inst (
        .clk(clk_2x),
        .rst(rst),
        .in1(fpu_in1),
        .in2(fpu_in2),
        .in3(fpu_in3),
        .valid_in(fpu_valid_in),
        .out(fpu_out),
        .valid_out(fpu_valid_out)
    );

    // Track output phase (accounts for FPU pipeline latency)
    // Right now the FPU has an odd number of stages so this should
    // probably be the opposite of input phase
    wire out_phase = ~phase;

    // Capture outputs based on output phase (demux serial back to parallel)
    logic [31:0] out_0_reg, out_1_reg;
    logic valid_out_0_reg, valid_out_1_reg;

    always_ff @(posedge clk_2x) begin
        if (rst) begin
            valid_out_0_reg <= 0;
            valid_out_1_reg <= 0;
        end else begin
            if (!out_phase) begin
                out_0_reg <= fpu_out;
                valid_out_0_reg <= fpu_valid_out;
            end else begin
                out_1_reg <= fpu_out;
                valid_out_1_reg <= fpu_valid_out;
            end
        end
    end

    assign out_0 = out_0_reg;
    assign out_1 = out_1_reg;
    assign valid_out_0 = valid_out_0_reg;
    assign valid_out_1 = valid_out_1_reg;

endmodule
