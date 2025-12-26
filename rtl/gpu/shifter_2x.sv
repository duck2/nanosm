`timescale 1ns / 1ps

/**
 * Shifter 2x clock wrapper
 * Takes 2-wide inputs/outputs on clk domain
 * Internally runs a single shifter at clk_2x
 */
module shifter_2x (
    input wire clk,
    input wire clk_2x,
    input wire rst,

    // 2-wide inputs (accent on clk_2x edges)
    input wire [1:0] op_0,
    input wire [1:0] op_1,
    input wire [4:0] shift_amt_0,
    input wire [4:0] shift_amt_1,
    input wire [31:0] in1_0,
    input wire [31:0] in1_1,

    // 2-wide outputs (clk domain)
    output wire [31:0] out_0,
    output wire [31:0] out_1
);

    // Phase toggle in clk_2x domain
    logic phase;
    always_ff @(posedge clk_2x) begin
        if (rst)
            phase <= 0;
        else
            phase <= ~phase;
    end

    // Mux inputs based on phase - no input registers, sample directly
    wire [1:0] shifter_op = phase ? op_1 : op_0;
    wire [4:0] shifter_shift_amt = phase ? shift_amt_1 : shift_amt_0;
    wire [31:0] shifter_in1 = phase ? in1_1 : in1_0;

    // Shifter instance running at clk_2x
    wire [31:0] shifter_out;

    shifter shifter_inst (
        .clk(clk_2x),
        .rst(rst),
        .op(shifter_op),
        .shift_amt(shifter_shift_amt),
        .in1(shifter_in1),
        .out(shifter_out)
    );

    // Track output phase (accounts for shifter 2-cycle latency, even number)
    wire out_phase = phase;

    // Capture outputs based on output phase (demux serial back to parallel)
    logic [31:0] out_0_reg, out_1_reg;

    always_ff @(posedge clk_2x) begin
        if (rst) begin
            out_0_reg <= 0;
            out_1_reg <= 0;
        end else begin
            if (!out_phase)
                out_0_reg <= shifter_out;
            else
                out_1_reg <= shifter_out;
        end
    end

    assign out_0 = out_0_reg;
    assign out_1 = out_1_reg;

endmodule
