/**
 * Float to Fixed-Point Converter (ftofx)
 * Converts F32 to signed 16-bit fixed-point. Mag mode only for now.
 * Latency: 4 cycles
 *
 * Output: S = trunc(x * 2^F) where F = frac_bits, truncation toward zero.
 * Naturally clamped to [-32767, 32767].
 */
`timescale 1ps / 1ps

module ftofx (
    input wire clk,
    input wire rst,
    input wire [31:0] in,
    input wire [3:0] frac_bits,
    output wire [15:0] out
);
    // F32 field extraction (combinational)
    wire in_sign = in[31];
    wire [7:0] in_exp = in[30:23];
    wire [22:0] in_mant = in[22:0];

    // =========================================================================
    // Stage 1: Register raw inputs
    // =========================================================================
    logic s1_sign;
    logic [7:0] s1_exp;
    logic [22:0] s1_mant;
    logic [3:0] s1_frac_bits;

    always_ff @(posedge clk) begin
        if (rst) begin
            s1_sign <= 0;
            s1_exp <= 0;
            s1_mant <= 0;
            s1_frac_bits <= 0;
        end else begin
            s1_sign <= in_sign;
            s1_exp <= in_exp;
            s1_mant <= in_mant;
            s1_frac_bits <= frac_bits;
        end
    end

    // =========================================================================
    // Stage 2: Compute shift value (normalized to start bit for extraction)
    // =========================================================================
    // We want: out_mag = (mant24 * 2^p)[14:0] where p = exp - 150 + frac_bits
    // Equivalently, build intermediate = {14'b0, mant24, 14'b0} (52 bits),
    // then extract 15 bits starting at position "start".
    //
    // The "1" of mant24 is at bit 37 of intermediate.
    // We want it at bit pos = frac_bits + exp - 127 of the 15-bit output.
    // So: start = 37 - pos = 164 - exp - frac_bits
    //
    // Valid range: start in [0, 37] -> 38 values
    // start > 37: underflow, output is 0
    // start < 0: overflow
    // TODO for human: handle overflow case - we still want frac bits even when
    //   integer part overflows. Useful for phase/wrap mode. ~39 total shift values.

    wire s1_is_zero = s1_exp == 0; // DAZ
    wire [23:0] s1_mant24 = {1'b1, s1_mant};

    wire signed [9:0] s1_start_signed = 10'sd164 - {2'b0, s1_exp} - {6'b0, s1_frac_bits};
    wire [5:0] s1_start = s1_start_signed[5:0];

    logic s2_sign;
    logic [23:0] s2_mant24;
    logic [5:0] s2_start;
    logic s2_underflow;
    logic s2_overflow;

    always_ff @(posedge clk) begin
        if (rst) begin
            s2_sign <= 0;
            s2_mant24 <= 0;
            s2_start <= 0;
            s2_underflow <= 0;
            s2_overflow <= 0;
        end else begin
            s2_sign <= s1_sign;
            s2_mant24 <= s1_mant24;
            s2_start <= s1_start;
            s2_underflow <= s1_underflow;
            s2_overflow <= s1_overflow;
        end
    end

    // =========================================================================
    // Stage 3: Generate 0-padded intermediate (14+24+14=52), select 15 bits
    // =========================================================================
    // intermediate[51:38] = 14'b0, [37:14] = mant24, [13:0] = 14'b0
    // Extract intermediate[start+14 : start]
    logic [14:0] s3_selected;
    logic [5:0] s3_start;

    wire [51:0] s2_wide = {14'b0, s2_mant24, 14'b0};
    wire [14:0] s2_selected = s2_wide[s2_start+14:s2_start];

    logic s3_sign;

    always_ff @(posedge clk) begin
        if (rst) begin
            s3_sign <= 0;
            s3_selected <= 0;
        end else begin
            s3_sign <= s2_sign;
            s3_selected <= s2_selected;
        end
    end

    // =========================================================================
    // Stage 4: Generate sign and do 2's complement
    // =========================================================================
    wire [15:0] s3_unsigned = {1'b0, s3_mag};
    wire [15:0] s3_signed_out = s3_sign ? -s3_unsigned : s3_unsigned;

    logic [15:0] s4_out;

    always_ff @(posedge clk) begin
        if (rst) begin
            s4_out <= 0;
        end else begin
            s4_out <= s3_signed_out;
        end
    end

    assign out = s4_out;

endmodule
