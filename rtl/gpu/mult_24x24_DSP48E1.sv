/**
 * 24x24 unsigned multiplier using 2 DSP48E1 slices
 * Computes P = A * B using two 24x17 multiplication and the MAC function of one of the DSPs
 * We are trying to compute:
 * P = (A * B[23:17]) << 17 + (A * B[16:0])
 * However, the cascade input doesn't have "<< 17" functionality and will limit to ~300 MHz when
 * using the fabric path from P->C due to high setup requirement
 * The cascade input does have >> 17, so we will compute:
 * P >> 17 = (A * B[23:17]) + (A * B[16:0]) >> 17
 * and we will extract the lower 17 bits from the lower stage into a fabric FF to combine
 * the results. Apparently that's what Xilinx tells you to do?
 */

module mult_24x24_DSP48E1 (
    input wire clk,
    input wire rst,
    input wire [23:0] a,
    input wire [23:0] b,
    output wire [47:0] p
);
    // DSP data path signals
    wire [29:0] dsp1_a;
    wire [17:0] dsp1_b;
    wire [47:0] dsp1_p;

    wire [29:0] dsp2_a;
    wire [17:0] dsp2_b;
    wire [47:0] dsp2_p;

    // DSP2: Compute A * B[23:17] + DSP1_P >> 17
    assign dsp2_a = {6'b0, a};
    assign dsp2_b = {11'b0, b[23:17]};

    // DSP1: Compute A * B[16:0]
    assign dsp1_a = {6'b0, a};
    assign dsp1_b = {1'b0, b[16:0]};

    // DSP2 config: P = PCIN >> 17 + M
    // Note that 01_01 chooses a single M (see UG479)
    wire [6:0] dsp2_opmode = 7'b101_01_01;
    // DSP1 config: P = M
    wire [6:0] dsp1_opmode = 7'b000_01_01;

    wire [47:0] dsp1_cascout;

    // Capture the lower 17 bits from DSP1 output
    // We need to hold this value until DSP2 computes the upper part
    logic [16:0] dsp1_p2;
    always_ff @(posedge clk) begin
        dsp1_p2 <= dsp1_p[16:0];
    end

    // Concatenate into output
    assign p = {dsp2_p[30:0], dsp1_p2};

    DSP48E1 #(
        .AREG(2), .BREG(2), .CREG(0), .DREG(0), .ADREG(0),
        .MREG(1), .PREG(1),
        .ALUMODEREG(0), .CARRYINREG(0), .CARRYINSELREG(0),
        .INMODEREG(0), .OPMODEREG(0),
        .USE_DPORT("FALSE"),
        .USE_MULT("MULTIPLY"),
        .USE_SIMD("ONE48")
    ) dsp2 (
        .CLK(clk),
        .RSTA(rst), .RSTB(rst), .RSTC(rst), .RSTD(rst),
        .RSTM(rst), .RSTP(rst), .RSTALLCARRYIN(rst),
        .RSTALUMODE(rst), .RSTCTRL(rst), .RSTINMODE(rst),

        .A(dsp2_a),
        .B(dsp2_b),
        .C(48'b0),
        .D(25'b0),

        .ALUMODE(4'b0000),
        .CARRYINSEL(3'b000),
        .CARRYIN(1'b0),
        .INMODE(5'b00000),
        .OPMODE(dsp2_opmode),

        .CEA1(1'b1), .CEA2(1'b1), .CEAD(1'b0), .CEALUMODE(1'b1),
        .CEB1(1'b1), .CEB2(1'b1), .CEC(1'b0), .CECARRYIN(1'b1),
        .CECTRL(1'b1), .CED(1'b0), .CEINMODE(1'b1), .CEM(1'b1), .CEP(1'b1),

        .ACIN(30'b0), .BCIN(18'b0), .PCIN(dsp1_cascout),
        .CARRYCASCIN(1'b0), .MULTSIGNIN(1'b0),

        .P(dsp2_p),
        .ACOUT(), .BCOUT(), .PCOUT(), .CARRYCASCOUT(), .MULTSIGNOUT(),
        .OVERFLOW(), .PATTERNBDETECT(), .PATTERNDETECT(), .UNDERFLOW(), .CARRYOUT()
    );

    DSP48E1 #(
        .AREG(1), .BREG(1), .CREG(0), .DREG(0), .ADREG(0),
        .MREG(1), .PREG(1),
        .ALUMODEREG(0), .CARRYINREG(0), .CARRYINSELREG(0),
        .INMODEREG(0), .OPMODEREG(0),
        .USE_DPORT("FALSE"),
        .USE_MULT("MULTIPLY"),
        .USE_SIMD("ONE48")
    ) dsp1 (
        .CLK(clk),
        .RSTA(rst), .RSTB(rst), .RSTC(rst), .RSTD(rst),
        .RSTM(rst), .RSTP(rst), .RSTALLCARRYIN(rst),
        .RSTALUMODE(rst), .RSTCTRL(rst), .RSTINMODE(rst),

        .A(dsp1_a),
        .B(dsp1_b),
        .C(48'b0),
        .D(25'b0),

        .ALUMODE(4'b0000),
        .CARRYINSEL(3'b000),
        .CARRYIN(1'b0),
        .INMODE(5'b00000),
        .OPMODE(dsp1_opmode),

        .CEA1(1'b0), .CEA2(1'b1), .CEAD(1'b0), .CEALUMODE(1'b1),
        .CEB1(1'b0), .CEB2(1'b1), .CEC(1'b0), .CECARRYIN(1'b1),
        .CECTRL(1'b1), .CED(1'b0), .CEINMODE(1'b1), .CEM(1'b1), .CEP(1'b1),

        .ACIN(30'b0), .BCIN(18'b0), .PCIN(48'b0),
        .CARRYCASCIN(1'b0), .MULTSIGNIN(1'b0),

        .P(dsp1_p),
        .ACOUT(), .BCOUT(), .PCOUT(dsp1_cascout), .CARRYCASCOUT(), .MULTSIGNOUT(),
        .OVERFLOW(), .PATTERNBDETECT(), .PATTERNDETECT(), .UNDERFLOW(), .CARRYOUT()
    );
endmodule
