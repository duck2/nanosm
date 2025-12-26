/**
 * DSP48E1 ALU - Single-cycle ALU operations using Xilinx DSP48E1 primitive
 * 
 * DSP48E1 High-Level Capabilities:
 * ================================
 * The DSP can perform three types of operations selected by ALUMODE:
 *   1. P = Z + X + Y         (Addition)
 *   2. P = Z - (X + Y)       (Subtraction)
 *   3. P = X (bitwise) Z     (Bitwise operations: AND, OR, XOR, etc.)
 * 
 * Input Selection via OPMODE:
 *   - X can be: 0, M (multiplier output A*B), P (feedback), or A:B (concatenated)
 *   - Y can be: 0, M (multiplier output), 48'hFFFF_FFFF_FFFF, or C
 *   - Z can be: 0, P (feedback), or C
 * 
 * ALU Operation Mapping:
 * ======================
 * Requirements: 1. Single-cycle operation, 2. Support bitwise operations
 * 
 * Strategy:
 *   1. MUL:     A=in1, B=in2, C=x,    select X=M, Y=M, Z=0
 *   2. ADD:     A=0,   B=in2, C=in1,  select X=A:B, Y=0, Z=C
 *   3. SUB:     A=0,   B=in2, C=in1,  select X=A:B, Y=0, Z=C, subtract mode
 *   4. Bitwise: A=0,   B=in2, C=in1,  select X=A:B, Y=0, Z=C, bitwise mode
 *
 */

module alu_DSP48E1 (
    input  logic        clk,
    input  logic        rst,
    
    // Control Interface
    input  logic [4:0]  op,         // Operation select
    
    // Data Interface
    input  wire [31:0] in1,          // First operand
    input  wire [31:0] in2,          // Second operand
    output logic [31:0] out           // Result
);

    // Operation encoding
    localparam OP_ADD  = 5'b00000;  // ADD: out = in1 + in2
    localparam OP_SUB  = 5'b00001;  // SUB: out = in1 - in2
    localparam OP_MULS  = 5'b00010;  // MUL: out = (in1[15:0] * in2[15:0]) (signed)
    localparam OP_MULU  = 5'b00011;  // MULU: out = (in1[15:0] * in2[15:0]) (unsigned)
    localparam OP_XOR  = 5'b00100;  // XOR: out = in1 ^ in2
    localparam OP_OR   = 5'b00101;  // OR:  out = in1 | in2
    localparam OP_AND  = 5'b00111;  // AND: out = in1 & in2

    // Output selection encoding (pipelined with DSP)
    localparam OUT_SEL_NORMAL = 1'b0;  // for ADD/SUB/bitwise
    localparam OUT_SEL_MUL    = 1'b1;  // for MUL

    // DSP control signals
    logic [6:0] opmode;
    logic [3:0] alumode;
    logic [2:0] carryinsel;
    logic       carryin;
    logic [4:0] inmode;

    // DSP data path signals
    wire [29:0] dsp_a;    // DSP A input (30 bits)
    wire [17:0] dsp_b;    // DSP B input (18 bits)
    wire [47:0] dsp_c;    // DSP C input (48 bits)
    wire [24:0] dsp_d;    // DSP D input (25 bits)
    wire [47:0] dsp_p;    // DSP P output (48 bits)

    // Operand alignment. We use D input and set INMODE so that M=B*D.
    // Two gotchas here: DSP mul is signed by default, so mux top bits to sign or zero extend.
    // Shift A:B and C 7 bits to the right to align output.
    // A:B shifted 7 bits to the right puts 10 bits of the ADD operand into B.
    // (9 bits from in2, 1 carry-break). Mux those bits as we need B for multiplication.
    wire sign_mul_in1 = (op == OP_MULS) ? in1[15] : 0;
    wire sign_mul_in2 = (op == OP_MULS) ? in2[15] : 0;
    // Place a 1 carefully when subtracting bc lower bits have trash and won't balloon up a carry properly
    wire c_sub_adjust = (op == OP_SUB);
    wire [9:0] dsp_b_top = (op == OP_MULS || op == OP_MULU) ? {sign_mul_in1, in1[15:7]} : {in2[8:0], 1'b0};
    assign dsp_a = {7'b0, in2[31:9]};
    assign dsp_b = {dsp_b_top, in1[6:0], 1'b0};
    assign dsp_c = {7'b0, in1[31:0], c_sub_adjust, 8'b0};
    assign dsp_d = {sign_mul_in2, in2[15:0], 8'b0};

    // Operation mode selection (combinational, DSP will register internally)
    always_comb begin
        case (op)
            OP_ADD: begin
                // ADD: P = in1 + in2
                opmode = 7'b011_00_11;  // Z=C Y=0 X=A:B
                alumode = 4'b0000;  // P = Z + X + Y
            end
            OP_SUB: begin
                // SUB: P = in1 - in2
                opmode = 7'b011_00_11;  // Z=C Y=0 X=A:B
                alumode = 4'b0011;  // P = Z - X - Y
            end
            OP_MULS, OP_MULU: begin
                // MUL: P = in1 * in2 (lower 16 bits)
                opmode = 7'b000_01_01;  // Z=0 Y=M X=M 
                alumode = 4'b0000;  // P = Z + X + Y
            end
            OP_XOR: begin
                // XOR: P = in1 ^ in2
                opmode = 7'b011_00_11;  // Z=C, P = Z ^ X, X=A:B
                alumode = 4'b0100;  // P = Z ^ X
            end
            OP_OR: begin
                // OR: P = in1 | in2
                opmode = 7'b011_10_11;  // Z=C, P = Z | X, X=A:B
                alumode = 4'b1100;  // P = Z | X
            end
            OP_AND: begin
                // AND: P = in1 & in2
                opmode = 7'b011_00_11;  // Z=C, P = Z & X, X=A:B
                alumode = 4'b1100;  // P = Z & X
            end
            default: begin
                opmode = 7'b011_00_11;  // Z=C Y=0 X=A:B
                alumode = 4'b0000;  // P = Z + X + Y
            end
        endcase
    end

    // Get output from 7 bits below the msb (upper 2 bits are reserved for unsigned mul, 5 bits down bc of 25x18 -> 48)
    assign out = dsp_p[40:9];

    // Fixed control signals
    assign carryinsel = 3'b000;
    assign carryin = 1'b0;
    assign inmode = 5'b0_1111;  // Use D for multiplier A port

    // DSP48E1 instance with AREG=1, BREG=1, CREG=1, DREG=1, OPMODEREG=1, ALUMODEREG=1, PREG=1, MREG=0
    DSP48E1 #(
        .ACASCREG(1),           // match AREG
        .ADREG(0),              // No pre-adder pipeline
        .ALUMODEREG(1),         // Register ALUMODE
        .AREG(1),               // A input register
        .BCASCREG(1),           // match BREG
        .BREG(1),               // B input register
        .CARRYINREG(0),         // Don't register CARRYIN
        .CARRYINSELREG(0),      // Don't register CARRYINSEL
        .CREG(1),               // C input register
        .DREG(1),               // D input register
        .INMODEREG(0),          // Don't register INMODE
        .MREG(0),               // No multiplier pipeline register
        .OPMODEREG(1),          // Register OPMODE
        .PREG(1),               // Output register
        .USE_DPORT("TRUE"),    // Don't use pre-adder
        .USE_MULT("DYNAMIC"),   // Maybe use multiplier
        .USE_SIMD("ONE48")      // Single 48-bit operation
    ) dsp_inst (
        // Clock and Reset
        .CLK(clk),              // Clock
        .RSTA(rst),             // Reset for A pipeline
        .RSTALLCARRYIN(rst),    // Reset for carry pipeline
        .RSTALUMODE(rst),       // Reset for ALUMODE pipeline
        .RSTB(rst),             // Reset for B pipeline
        .RSTC(rst),             // Reset for C pipeline
        .RSTCTRL(rst),          // Reset for OPMODE pipeline
        .RSTD(rst),             // Reset for D pipeline
        .RSTINMODE(rst),        // Reset for INMODE pipeline
        .RSTM(rst),             // Reset for multiplier pipeline
        .RSTP(rst),             // Reset for P pipeline

        // Data input ports
        .A(dsp_a),              // 30-bit A input
        .B(dsp_b),              // 18-bit B input
        .C(dsp_c),              // 48-bit C input
        .D(dsp_d),              // 25-bit D input

        // Control input ports
        .ALUMODE(alumode),      // 4-bit ALU control
        .CARRYINSEL(carryinsel),// 3-bit carry select
        .CARRYIN(carryin),      // 1-bit carry input
        .INMODE(inmode),        // 5-bit input mode
        .OPMODE(opmode),        // 7-bit operation mode

        // Clock enable ports (all active)
        .CEA1(1'b0),  // AREG=1
        .CEA2(1'b1),
        .CEAD(1'b1),
        .CEALUMODE(1'b1),
        .CEB1(1'b0),  // BREG=1
        .CEB2(1'b1),
        .CEC(1'b1),
        .CECARRYIN(1'b1),
        .CECTRL(1'b1),
        .CED(1'b1),
        .CEINMODE(1'b1),
        .CEM(1'b0),  // MREG=0
        .CEP(1'b1),

        // Cascade ports (unused)
        .ACIN(30'b0),
        .BCIN(18'b0),
        .PCIN(48'b0),
        .CARRYCASCIN(1'b0),
        .MULTSIGNIN(1'b0),

        // Output ports
        .P(dsp_p),             // 48-bit output
        .ACOUT(),              // Unused cascade outputs
        .BCOUT(),
        .PCOUT(),
        .CARRYCASCOUT(),
        .MULTSIGNOUT(),

        // Status ports (unused)
        .OVERFLOW(),
        .PATTERNBDETECT(),
        .PATTERNDETECT(),
        .UNDERFLOW(),
        .CARRYOUT()
    );

endmodule
