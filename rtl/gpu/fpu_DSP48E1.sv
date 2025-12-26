/**
 * Float FMAC using DSP48E1
 * Always computes: out = in1 * in2 + in3
 * Note that it operates with 27 bit mantissa precision and FTZ/DAZ
 * This means ADD, SUB and MUL are bit-exact
 * while FMA is "fused but not extended": it acts more like
 * rne(trunc27_with_sticky(a*b) + c) instead of rne(a*b + c).
 * It's still more accurate than separate MUL+ADD, but would have
 * quite a bit-ulp error if compared to rne(a*b + c).
 *
 * Controller can degenerate to other ops:
 *   ADD: in1=1.0, in2=a, in3=b  → out = a + b
 *   SUB: in1=1.0, in2=a, in3=-b → out = a - b
 *   MUL: in1=a, in2=b, in3=0.0  → out = a * b
 *
 * This first targeted 166 MHz (on Artix 7) and 10 stages, but after
 * seeing that I have to double-pump this thing to have any decent FLOPS,
 * it now targets 300 MHz and 15 stages. Note that the Xilinx FMA IP can
 * do ~400 MHz with 17 stages, but well, that's the vendor IP. The vendor
 * IP is also FTZ/DAZ and the FMA is not "high-precision" except for Versal
 * devices according to ug1399-vitis-hls. */

module fpu_DSP48E1 (
    input  wire clk,
    input  wire rst,

    input  wire [31:0] in1,
    input  wire [31:0] in2,
    input  wire [31:0] in3,
    input  wire valid_in,

    output wire [31:0] out,
    output wire valid_out
);
    /**
     * 1. Small comb before first FFs. Multiplier inputs use internal DSP
     * registers so we want to feed the inputs into there as soon as possible
     * and start moving the other stuff through the side pipeline.
     * After this stage we won't have access to mant1 and mant2,
     * so we have some minimal logic to run the is_mant1_zero and is_mant2_zero check
     */

    logic s1_valid;
    logic [8:0] s1_signexp1;
    logic [8:0] s1_signexp2;
    logic [31:0] s1_in3;
    logic s1_mant1_is_zero;
    logic s1_mant2_is_zero;

    always_ff @(posedge clk) begin
        if (rst) begin
            s1_valid <= 0;
        end else begin
            s1_valid <= valid_in;
        end
    end

    always_ff @(posedge clk) begin
        s1_signexp1 <= in1[31:23];
        s1_signexp2 <= in2[31:23];
        s1_in3 <= in3;
        s1_mant1_is_zero <= (in1[22:0] == 23'd0);
        s1_mant2_is_zero <= (in2[22:0] == 23'd0);
    end

    /* Feed in1 and in2 mantissa directly into multiplier.
     * Mult takes 4 cycles */
    wire [47:0] s3_mult_p;
    wire [23:0] mant1_in = {1'b1, in1[22:0]};
    wire [23:0] mant2_in = {1'b1, in2[22:0]};

    mult_24x24_DSP48E1 multiplier (
        .clk(clk),
        .rst(rst),
        .a(mant1_in),
        .b(mant2_in),
        .p(s3_mult_p)
    );

    /**
     * 1b. Wait for mult results, start NaN/Inf handling (compare to 0x00/0xFF)
     * Start calculating preliminary result exponent
     */
    logic s1b_valid;
    logic s1b_sign1;
    logic s1b_sign2;
    logic [31:0] s1b_in3;
    logic signed [9:0] s1b_exp_mul10_pre;
    logic s1b_exp1_is_zero;
    logic s1b_exp2_is_zero;
    logic s1b_exp3_is_zero;
    logic s1b_exp1_is_ff;
    logic s1b_exp2_is_ff;
    logic s1b_exp3_is_ff;
    logic s1b_mant1_is_zero;
    logic s1b_mant2_is_zero;
    logic s1b_mant3_is_zero;

    wire [7:0] s1_exp1 = s1_signexp1[7:0];
    wire [7:0] s1_exp2 = s1_signexp2[7:0];
    wire [7:0] s1_exp3 = s1_in3[30:23];

    /* Calculate preliminary mult exponent (still needs to be debiased + carry added) */
    wire signed [9:0] s1_exp_mul10_pre = $signed({2'b0, s1_exp1}) + $signed({2'b0, s1_exp2});

    always_ff @(posedge clk) begin
        if (rst) begin
            s1b_valid <= 0;
        end else begin
            s1b_valid <= s1_valid;
        end
    end

    always_ff @(posedge clk) begin
        s1b_sign1 <= s1_signexp1[8];
        s1b_sign2 <= s1_signexp2[8];
        s1b_in3 <= s1_in3;
        s1b_exp_mul10_pre <= s1_exp_mul10_pre;
        s1b_exp1_is_zero <= s1_exp1 == 8'b0;
        s1b_exp2_is_zero <= s1_exp2 == 8'b0;
        s1b_exp3_is_zero <= s1_exp3 == 8'b0;
        s1b_exp1_is_ff <= s1_exp1 == 8'hFF;
        s1b_exp2_is_ff <= s1_exp2 == 8'hFF;
        s1b_exp3_is_ff <= s1_exp3 == 8'hFF;
        s1b_mant1_is_zero <= s1_mant1_is_zero;
        s1b_mant2_is_zero <= s1_mant2_is_zero;
        s1b_mant3_is_zero <= s1_in3[22:0] == 23'd0;
    end

    /**
     * 2. Process inputs while waiting for multiplication result.
     * Do DAZ/NaN/Inf handling and compute preliminary mult exp.
     *
     * We should try to do as much computation as possible in parallel
     * while waiting for the multiplication result. One candidate is
     * getting the final exponent/exp diff. Even though we don't know if
     * we will increase exp due to a mantissa multiplication carry,
     * we can still compute both carry/no carry cases here and then
     * select after multiplication. That should result in a much more
     * balanced pipeline
     */

    logic s2_valid;
    logic s2_sign_mul;
    logic signed [9:0] s2_exp_mul10_nocarry;
    logic signed [9:0] s2_exp_mul10_carry;
    logic [31:0] s2_in3;
    logic s2_mul_result_is_zero;
    logic s2_is_zero3;
    /* Do not extract SRLs for nan/inf flags. That forces placement of
     * pipeline stages to to "curl up" around the SRL */
    (* shreg_extract = "no" *)
    logic s2_result_is_nan;
    (* shreg_extract = "no" *)
    logic s2_result_is_inf;

    /* DAZ (something is 0 if its exp == 0) */
    wire s1b_is_zero1 = s1b_exp1_is_zero;
    wire s1b_is_zero2 = s1b_exp2_is_zero;
    wire s1b_is_zero3 = s1b_exp3_is_zero;

    /* We did the 00 and FF comparisons one stage up, otherwise this logic gets too deep */
    wire s1b_is_nan1 = s1b_exp1_is_ff && !s1b_mant1_is_zero;
    wire s1b_is_nan2 = s1b_exp2_is_ff && !s1b_mant2_is_zero;
    wire s1b_is_nan3 = s1b_exp3_is_ff && !s1b_mant3_is_zero;

    wire s1b_is_inf1 = s1b_exp1_is_ff && s1b_mant1_is_zero;
    wire s1b_is_inf2 = s1b_exp2_is_ff && s1b_mant2_is_zero;
    wire s1b_is_inf3 = s1b_exp3_is_ff && s1b_mant3_is_zero;

    wire s1b_sign_mul = s1b_sign1 ^ s1b_sign2;

    /* inf * 0 = nan */
    wire s1b_is_infx0 = (s1b_is_inf1 && s1b_is_zero2) || (s1b_is_inf2 && s1b_is_zero1);

    /* inf - inf = nan */
    wire s1b_is_infminf = (s1b_is_inf1 || s1b_is_inf2)
        && s1b_is_inf3
        && (s1b_sign_mul != s1b_in3[31]);

    wire s1b_result_is_nan = s1b_is_nan1 | s1b_is_nan2 | s1b_is_nan3 | s1b_is_infx0 | s1b_is_infminf;
    wire s1b_result_is_inf = s1b_is_inf1 | s1b_is_inf2 | s1b_is_inf3;

    always_ff @(posedge clk) begin
        if (rst) begin
            s2_valid <= 0;
        end else begin
            s2_valid <= s1b_valid;
        end
    end

    /* Calculate mult exponents for no-carry and carry case */
    wire signed [9:0] s1b_exp_mul10_nocarry = s1b_exp_mul10_pre - 10'sd127;
    wire signed [9:0] s1b_exp_mul10_carry = s1b_exp_mul10_pre - 10'sd127 + 10'sd1;

    always_ff @(posedge clk) begin
        s2_sign_mul <= s1b_sign_mul;
        s2_exp_mul10_nocarry <= s1b_exp_mul10_nocarry;
        s2_exp_mul10_carry <= s1b_exp_mul10_carry;
        s2_in3 <= s1b_in3;
        s2_result_is_nan <= s1b_result_is_nan;
        s2_result_is_inf <= s1b_result_is_inf;
        s2_mul_result_is_zero <= s1b_is_zero1 | s1b_is_zero2;
        s2_is_zero3 = s1b_is_zero3;
    end

    /**
     * 3. Propagate while waiting for the multiplication result.
     * Keep doing what we can do without knowing the mul result, such as
     * processing DAZ for in3 and running exp comparison in case of
     * carry vs. no carry
     */
    logic s3_valid;
    logic s3_sign_mul;
    logic s3_sign3;
    logic [9:0] s3_exp_mul10_nocarry;
    logic [9:0] s3_exp_mul10_carry;
    logic s3_exp_mul_ge_exp3_nocarry;
    logic s3_exp_mul_ge_exp3_carry;
    logic [7:0] s3_exp3;
    logic [23:0] s3_mant3_24;
    logic s3_result_is_nan;
    logic s3_result_is_inf;
    logic s3_mul_result_is_zero;

    wire s2_sign3 = s2_in3[31];
    wire [7:0] s2_exp3 = s2_in3[30:23];
    wire [23:0] s2_mant3_24 = s2_is_zero3 ? 24'b0 : {1'b1, s2_in3[22:0]};

    wire s2_exp_mul_ge_exp3_nocarry = s2_exp_mul10_nocarry >= $signed({2'b0, s2_exp3});
    wire s2_exp_mul_ge_exp3_carry = s2_exp_mul10_carry >= $signed({2'b0, s2_exp3});

    always_ff @(posedge clk) begin
        if (rst) begin
            s3_valid <= 0;
        end else begin
            s3_valid <= s2_valid;
        end
    end

    always_ff @(posedge clk) begin
        s3_sign_mul <= s2_sign_mul;
        s3_sign3 <= s2_sign3;
        s3_exp_mul10_nocarry <= s2_exp_mul10_nocarry;
        s3_exp_mul10_carry <= s2_exp_mul10_carry;
        s3_exp_mul_ge_exp3_nocarry <= s2_exp_mul_ge_exp3_nocarry;
        s3_exp_mul_ge_exp3_carry <= s2_exp_mul_ge_exp3_carry;
        s3_exp3 <= s2_exp3;
        s3_mant3_24 <= s2_mant3_24;
        s3_result_is_nan <= s2_result_is_nan;
        s3_result_is_inf <= s2_result_is_inf;
        s3_mul_result_is_zero <= s2_mul_result_is_zero;
    end

    /**
     * 4a. Mult result is ready.
     * Use mult carry to select between previously computed intermediate
     * exp values. Keep this and a few following stages relatively light as we
     * may need to route back through the DSP column
     */
    logic s4a_valid;
    logic s4a_sign_mul;
    logic s4a_sign3;
    logic signed [9:0] s4a_exp_hi;
    logic signed [9:0] s4a_exp_lo;
    logic s4a_exp_mul_ge_exp3;
    logic [26:0] s4a_mul27;
    logic [23:0] s4a_mant3_24;
    logic s4a_result_is_nan;
    logic s4a_result_is_inf;

    wire s3_mul_carry = s3_mult_p[47];
    wire [26:0] s3_mul27_pre = s3_mul_carry ?
        {s3_mult_p[47:22], |s3_mult_p[21:0]}
        : {s3_mult_p[46:21], |s3_mult_p[20:0]};

    /* Select precomputed mul exp & comparison value based on carry */
    wire signed [9:0] s3_exp_mul = s3_mul_carry ? s3_exp_mul10_carry : s3_exp_mul10_nocarry;
    /* if mul result is killed bc DAZ always treat it exp as smaller than addend exp */
    wire s3_exp_mul_ge_exp3 = s3_mul_carry
        ? s3_exp_mul_ge_exp3_carry && !s3_mul_result_is_zero
        : s3_exp_mul_ge_exp3_nocarry && !s3_mul_result_is_zero;

    wire [26:0] s3_mul27 = s3_mul_result_is_zero ? 27'b0 : s3_mul27_pre;

    always_ff @(posedge clk) begin
        if (rst) begin
            s4a_valid <= 0;
        end else begin
            s4a_valid <= s3_valid;
        end
    end

    always_ff @(posedge clk) begin
        s4a_sign_mul <= s3_sign_mul;
        s4a_sign3 <= s3_sign3;
        s4a_exp_hi <= s3_exp_mul_ge_exp3 ? s3_exp_mul : {2'b0, s3_exp3};
        s4a_exp_lo <= s3_exp_mul_ge_exp3 ? {2'b0, s3_exp3} : s3_exp_mul;
        s4a_exp_mul_ge_exp3 <= s3_exp_mul_ge_exp3;
        s4a_mul27 <= s3_mul27;
        s4a_mant3_24 <= s3_mant3_24;
        s4a_result_is_nan <= s3_result_is_nan;
        s4a_result_is_inf <= s3_result_is_inf;
    end

    /**
     * 4b. Calculate exponent difference, choose hi/lo
     */
    logic s4b_valid;
    logic s4b_sign_hi;
    logic s4b_sign_lo;
    logic signed [9:0] s4b_exp_aligned10;
    logic [9:0] s4b_exp_diff;
    logic s4b_exp_mul_ge_exp3;
    logic [26:0] s4b_mant_hi27;
    logic [26:0] s4b_mant_lo27;
    logic s4b_result_is_nan;
    logic s4b_result_is_inf;

    wire signed [9:0] s4a_exp_diff = s4a_exp_hi - s4a_exp_lo;

    wire [26:0] s4a_mant3_27 = {s4a_mant3_24, 3'b0};
    wire [26:0] s4a_mant_hi27 = s4a_exp_mul_ge_exp3 ? s4a_mul27 : s4a_mant3_27;
    wire [26:0] s4a_mant_lo27 = s4a_exp_mul_ge_exp3 ? s4a_mant3_27 : s4a_mul27;
    wire s4a_sign_hi = s4a_exp_mul_ge_exp3 ? s4a_sign_mul : s4a_sign3;
    wire s4a_sign_lo = s4a_exp_mul_ge_exp3 ? s4a_sign3 : s4a_sign_mul;

    always_ff @(posedge clk) begin
        if (rst) begin
            s4b_valid <= 0;
        end else begin
            s4b_valid <= s4a_valid;
        end
    end

    always_ff @(posedge clk) begin
        s4b_sign_hi <= s4a_sign_hi;
        s4b_sign_lo <= s4a_sign_lo;
        s4b_exp_aligned10 <= s4a_exp_hi;
        s4b_exp_mul_ge_exp3 <= s4a_exp_mul_ge_exp3;
        s4b_exp_diff <= s4a_exp_diff;
        s4b_mant_hi27 <= s4a_mant_hi27;
        s4b_mant_lo27 <= s4a_mant_lo27;
        s4b_result_is_nan <= s4a_result_is_nan;
        s4b_result_is_inf <= s4a_result_is_inf;
    end

    /**
     * 4c. Calculate shift amount, start mantissa comparison
     */
    logic s4c_valid;
    logic s4c_sign_hi;
    logic s4c_sign_lo;
    logic signed [9:0] s4c_exp_aligned10;
    logic [4:0] s4c_shift_amt;
    logic s4c_exp_mul_ge_exp3;
    logic [26:0] s4c_mant_hi27;
    logic [26:0] s4c_mant_lo27;
    logic s4c_result_is_nan;
    logic s4c_result_is_inf;

    wire [4:0] s4b_shift_amt = (s4b_exp_diff > 10'sd31) ? 5'd31 : s4b_exp_diff[4:0];

    /* Start mantissa comparison. Sometimes exponents are equal and hi/lo is determined by
     * comparing mantissa values. Note that we compare pre-shift mantissas for obvious reasons */
    wire s4c_mant_lo_gt_hi;
    is_gt27 mant_comp(.clk(clk), .a(s4b_mant_lo27), .b(s4b_mant_hi27), .is_gt(s4c_mant_lo_gt_hi));

    always_ff @(posedge clk) begin
        if (rst) begin
            s4c_valid <= 0;
        end else begin
            s4c_valid <= s4b_valid;
        end
    end

    always_ff @(posedge clk) begin
        s4c_sign_hi <= s4b_sign_hi;
        s4c_sign_lo <= s4b_sign_lo;
        s4c_exp_aligned10 <= s4b_exp_aligned10;
        s4c_exp_mul_ge_exp3 <= s4b_exp_mul_ge_exp3;
        s4c_shift_amt <= s4b_shift_amt;
        s4c_mant_hi27 <= s4b_mant_hi27;
        s4c_mant_lo27 <= s4b_mant_lo27;
        s4c_result_is_nan <= s4b_result_is_nan;
        s4c_result_is_inf <= s4b_result_is_inf;
    end

    /**
     * 5a. Start alignment shift, wait for mantissa comparison
     */
    logic s5a_valid;
    logic s5a_sign_hi;
    logic s5a_sign_lo;
    logic signed [9:0] s5a_exp_aligned10;
    logic s5a_exps_equal;
    logic s5a_mant_lo_gt_hi;
    logic [26:0] s5a_mant_hi27;
    logic s5a_result_is_nan;
    logic s5a_result_is_inf;

    /* Start shifting */
    wire [26:0] s5a_mant_lo27;
    rshift27 s4c_rshift(.clk(clk), .in(s4c_mant_lo27), .shamt(s4c_shift_amt), .out(s5a_mant_lo27));

    always_ff @(posedge clk) begin
        if (rst) begin
            s5a_valid <= 0;
        end else begin
            s5a_valid <= s4c_valid;
        end
    end

    always_ff @(posedge clk) begin
        s5a_sign_hi <= s4c_sign_hi;
        s5a_sign_lo <= s4c_sign_lo;
        s5a_exp_aligned10 <= s4c_exp_aligned10;
        s5a_exps_equal <= (s4c_shift_amt == 5'b0);
        s5a_mant_lo_gt_hi <= s4c_mant_lo_gt_hi;
        s5a_mant_hi27 <= s4c_mant_hi27;
        s5a_result_is_nan <= s4c_result_is_nan;
        s5a_result_is_inf <= s4c_result_is_inf;
    end

    /**
     * 5b. Wait for alignment shift. Exchange operands if required
     */
    logic s5b_valid;
    logic s5b_sign;
    logic s5b_is_sub;
    logic signed [9:0] s5b_exp_aligned10;
    logic [26:0] s5b_mant_hi27;
    logic [26:0] s5b_mant_lo27;
    logic s5b_result_is_nan;
    logic s5b_result_is_inf;

    // Invert addition result if exps are equal but mant lo > hi
    wire s5a_should_invert = s5a_exps_equal && s5a_mant_lo_gt_hi;
    wire s5a_is_sub = s5a_sign_hi != s5a_sign_lo;
    wire s5a_sign = s5a_should_invert ? s5a_sign_lo : s5a_sign_hi;

    always_ff @(posedge clk) begin
        if (rst) begin
            s5b_valid <= 0;
        end else begin
            s5b_valid <= s5a_valid;
        end
    end

    always_ff @(posedge clk) begin
        s5b_sign <= s5a_sign;
        s5b_is_sub <= s5a_is_sub;
        s5b_mant_hi27 <= s5a_should_invert ? s5a_mant_lo27 : s5a_mant_hi27;
        s5b_mant_lo27 <= s5a_should_invert ? s5a_mant_hi27 : s5a_mant_lo27;
        s5b_exp_aligned10 <= s5a_exp_aligned10;
        s5b_result_is_nan <= s5a_result_is_nan;
        s5b_result_is_inf <= s5a_result_is_inf;
    end

    /**
     * 6. Do add or subtract
     * Do this in the fabric. We could use the DSP adder but
     * the C input we would need to use has ~2.2 ns setup time requirement
     * which takes more or less forever
     */
    logic s6_valid;
    logic s6_sign;
    logic signed [9:0] s6_exp_aligned10;
    logic [27:0] s6_mant_pre28;
    logic s6_result_is_nan;
    logic s6_result_is_inf;

    /* 27-bit add/subtract -> 28 */
    wire [27:0] s5b_mant_pre28 = {1'b0, s5b_mant_hi27}
        + ({1'b0, s5b_mant_lo27} ^ {28{s5b_is_sub}})
        + {27'b0, s5b_is_sub};

    always_ff @(posedge clk) begin
        if (rst) begin
            s6_valid <= 0;
        end else begin
            s6_valid <= s5b_valid;
        end
    end

    always_ff @(posedge clk) begin
        s6_sign <= s5b_sign;
        s6_exp_aligned10 <= s5b_exp_aligned10;
        s6_mant_pre28 <= s5b_mant_pre28;
        s6_result_is_nan <= s5b_result_is_nan;
        s6_result_is_inf <= s5b_result_is_inf;
    end

    /**
     * 7. Do LZC.
     * Both LZC and norm shift have a lot of logic levels, so it needs to
     * be broken up
     */
    logic s7_valid;
    logic s7_sign;
    logic [27:0] s7_mant_pre28;
    logic [9:0] s7_exp_aligned10;
    logic [4:0] s7_lzc;
    logic s7_result_is_nan;
    logic s7_result_is_inf;

    wire [4:0] s6b_lzc;
    lzc27 lzc(.in(s6_mant_pre28[26:0]), .out(s6b_lzc));

    always_ff @(posedge clk) begin
        if (rst) begin
            s7_valid <= 0;
        end else begin
            s7_valid <= s6_valid;
        end
    end

    always_ff @(posedge clk) begin
        s7_sign <= s6_sign;
        s7_mant_pre28 <= s6_mant_pre28;
        s7_exp_aligned10 <= s6_exp_aligned10;
        s7_lzc <= s6b_lzc;
        s7_result_is_nan <= s6_result_is_nan;
        s7_result_is_inf <= s6_result_is_inf;
    end

    /**
     * 8. Do normalization shift
     */
    logic s8_valid;
    logic s8_sign;
    logic [26:0] s8_mant_lshifted;
    logic [26:0] s8_mant_rshifted;
    logic s8_mant_carry;
    logic signed [9:0] s8_exp_norm10;
    logic s8_result_is_nan;
    logic s8_result_is_inf;

    // lshift takes +1 cycle. it will be available next stage
    wire [26:0] s7_mant_lshifted;
    lshift27 lshift(.clk(clk), .in(s7_mant_pre28[26:0]), .shamt(s7_lzc), .out(s8_mant_lshifted));

    // If carry, we right shift by 1 instead of taking the lshift result
    wire s7_mant_carry = s7_mant_pre28[27];
    wire s7_carry_sticky = s7_mant_pre28[1] | s7_mant_pre28[0];
    wire [26:0] s7_mant_rshifted = {s7_mant_pre28[27:2], s7_carry_sticky};

    // adjust exp after normalization. We can't clamp right now as this can be
    // "saved" from underflow due to rounding increment
    wire signed [9:0] s7_exp_norm10 = s7_mant_carry
        ? s7_exp_aligned10 + 10'sd1
        : s7_exp_aligned10 - $signed({4'b0, s7_lzc});

    always_ff @(posedge clk) begin
        if (rst) begin
            s8_valid <= 0;
        end else begin
            s8_valid <= s7_valid;
        end
    end

    always_ff @(posedge clk) begin
        s8_sign <= s7_sign;
        s8_mant_carry <= s7_mant_carry;
        s8_mant_rshifted <= s7_mant_rshifted;
        s8_exp_norm10 <= s7_exp_norm10;
        s8_result_is_nan <= s7_result_is_nan;
        s8_result_is_inf <= s7_result_is_inf;
    end

    /**
     * 8b. Choose mant_norm27, lshift or rshifted version
     */
    logic s8b_valid;
    logic s8b_sign;
    logic [26:0] s8b_mant_norm27;
    logic signed [9:0] s8b_exp_norm10;
    logic s8b_result_is_nan;
    logic s8b_result_is_inf;

    // Select lshift or rshift result
    wire [26:0] s8_mant_norm27 = s8_mant_carry ? s8_mant_rshifted : s8_mant_lshifted;

    always_ff @(posedge clk) begin
        if (rst) begin
            s8b_valid <= 0;
        end else begin
            s8b_valid <= s8_valid;
        end
    end

    always_ff @(posedge clk) begin
        s8b_sign <= s8_sign;
        s8b_mant_norm27 <= s8_mant_norm27;
        s8b_exp_norm10 <= s8_exp_norm10;
        s8b_result_is_nan <= s8_result_is_nan;
        s8b_result_is_inf <= s8_result_is_inf;
    end


    /**
     * 9. Round (32-bit increment, so gets its own stage)
     */
    logic s9_valid;
    logic s9_sign;
    logic [22:0] s9_mant_rounded;
    logic s9_round_carry;
    logic signed [9:0] s9_exp10;
    logic s9_result_is_nan;
    logic s9_result_is_inf;
    logic s9_mant_is_zero;

    // need this for zero formation downstairs
    wire s8b_mant_is_zero = s8b_mant_norm27 == 0;

    wire [22:0] s8b_mant_pre23 = s8b_mant_norm27[25:3];
    wire s8b_guard = s8b_mant_norm27[2];
    wire s8b_round = s8b_mant_norm27[1];
    wire s8b_sticky = s8b_mant_norm27[0];
    wire s8b_lsb = s8b_mant_norm27[3];

    wire s8b_inc = s8b_guard & (s8b_round | s8b_sticky | s8b_lsb);
    wire [23:0] s8b_mant_rounded = s8b_mant_pre23 + {22'b0, s8b_inc};
    wire s8b_round_carry = s8b_mant_rounded[23];

    always_ff @(posedge clk) begin
        if (rst) begin
            s9_valid <= 0;
        end else begin
            s9_valid <= s8b_valid;
        end
    end

    always_ff @(posedge clk) begin
        s9_sign <= s8b_sign;
        s9_mant_rounded <= s8b_mant_rounded[22:0];
        s9_round_carry <= s8b_round_carry;
        s9_exp10 <= s8b_exp_norm10;
        s9_result_is_nan <= s8b_result_is_nan;
        s9_result_is_inf <= s8b_result_is_inf;
        s9_mant_is_zero <= s8b_mant_is_zero;
    end

    /**
     * 9b. Rounding, part 2
     */
    logic s9b_valid;
    logic s9b_sign_result;
    logic [22:0] s9b_mant_rounded;
    logic signed [9:0] s9b_exp10;
    logic s9b_exp_underflow;
    logic s9b_exp_overflow;
    logic s9b_result_is_nan;
    logic s9b_result_is_inf;
    logic s9b_mant_is_zero;

    wire signed [9:0] s9_exp10_inc = s9_exp10 + $signed({9'b0, s9_round_carry});

    always_ff @(posedge clk) begin
        if (rst) begin
            s9b_valid <= 0;
        end else begin
            s9b_valid <= s9_valid;
        end
    end

    always_ff @(posedge clk) begin
        s9b_sign_result <= s9_sign;
        s9b_mant_rounded <= s9_mant_rounded;
        s9b_exp10 <= s9_exp10_inc;
        s9b_exp_underflow <= s9_exp10_inc <= 0;
        s9b_exp_overflow <= s9_exp10_inc >= 255;
        s9b_result_is_nan <= s9_result_is_nan;
        s9b_result_is_inf <= s9_result_is_inf;
        s9b_mant_is_zero <= s9_mant_is_zero;
    end

    /**
     * 10. Finalize
     */
    logic s10_valid;
    logic s10_sign;
    logic [22:0] s10_mant;
    logic [7:0] s10_exp;

    /* handle inf, nan, FTZ */
    logic s9b_sign;
    logic [7:0] s9b_exp;
    logic [22:0] s9b_mant;
    always_comb begin
        if(s9b_result_is_nan) begin
            s9b_sign = 1'b0;
            s9b_exp = 8'hFF;
            s9b_mant = 23'h400000;
        end else if (s9b_result_is_inf || s9b_exp_overflow) begin
            s9b_sign = s9b_sign_result;
            s9b_exp = 8'hFF;
            s9b_mant = 23'h0;
        end else if (s9b_exp_underflow || s9b_mant_is_zero) begin
            s9b_sign = 1'b0;
            s9b_exp = 8'h0;
            s9b_mant = 23'h0;
        end else begin
            s9b_sign = s9b_sign_result;
            s9b_exp = s9b_exp10[7:0];
            s9b_mant = s9b_mant_rounded[22:0];
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            s10_valid <= 0;
        end else begin
            s10_valid <= s9b_valid;
        end
    end

    always_ff @(posedge clk) begin
        s10_sign <= s9b_sign;
        s10_mant <= s9b_mant;
        s10_exp <= s9b_exp;
    end

    /* output */
    assign out = {s10_sign, s10_exp, s10_mant};
    assign valid_out = s10_valid;
endmodule

/* part of the FPU. compute LZC for 27-bit value */
module lzc27(input wire [26:0] in, output wire [4:0] out);
    function automatic logic [4:0] lzc (input logic [26:0] x);
        logic found;
        logic [4:0] out2;
        int i;

        begin
            found = 1'b0;
            out2 = 5'd27;
            for (i = 26; i >= 0; i--) begin
                if (!found && x[i]) begin
                    out2 = 26 - i;
                    found = 1'b1;
                end
            end

            lzc = out2;
        end
    endfunction

    assign out = lzc(in);
endmodule

/* part of FPU: shift right by up to 27, place
 * sticky = "OR of all bits shifted out" in the
 * last bit of out */
module rshift27 (
    input wire clk,
    input wire [26:0] in,
    input wire [4:0] shamt,
    output wire [26:0] out
);
    logic [4:0] shamt_reg;

    wire [26:0] st0 = shamt[0] ? {1'b0, in[26:1]} : in;
    wire [26:0] st1 = shamt[1] ? {2'b0, st0[26:2]} : st0;
    wire [26:0] st2 = shamt[2] ? {4'b0, st1[26:4]} : st1;
    logic [26:0] st2_reg;
    wire [26:0] st3 = shamt_reg[3] ? {8'b0, st2_reg[26:8]} : st2_reg;
    wire [26:0] st4 = shamt_reg[4] ? {16'b0, st3[26:16]} : st3;

    wire s0 = shamt[0] ? in[0] : 1'b0;
    wire s1 = shamt[1] ? s0 | |st0[1:0] : s0;
    wire s2 = shamt[2] ? s1 | |st1[3:0] : s1;
    logic s2_reg;
    wire s3 = shamt_reg[3] ? s2_reg | |st2_reg[7:0] : s2_reg;
    wire s4 = shamt_reg[4] ? s3 | |st3[15:0] : s3;

    always_ff @(posedge clk) begin
        shamt_reg <= shamt;
        st2_reg <= st2;
        s2_reg <= s2;
    end

    assign out = {st4[26:1], s4 | st4[0]};
endmodule

/* part of FPU: shift left by up to 27 */
module lshift27 (
    input wire clk,
    input wire [26:0] in,
    input wire [4:0] shamt,
    output wire [26:0] out
);
    logic [4:0] shamt_reg;

    wire [26:0] st0 = shamt[0] ? {in[25:0], 1'b0} : in;
    wire [26:0] st1 = shamt[1] ? {st0[24:0], 2'b0} : st0;
    wire [26:0] st2 = shamt[2] ? {st1[22:0], 4'b0} : st1;
    logic [26:0] st2_reg;
    wire [26:0] st3 = shamt_reg[3] ? {st2_reg[18:0], 8'b0} : st2_reg;
    wire [26:0] st4 = shamt_reg[4] ? {st3[10:0], 16'b0} : st3;

    always_ff @(posedge clk) begin
        shamt_reg <= shamt;
        st2_reg <= st2;
    end

    assign out = st4;
endmodule

/* part of FPU: compare two 27 bit numbers in a pipelined
 * fashion (otherwise it generates too long of a carry chain)
 * We only care if a > b or not (is mant_lo > mant_hi?) */
module is_gt27 (
    input wire clk,
    input wire [26:0] a,
    input wire [26:0] b,
    output wire is_gt
);
    logic [9:0] a_lo;
    logic [9:0] b_lo;
    logic is_gt_hi;
    logic is_eq_hi;

    always_ff @(posedge clk) begin
        a_lo <= a[9:0];
        b_lo <= b[9:0];
        is_gt_hi <= a[26:10] > b[26:10];
        is_eq_hi <= a[26:10] == b[26:10];
    end

    wire is_gt_lo = a_lo > b_lo;
    assign is_gt = is_gt_hi || (is_eq_hi && is_gt_lo);
endmodule
