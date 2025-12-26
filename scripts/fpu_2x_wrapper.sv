module fpu_2x_wrapper (
    input wire clk100,
    input wire rst
);
    // Input buffer for 100 MHz board clock
    wire clk100_ibuf;
    IBUF u_ibuf_clk100 (
        .I(clk100),
        .O(clk100_ibuf)
    );

    // Put the 100 MHz onto a global clock net
    wire clk100_bufg;
    BUFG u_bufg_clk100 (
        .I(clk100_ibuf),
        .O(clk100_bufg)
    );

    // MMCM signals
    wire clkfb_mmcm;
    wire clkfb_bufg;
    wire clk125;
    wire clk250;
    wire mmcm_locked;

    // MMCM feedback BUFG
    BUFG u_bufg_clkfb (
        .I(clkfb_mmcm),
        .O(clkfb_bufg)
    );

    /* Derive 125 and 250 MHz clocks from clk100 */
    MMCME2_BASE #(
        .BANDWIDTH("OPTIMIZED"),
        .CLKFBOUT_MULT_F(10.0),  // 100 x 10 -> 1000 MHz
        .CLKFBOUT_PHASE(0.0),
        .CLKIN1_PERIOD(10.0),  // 100 MHz clock (10 ns)
        // CLKOUT0_DIVIDE - CLKOUT6_DIVIDE: Divide amount for each CLKOUT (1-128)
        .CLKOUT1_DIVIDE(4.0),  // 1000 / 4 -> 250 MHz
        .CLKOUT2_DIVIDE(1),
        .CLKOUT3_DIVIDE(1),
        .CLKOUT4_DIVIDE(1),
        .CLKOUT5_DIVIDE(1),
        .CLKOUT6_DIVIDE(1),
        .CLKOUT0_DIVIDE_F(8.0),  // 1000 / 8 -> 125 MHz
        // CLKOUT0_DUTY_CYCLE - CLKOUT6_DUTY_CYCLE: Duty cycle for each CLKOUT (0.01-0.99).
        .CLKOUT0_DUTY_CYCLE(0.5),
        .CLKOUT1_DUTY_CYCLE(0.5),
        .CLKOUT2_DUTY_CYCLE(0.5),
        .CLKOUT3_DUTY_CYCLE(0.5),
        .CLKOUT4_DUTY_CYCLE(0.5),
        .CLKOUT5_DUTY_CYCLE(0.5),
        .CLKOUT6_DUTY_CYCLE(0.5),
        // CLKOUT0_PHASE - CLKOUT6_PHASE: Phase offset for each CLKOUT (-360.000-360.000).
        .CLKOUT0_PHASE(0.0),
        .CLKOUT1_PHASE(0.0),
        .CLKOUT2_PHASE(0.0),
        .CLKOUT3_PHASE(0.0),
        .CLKOUT4_PHASE(0.0),
        .CLKOUT5_PHASE(0.0),
        .CLKOUT6_PHASE(0.0),
        .CLKOUT4_CASCADE("FALSE"), // Cascade CLKOUT4 counter with CLKOUT6 (FALSE, TRUE)
        .DIVCLK_DIVIDE(1),         // Master division value (1-106)
        .REF_JITTER1(0.0),         // Reference input jitter in UI (0.000-0.999).
        .STARTUP_WAIT("FALSE")     // Delays DONE until MMCM is locked (FALSE, TRUE)
    )
    mmcm_inst (
        // Clock Outputs: 1-bit (each) output: User configurable clock outputs
        .CLKOUT0(clk125_mmcm),
        .CLKOUT0B(),
        .CLKOUT1(clk250_mmcm),
        .CLKOUT1B(),
        .CLKOUT2(),
        .CLKOUT2B(),
        .CLKOUT3(),
        .CLKOUT3B(),
        .CLKOUT4(),
        .CLKOUT5(),
        .CLKOUT6(),
        // Feedback Clocks: 1-bit (each) output: Clock feedback ports
        .CLKFBOUT(clkfb_mmcm),   // 1-bit output: Feedback clock
        .CLKFBOUTB(), // 1-bit output: Inverted CLKFBOUT
        // Status Ports: 1-bit (each) output: MMCM status ports
        .LOCKED(mmcm_locked),       // 1-bit output: LOCK
        // Clock Inputs: 1-bit (each) input: Clock input
        .CLKIN1(clk100_bufg),       // 1-bit input: Clock
        // Control Ports: 1-bit (each) input: MMCM control ports
        .PWRDWN(1'b0),       // 1-bit input: Power-down
        .RST(rst),             // 1-bit input: Reset
        // Feedback Clocks: 1-bit (each) input: Clock feedback ports
        .CLKFBIN(clkfb_bufg)
    );

    // BUFG for clk125 and clk250
    wire clk125_bufg, clk250_bufg;
    BUFG u_bufg_clk_sys (
        .I(clk125_mmcm),
        .O(clk125_bufg)
    );
    BUFG u_bufg_clk_fpu2 (
        .I(clk250_mmcm),
        .O(clk250_bufg)
    );

    // INTERNAL REG INPUTS (not top-level ports)
    (* keep = "true" *) reg [31:0] in1_0_reg = 32'h3F800000; // 1.0
    (* keep = "true" *) reg [31:0] in1_1_reg = 32'h40000000; // 2.0
    (* keep = "true" *) reg [31:0] in2_0_reg = 32'h40400000; // 3.0
    (* keep = "true" *) reg [31:0] in2_1_reg = 32'h40800000; // 4.0
    (* keep = "true" *) reg [31:0] in3_0_reg = 32'h40A00000; // 5.0
    (* keep = "true" *) reg [31:0] in3_1_reg = 32'h40C00000; // 6.0
    (* keep = "true" *) reg valid_0_reg = 1'b1;
    (* keep = "true" *) reg valid_1_reg = 1'b1;

    // Make the regs toggle slowly so they aren't constants
    always @(posedge clk125_bufg) begin
        if (!rst) begin
            in1_0_reg <= {in1_0_reg[30:0], in1_0_reg[31]};
            in1_1_reg <= in1_1_reg + 32'h01010101;
            in2_0_reg <= {in2_0_reg[0], in2_0_reg[31:1]};
            in2_1_reg <= in2_1_reg ^ 32'hDEADBEEF;
            in3_0_reg <= in3_0_reg + 32'h00010001;
            in3_1_reg <= {in3_1_reg[30:0], in3_1_reg[31]};
            valid_0_reg <= ~valid_0_reg;
            valid_1_reg <= valid_1_reg;
        end
    end

    // FPU outputs
    (* keep = "true" *) wire [31:0] out_0_wire;
    (* keep = "true" *) wire [31:0] out_1_wire;
    (* keep = "true" *) wire valid_out_0_wire;
    (* keep = "true" *) wire valid_out_1_wire;

    (* dont_touch = "true" *)
    fpu_2x fpu_2x_inst (
        .clk(clk125_bufg),
        .clk_2x(clk250_bufg),
        .rst(rst),
        .in1_0(in1_0_reg),
        .in1_1(in1_1_reg),
        .in2_0(in2_0_reg),
        .in2_1(in2_1_reg),
        .in3_0(in3_0_reg),
        .in3_1(in3_1_reg),
        .valid_in_0(valid_0_reg),
        .valid_in_1(valid_1_reg),
        .out_0(out_0_wire),
        .out_1(out_1_wire),
        .valid_out_0(valid_out_0_wire),
        .valid_out_1(valid_out_1_wire)
    );

    // Keep outputs alive
    (* keep = "true" *) reg [31:0] sink_0_reg;
    (* keep = "true" *) reg [31:0] sink_1_reg;
    (* keep = "true" *) reg sink_valid_0_reg;
    (* keep = "true" *) reg sink_valid_1_reg;

    always @(posedge clk125_bufg) begin
        sink_0_reg <= out_0_wire;
        sink_1_reg <= out_1_wire;
        sink_valid_0_reg <= valid_out_0_wire;
        sink_valid_1_reg <= valid_out_1_wire;
    end

endmodule
