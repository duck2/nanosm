`timescale 1ns / 1ps

module top (
    input wire clk100,
    input wire rst_n,
    output wire led
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
    wire clk125_mmcm;
    wire clk250_mmcm;
    wire mmcm_locked;

    // MMCM feedback BUFG
    BUFG u_bufg_clkfb (
        .I(clkfb_mmcm),
        .O(clkfb_bufg)
    );

    // Derive 125 MHz and 250 MHz clocks from clk100
    MMCME2_BASE #(
        .BANDWIDTH("OPTIMIZED"),
        .CLKFBOUT_MULT_F(10.0),
        .CLKFBOUT_PHASE(0.0),
        .CLKIN1_PERIOD(10.0),
        .CLKOUT0_DIVIDE_F(8.0),
        .CLKOUT1_DIVIDE(4),
        .CLKOUT2_DIVIDE(1),
        .CLKOUT3_DIVIDE(1),
        .CLKOUT4_DIVIDE(1),
        .CLKOUT5_DIVIDE(1),
        .CLKOUT6_DIVIDE(1),
        .CLKOUT0_DUTY_CYCLE(0.5),
        .CLKOUT1_DUTY_CYCLE(0.5),
        .CLKOUT2_DUTY_CYCLE(0.5),
        .CLKOUT3_DUTY_CYCLE(0.5),
        .CLKOUT4_DUTY_CYCLE(0.5),
        .CLKOUT5_DUTY_CYCLE(0.5),
        .CLKOUT6_DUTY_CYCLE(0.5),
        .CLKOUT0_PHASE(0.0),
        .CLKOUT1_PHASE(0.0),
        .CLKOUT2_PHASE(0.0),
        .CLKOUT3_PHASE(0.0),
        .CLKOUT4_PHASE(0.0),
        .CLKOUT5_PHASE(0.0),
        .CLKOUT6_PHASE(0.0),
        .CLKOUT4_CASCADE("FALSE"),
        .DIVCLK_DIVIDE(1),
        .REF_JITTER1(0.0),
        .STARTUP_WAIT("FALSE")
    ) u_mmcm (
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
        .CLKFBOUT(clkfb_mmcm),
        .CLKFBOUTB(),
        .LOCKED(mmcm_locked),
        .CLKIN1(clk100_bufg),
        .PWRDWN(1'b0),
        .RST(~rst_n),
        .CLKFBIN(clkfb_bufg)
    );

    // BUFG for clk125 and clk250
    wire clk125_bufg, clk250_bufg;
    BUFG u_bufg_clk125 (
        .I(clk125_mmcm),
        .O(clk125_bufg)
    );
    BUFG u_bufg_clk250 (
        .I(clk250_mmcm),
        .O(clk250_bufg)
    );

    // Internal reset synchronized to clk125
    wire rst_internal = ~rst_n | ~mmcm_locked;

    // GPU Cluster
    cluster u_cluster (
        .clk(clk125_bufg),
        .clk_2x(clk250_bufg),
        .rst(rst_internal)
    );

    // Heartbeat LED to show design is alive
    reg [26:0] led_counter;
    always @(posedge clk125_bufg) begin
        if (rst_internal)
            led_counter <= 27'd0;
        else
            led_counter <= led_counter + 1;
    end
    assign led = led_counter[26];

endmodule
