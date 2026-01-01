`timescale 1ns / 1ps

/**
 * 32x32xN register bank using LUTRAM.
 * write_en can be driven by a predicate register */
module rf #(
    parameter LANES = 8
) (
    input logic clk,
    // no RST for LUTRAM
    // Read port A
    input logic [4:0]  rd_addr_a,
    output logic [31:0] rd_data_a [LANES-1:0],
    // Read port B
    input logic [4:0] rd_addr_b,
    output logic [31:0] rd_data_b [LANES-1:0],
    // Read port C
    input logic [4:0] rd_addr_c,
    output logic [31:0] rd_data_c [LANES-1:0],
    // Write port (shared addr, per-lane enables/data)
    input logic [LANES-1:0] wr_en,
    input logic [4:0] wr_addr, // [5:1] reg index, [0] even/odd
    input logic [31:0] wr_data [LANES-1:0]
);
    // R0 is zero register
    wire is_zero_reg = wr_addr == 5'b0;

    // Instantiate lanes
    generate
        for (genvar lane = 0; lane < LANES; lane++) begin : g_lane
            ram32x32 u_bank (
                .clk(clk),
                .rd_addr_a(rd_addr_a),
                .rd_data_a(rd_data_a[lane]),
                .rd_addr_b(rd_addr_b),
                .rd_data_b(rd_data_b[lane]),
                .rd_addr_c(rd_addr_c),
                .rd_data_c(rd_data_c[lane]),
                .wr_en(wr_en[lane] & ~is_zero_reg),  // Disable writes to R0
                .wr_addr(wr_addr),
                .wr_data(wr_data[lane])
            );
        end
    endgenerate
endmodule

/* 32x32 bank using 16 RAM32M primitives in 32x2 quad-port mode (3R1W) */
module ram32x32 (
    input logic clk,
    input logic [4:0] rd_addr_a,
    output logic [31:0] rd_data_a,
    input logic [4:0] rd_addr_b,
    output logic [31:0] rd_data_b,
    input logic [4:0] rd_addr_c,
    output logic [31:0] rd_data_c,
    input logic wr_en,
    input logic [4:0] wr_addr,
    input logic [31:0] wr_data
);
    /* actually, register the outputs. This is free since we are
     * using the LUT section of the slice, and there are 8 FFs per
     * slice, and one RAM32M consumes one slice to provide 6 bits of
     * read so yeah you get the idea */
    logic [31:0] rd_data_a_comb, rd_data_b_comb, rd_data_c_comb;

    always_ff @(posedge clk) begin
        rd_data_a <= rd_data_a_comb;
        rd_data_b <= rd_data_b_comb;
        rd_data_c <= rd_data_c_comb;
    end

    generate
        for (genvar i = 0; i < 16; i++) begin : g_ram
            RAM32M #(
                .INIT_A(64'h0),
                .INIT_B(64'h0),
                .INIT_C(64'h0),
                .INIT_D(64'h0)
            ) u_ram (
                .DOA(rd_data_a_comb[i*2 +: 2]),
                .DOB(rd_data_b_comb[i*2 +: 2]),
                .DOC(rd_data_c_comb[i*2 +: 2]),
                .DOD(),
                .ADDRA(rd_addr_a),
                .ADDRB(rd_addr_b),
                .ADDRC(rd_addr_c),
                .ADDRD(wr_addr),
                .DIA(wr_data[i*2 +: 2]),
                .DIB(wr_data[i*2 +: 2]),
                .DIC(wr_data[i*2 +: 2]),
                .DID(wr_data[i*2 +: 2]),
                .WCLK(clk),
                .WE(wr_en)
            );
        end
    endgenerate
endmodule
