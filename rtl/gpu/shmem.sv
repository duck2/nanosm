`timescale 1ns / 1ps

/** Shared memory: LANES x ram512x32 = 512 rows × (LANES × 32) bits. */
module shmem #(
    parameter LANES = 8
) (
    input  logic clk,
    // Write port - full LANES-lane vector
    input  logic we,
    input  logic [8:0] waddr,
    input  logic [31:0] wdata [LANES-1:0],
    // Read port - full LANES-lane vector
    input  logic re,
    input  logic [8:0] raddr,
    output logic [31:0] rdata [LANES-1:0]
);
    // Instantiate LANES banks
    generate
        for (genvar lane = 0; lane < LANES; lane++) begin : banks
            ram512x32 bank (
                .clk(clk),
                .we(we),
                .waddr(waddr),
                .wdata(wdata[lane]),
                .re(re),
                .raddr(raddr),
                .rdata(rdata[lane])
            );
        end
    endgenerate
endmodule

/** Simple dual-port 512x32 RAM. maps to one RAMB18E1 */
module ram512x32 (
    input  logic clk,
    // Write port
    input  logic we,
    input  logic [8:0] waddr,
    input  logic [31:0] wdata,
    // Read port
    input  logic re,
    input  logic [8:0] raddr,
    output logic [31:0] rdata
);

    (* ram_style = "block" *)
    logic [31:0] mem [511:0];

    always_ff @(posedge clk) begin
        if (we)
            mem[waddr] <= wdata;
    end

    always_ff @(posedge clk) begin
        if (re)
            rdata <= mem[raddr];
    end

endmodule
