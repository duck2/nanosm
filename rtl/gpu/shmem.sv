`timescale 1ns / 1ps

/** 
 * Lane-interleaved shared memory with bank conflict resolution.
 * addr[2:0] selects bank, addr[11:3] selects row within bank.
 * Total: 8 banks × 512 rows × 32 bits = 16KB
 * 
 * Single arbiter handles both read and write (mutual exclusion).
 * Uses single crossbar for both write (lane→bank) and read (bank→lane).
 */
module shmem #(
    parameter LANES = 8,
    parameter ADDR_W = 12
) (
    input wire clk,
    input wire rst,

    // Write port
    input wire w_valid,
    input wire [ADDR_W-1:0] waddr [LANES-1:0],
    input wire [31:0] wdata [LANES-1:0],
    input wire [LANES-1:0] w_lane_mask,
    output wire w_ready,

    // Read port
    input wire r_valid,
    input wire [ADDR_W-1:0] raddr [LANES-1:0],
    input wire [LANES-1:0] r_lane_mask,
    output wire r_ready,
    output wire r_done,
    output wire [31:0] rdata [LANES-1:0]
);
    localparam BANK_BITS = $clog2(LANES);
    localparam ROW_BITS = ADDR_W - BANK_BITS;

    // ========================================================================
    // Request muxing: write has priority, mutual exclusion
    // ========================================================================
    wire busy;
    wire can_start = !busy;
    wire do_write = w_valid && can_start;
    wire do_read = r_valid && can_start && !w_valid;
    wire arb_start = do_write || do_read;

    wire [ADDR_W-1:0] arb_addr [LANES-1:0];
    wire [LANES-1:0] arb_mask = do_write ? w_lane_mask : r_lane_mask;
    generate
        for (genvar i = 0; i < LANES; i++) begin : g_addr_mux
            assign arb_addr[i] = do_write ? waddr[i] : raddr[i];
        end
    endgenerate

    // Latch is_write on start
    reg is_write_r;
    always_ff @(posedge clk) begin
        if (arb_start)
            is_write_r <= do_write;
    end

    // ========================================================================
    // Arbiter instance
    // ========================================================================
    wire done;
    wire [LANES-1:0] grant;
    wire [LANES-1:0] bank_en;
    wire [ROW_BITS-1:0] bank_addr [LANES-1:0];
    wire [BANK_BITS-1:0] lane_sel_bank [LANES-1:0];
    wire [BANK_BITS-1:0] bank_sel_lane [LANES-1:0];

    shmem_arbiter #(.LANES(LANES), .ADDR_W(ADDR_W)) u_arb (
        .clk(clk), .rst(rst),
        .start(arb_start),
        .addr(arb_addr),
        .lane_mask(arb_mask),
        .busy(busy),
        .done(done),
        .grant(grant),
        .bank_en(bank_en),
        .bank_addr(bank_addr),
        .lane_sel_bank(lane_sel_bank),
        .bank_sel_lane(bank_sel_lane)
    );

    assign w_ready = !busy;
    assign r_ready = !busy;

    // r_done delayed 1 cycle to align with BRAM latency
    reg done_d, is_write_d;
    always_ff @(posedge clk) begin
        done_d <= done;
        is_write_d <= is_write_r;
    end
    assign r_done = done_d && !is_write_d;

    // ========================================================================
    // Latch lane_sel_bank when granted (for read data routing after BRAM latency)
    // ========================================================================
    reg [BANK_BITS-1:0] lane_sel_bank_r [LANES-1:0];
    reg [LANES-1:0] grant_d;

    always_ff @(posedge clk) begin
        grant_d <= grant;
        for (int l = 0; l < LANES; l++)
            if (grant[l])
                lane_sel_bank_r[l] <= lane_sel_bank[l];
    end

    // ========================================================================
    // Unified data latch: captures wdata (from outside) or rdata (from xbar)
    // ========================================================================
    reg [31:0] data_r [LANES-1:0];
    reg [LANES-1:0] lane_captured;

    // Forward declaration for xbar_out (used in data_r capture)
    wire [31:0] xbar_out [LANES-1:0];

    always_ff @(posedge clk) begin
        if (rst) begin
            lane_captured <= '0;
        end else begin
            // Capture wdata on write start
            if (arb_start && do_write) begin
                for (int i = 0; i < LANES; i++)
                    data_r[i] <= wdata[i];
            end
            // Clear lane_captured on read start
            if (arb_start && !do_write)
                lane_captured <= '0;
            // Capture rdata as lanes complete
            for (int l = 0; l < LANES; l++) begin
                if (grant_d[l] && !lane_captured[l]) begin
                    data_r[l] <= xbar_out[l];
                    lane_captured[l] <= 1'b1;
                end
            end
        end
    end

    // ========================================================================
    // Crossbar: shared for writes (lane→bank) and reads (bank→lane)
    // ========================================================================
    wire [31:0] bank_rdata [LANES-1:0];

    // Crossbar inputs: data_r for writes, bank_rdata for reads
    wire [31:0] xbar_in [LANES-1:0];
    generate
        for (genvar i = 0; i < LANES; i++) begin : g_xbar_in
            assign xbar_in[i] = is_write_r ? data_r[i] : bank_rdata[i];
        end
    endgenerate

    // Crossbar select: 3-bit index per output
    // Write: sel[bank] = which lane to read from
    // Read:  sel[lane] = which bank to read from (latched when granted)
    wire [BANK_BITS-1:0] xbar_sel [LANES-1:0];
    generate
        for (genvar o = 0; o < LANES; o++) begin : g_xbar_sel
            assign xbar_sel[o] = is_write_r ? bank_sel_lane[o] : lane_sel_bank_r[o];
        end
    endgenerate

    // Crossbar output (forward-declared above)
    shmem_xbar #(.LANES(LANES)) u_xbar (
        .sel(xbar_sel),
        .din(xbar_in),
        .dout(xbar_out)
    );

    // ========================================================================
    // Bank write path
    // ========================================================================
    wire [LANES-1:0] bank_we;
    wire [31:0] bank_wdata [LANES-1:0];
    wire [LANES-1:0] bank_re;

    generate
        for (genvar b = 0; b < LANES; b++) begin : g_bank_ctrl
            assign bank_we[b] = is_write_r && bank_en[b];
            assign bank_wdata[b] = xbar_out[b];
            assign bank_re[b] = bank_en[b] && !is_write_r;
        end
    endgenerate

    // Output: use data_r for captured lanes, fresh xbar_out for just-completed
    generate
        for (genvar l = 0; l < LANES; l++) begin : g_rout
            assign rdata[l] = lane_captured[l] ? data_r[l] : xbar_out[l];
        end
    endgenerate

    // ========================================================================
    // Bank instances
    // ========================================================================
    generate
        for (genvar b = 0; b < LANES; b++) begin : banks
            ram512x32 u_bank (
                .clk(clk),
                .we(bank_we[b]),
                .waddr(bank_addr[b]),
                .wdata(bank_wdata[b]),
                .re(bank_re[b]),
                .raddr(bank_addr[b]),
                .rdata(bank_rdata[b])
            );
        end
    endgenerate

endmodule


/** N×N crossbar: each output selects one input via 3-bit index */
module shmem_xbar #(
    parameter LANES = 8,
    parameter SEL_W = $clog2(LANES)
) (
    input wire [SEL_W-1:0] sel [LANES-1:0],
    input wire [31:0] din [LANES-1:0],
    output wire [31:0] dout [LANES-1:0]
);
    generate
        for (genvar o = 0; o < LANES; o++) begin : g_out
            assign dout[o] = din[sel[o]];
        end
    endgenerate
endmodule


/** Simple dual-port 512x32 RAM. Maps to one RAMB18E1. */
module ram512x32 (
    input wire clk,
    input wire we,
    input wire [8:0] waddr,
    input wire [31:0] wdata,
    input wire re,
    input wire [8:0] raddr,
    output reg [31:0] rdata
);
    (* ram_style = "block" *)
    reg [31:0] mem [511:0];

    always_ff @(posedge clk) begin
        if (we) mem[waddr] <= wdata;
    end

    always_ff @(posedge clk) begin
        if (re) rdata <= mem[raddr];
    end
endmodule
