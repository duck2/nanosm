`timescale 1ns / 1ps

/** Shared memory arbiter with broadcast detection.
 *  Broadcast compares all lanes to lane 0 to simplify logic (even if lane 0 is masked)
 *  Grant includes inactive lanes on broadcast. The user should handle write masking in RF */
module shmem_arbiter #(
    parameter LANES = 8,
    parameter ADDR_W = 12
) (
    input wire clk,
    input wire rst,

    input wire start,
    input wire [ADDR_W-1:0] addr [LANES-1:0],
    input wire [LANES-1:0] lane_mask,

    output wire busy,
    output wire done,
    output wire [LANES-1:0] grant,
    output wire [LANES-1:0] bank_en,
    output wire [ROW_BITS-1:0] bank_addr [LANES-1:0],
    output wire [BANK_BITS-1:0] lane_sel_bank [LANES-1:0],  // xbar: lane L reads from bank
    output wire [BANK_BITS-1:0] bank_sel_lane [LANES-1:0]   // xbar: bank B writes from lane
);
    localparam BANK_BITS = $clog2(LANES);
    localparam ROW_BITS = ADDR_W - BANK_BITS;

    // State
    logic [LANES-1:0] lane_done;
    logic [ADDR_W-1:0] addr_r [LANES-1:0];
    logic [LANES-1:0] mask_r;
    logic active;

    assign busy = active;
    wire [LANES-1:0] remaining = mask_r & ~lane_done;

    // Address decomposition
    wire [BANK_BITS-1:0] lane_bank [LANES-1:0];
    wire [ROW_BITS-1:0] lane_row [LANES-1:0];
    generate
        for (genvar l = 0; l < LANES; l++) begin : g_addr
            assign lane_bank[l] = addr_r[l][BANK_BITS-1:0];
            assign lane_row[l] = addr_r[l][ADDR_W-1:BANK_BITS];
            assign lane_sel_bank[l] = lane_bank[l];
        end
    endgenerate

    // Broadcast: all addresses identical?
    wire [LANES-1:0] same_as_0;
    generate
        for (genvar l = 0; l < LANES; l++) begin : g_bc
            assign same_as_0[l] = addr_r[l] == addr_r[0];
        end
    endgenerate
    wire is_broadcast = &same_as_0;

    // Per-bank: which remaining lanes target this bank? One-hot winner.
    wire [LANES-1:0] bank_targets [LANES-1:0];
    wire [LANES-1:0] bank_winner [LANES-1:0];
    wire [LANES-1:0] bank_has_winner;

    generate
        for (genvar b = 0; b < LANES; b++) begin : g_bank
            for (genvar l = 0; l < LANES; l++) begin : g_tgt
                assign bank_targets[b][l] = remaining[l] && (lane_bank[l] == b[BANK_BITS-1:0]);
            end

            // Priority encode: first lane wins (prefix-OR)
            wire [LANES:0] prefix;
            assign prefix[0] = 1'b0;
            for (genvar l = 0; l < LANES; l++) begin : g_pfx
                assign prefix[l+1] = prefix[l] | bank_targets[b][l];
            end
            for (genvar l = 0; l < LANES; l++) begin : g_win
                assign bank_winner[b][l] = bank_targets[b][l] & ~prefix[l];
            end
            assign bank_has_winner[b] = prefix[LANES];

            // Bank address: one-hot mux via masked lanes then cascading OR
            assign bank_en[b] = active && bank_has_winner[b];
            wire [ROW_BITS-1:0] row_masked [LANES-1:0];
            wire [ROW_BITS-1:0] row_or [LANES:0];
            assign row_or[0] = '0;
            for (genvar l = 0; l < LANES; l++) begin : g_mask
                assign row_masked[l] = {ROW_BITS{bank_winner[b][l]}} & lane_row[l];
                assign row_or[l+1] = row_or[l] | row_masked[l];
            end
            assign bank_addr[b] = row_or[LANES];

            // One-hot to binary encoder for winning lane index
            wire [BANK_BITS-1:0] widx;
            assign widx[0] = bank_winner[b][1] | bank_winner[b][3] | bank_winner[b][5] | bank_winner[b][7];
            assign widx[1] = bank_winner[b][2] | bank_winner[b][3] | bank_winner[b][6] | bank_winner[b][7];
            assign widx[2] = bank_winner[b][4] | bank_winner[b][5] | bank_winner[b][6] | bank_winner[b][7];
            assign bank_sel_lane[b] = widx;
        end
    endgenerate

    // Grant: OR of bank winners per lane
    wire [LANES-1:0] normal_grant;
    generate
        for (genvar l = 0; l < LANES; l++) begin : g_grant
            wire [LANES-1:0] won_per_bank;
            for (genvar b = 0; b < LANES; b++) begin : g_wb
                assign won_per_bank[b] = bank_winner[b][l];
            end
            assign normal_grant[l] = |won_per_bank;
        end
    endgenerate

    assign grant = active ? (is_broadcast ? {LANES{1'b1}} : normal_grant) : '0;

    // Done: broadcast instant, otherwise when all remaining served
    wire [LANES-1:0] lane_done_next = lane_done | grant;
    assign done = active && (is_broadcast || ((mask_r & ~lane_done_next) == '0));

    // State machine
    always_ff @(posedge clk) begin
        if (rst) begin
            active <= 1'b0;
            lane_done <= '0;
            mask_r <= '0;
        end else if (!active && start && |lane_mask) begin
            active <= 1'b1;
            lane_done <= '0;
            mask_r <= lane_mask;
            for (int i = 0; i < LANES; i++)
                addr_r[i] <= addr[i];
        end else if (active) begin
            lane_done <= lane_done_next;
            if (is_broadcast || ((mask_r & ~lane_done_next) == '0))
                active <= 1'b0;
        end
    end

endmodule
