`timescale 1ns / 1ps

/**
 * LSU (Load-Store Unit) - LANES-bank BRAM memory controller
 *
 * Design Context (A7-35T-1L resource constraints):
 * ================================================
 * One clock region has ~20 BRAMs, ~20 DSPs, ~1000 slices.
 * We use LANES/2 BRAM36 (= LANES BRAM18) in 512x32 SDP mode for LANES banks.
 * 
 * The key insight is that BRAM serves dual purposes:
 *   1. External memory interface (vector load/store)
 *   2. Internal scratch space for lane-serial operations
 *
 * FMA Streaming Problem:
 * ======================
 * Lane-serial FPU needs 3 operands per cycle but RF has only 2 read ports.
 * Solution: stage operands in BRAM with bank interleaving.
 *
 * Load x1 with offset 0:  x1[i] → bank i
 * Load x2 with offset 5:  x2[i] → bank (i+5) % LANES
 * Load x3 with offset 10: x3[i] → bank (i+10) % LANES
 *
 * Now streaming cycle i can read 3 different banks simultaneously:
 *   bank i       → x1[i]
 *   bank (i+5)   → x2[i]
 *   bank (i+10)  → x3[i]
 *
 * No conflicts because 0, 5, 10 are distinct mod LANES.
 *
 * This LSU exposes two modes:
 *   1. VECTOR: LANES-wide load/store, 1 cycle aligned, 2 cycles unaligned
 *      - Supports rotation (bank offset) for interleaved staging
 *   2. INDEXED: Up to 3 simultaneous bank accesses (must be conflict-free)
 *      - Used by stream_feeder for FMA operand reads
 *
 * Vector Shuffling Bonus:
 * =======================
 * The rotation parameter enables cheap vector permutations:
 *   - Load with rotation R, store with rotation 0 = rotate left by R
 *   - Combined with indexed mode, can do arbitrary lane shuffles
 */

module lsu #(
    parameter LANES = 8
) (
    input  logic clk,
    input  logic rst,

    // === Vector Access Interface ===
    input  logic vec_req,
    output logic vec_ready,
    output logic vec_done,
    input  logic vec_we,
    input  logic [13:0] vec_addr,       // Word address [13:4]=row, [3:0]=bank_start
    input  logic [$clog2(LANES)-1:0] vec_rotation,    // Output rotation (element i → slot (i+rot)%LANES)
    input  logic [31:0] vec_wdata [LANES-1:0],
    output logic [31:0] vec_rdata [LANES-1:0],

    // === Indexed Access Interface (up to 3 ports) ===
    input  logic idx_req,
    output logic idx_ready,
    output logic idx_done,
    input  logic idx_we,
    input  logic [1:0] idx_count,       // 0=1 access, 1=2 accesses, 2=3 accesses
    input  logic [$clog2(LANES)-1:0] idx_bank [2:0],  // Which banks (caller ensures no conflicts)
    input  logic [9:0] idx_row [2:0],   // Which rows
    input  logic [31:0] idx_wdata [2:0],
    output logic [31:0] idx_rdata [2:0],

    // === BRAM Interface (directly to LANES banks) ===
    output logic bram_en [LANES-1:0],
    output logic bram_we [LANES-1:0],
    output logic [9:0] bram_addr [LANES-1:0],
    output logic [31:0] bram_wdata [LANES-1:0],
    input  logic [31:0] bram_rdata [LANES-1:0]
);

    typedef enum logic [2:0] {
        IDLE,
        VEC_CYCLE1,     // First vector access (all banks or partial)
        VEC_CYCLE2,     // Second vector access (for unaligned)
        VEC_DONE,       // Capture vector result
        IDX_CYCLE,      // Issue indexed access
        IDX_DONE        // Capture indexed result
    } state_t;

    state_t state, state_next;

    // Registered request parameters
    logic vec_we_r;
    logic [13:0] vec_addr_r;
    logic [$clog2(LANES)-1:0] vec_rotation_r;
    logic [31:0] vec_wdata_r [LANES-1:0];

    logic idx_we_r;
    logic [1:0] idx_count_r;
    logic [$clog2(LANES)-1:0] idx_bank_r [2:0];
    logic [9:0] idx_row_r [2:0];
    logic [31:0] idx_wdata_r [2:0];

    // Derived signals
    localparam LANE_BITS = $clog2(LANES);
    logic [LANE_BITS-1:0] bank_offset;
    logic [9:0] row_base;
    logic is_aligned;
    assign bank_offset = vec_addr_r[LANE_BITS-1:0];
    assign row_base = vec_addr_r[13:LANE_BITS];
    assign is_aligned = (bank_offset == 0);

    // Intermediate storage for unaligned access
    logic [31:0] vec_rdata_cycle1 [LANES-1:0];

    // State machine
    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            vec_we_r <= 0;
            idx_we_r <= 0;
            idx_count_r <= 0;
            for (int i = 0; i < LANES; i++) begin
                vec_wdata_r[i] <= 32'd0;
                vec_rdata[i] <= 32'd0;
                vec_rdata_cycle1[i] <= 32'd0;
            end
            for (int i = 0; i < 3; i++) begin
                idx_bank_r[i] <= 4'd0;
                idx_row_r[i] <= 10'd0;
                idx_wdata_r[i] <= 32'd0;
                idx_rdata[i] <= 32'd0;
            end
        end else begin
            state <= state_next;

            // Latch vector request
            if (vec_req && vec_ready) begin
                vec_we_r <= vec_we;
                vec_addr_r <= vec_addr;
                vec_rotation_r <= vec_rotation;
                for (int i = 0; i < LANES; i++)
                    vec_wdata_r[i] <= vec_wdata[i];
            end

            // Latch indexed request
            if (idx_req && idx_ready) begin
                idx_we_r <= idx_we;
                idx_count_r <= idx_count;
                for (int i = 0; i < 3; i++) begin
                    idx_bank_r[i] <= idx_bank[i];
                    idx_row_r[i] <= idx_row[i];
                    idx_wdata_r[i] <= idx_wdata[i];
                end
            end

            // Capture cycle 1 results for unaligned
            if (state == VEC_CYCLE1 && !vec_we_r) begin
                for (int i = 0; i < LANES; i++)
                    vec_rdata_cycle1[i] <= bram_rdata[i];
            end

            // Final vector result capture with rotation
            if (state == VEC_DONE && !vec_we_r) begin
                for (int i = 0; i < LANES; i++) begin
                    automatic logic [LANE_BITS:0] src_idx = (LANE_BITS+1)'(i) + (LANE_BITS+1)'(bank_offset);
                    automatic logic [LANE_BITS-1:0] rot_idx = (LANE_BITS'(i) - vec_rotation_r) & (LANES-1);
                    if (is_aligned) begin
                        vec_rdata[rot_idx] <= vec_rdata_cycle1[i];
                    end else begin
                        // Unaligned: combine data from two rows
                        if (src_idx < LANES)
                            vec_rdata[rot_idx] <= vec_rdata_cycle1[src_idx[LANE_BITS-1:0]];
                        else
                            vec_rdata[rot_idx] <= bram_rdata[src_idx[LANE_BITS-1:0]];
                    end
                end
            end

            // Capture indexed results
            if (state == IDX_DONE && !idx_we_r) begin
                for (int i = 0; i < 3; i++) begin
                    if (i[1:0] <= idx_count_r)
                        idx_rdata[i] <= bram_rdata[idx_bank_r[i]];
                end
            end
        end
    end

    // Next state and output logic
    always_comb begin
        state_next = state;
        vec_ready = 1'b0;
        vec_done = 1'b0;
        idx_ready = 1'b0;
        idx_done = 1'b0;

        for (int i = 0; i < LANES; i++) begin
            bram_en[i] = 1'b0;
            bram_we[i] = 1'b0;
            bram_addr[i] = 10'd0;
            bram_wdata[i] = 32'd0;
        end

        case (state)
            IDLE: begin
                vec_ready = 1'b1;
                idx_ready = 1'b1;
                if (vec_req)
                    state_next = VEC_CYCLE1;
                else if (idx_req)
                    state_next = IDX_CYCLE;
            end

            VEC_CYCLE1: begin
                // Issue first row access
                for (int i = 0; i < LANES; i++) begin
                    if (is_aligned || ((LANE_BITS+1)'(i) + (LANE_BITS+1)'(bank_offset)) < LANES) begin
                        bram_en[i] = 1'b1;
                        bram_we[i] = vec_we_r;
                        bram_addr[i] = row_base;
                        bram_wdata[i] = vec_wdata_r[(LANE_BITS'(i) - vec_rotation_r) & (LANES-1)];
                    end
                end
                state_next = is_aligned ? VEC_DONE : VEC_CYCLE2;
            end

            VEC_CYCLE2: begin
                // Issue second row access (for unaligned)
                for (int i = 0; i < LANES; i++) begin
                    if (((LANE_BITS+1)'(i) + (LANE_BITS+1)'(bank_offset)) >= LANES) begin
                        bram_en[i] = 1'b1;
                        bram_we[i] = vec_we_r;
                        bram_addr[i] = row_base + 10'd1;
                        bram_wdata[i] = vec_wdata_r[(LANE_BITS'(i) - vec_rotation_r) & (LANES-1)];
                    end
                end
                state_next = VEC_DONE;
            end

            VEC_DONE: begin
                vec_done = 1'b1;
                state_next = IDLE;
            end

            IDX_CYCLE: begin
                // Issue indexed accesses (up to 3 banks)
                for (int i = 0; i < 3; i++) begin
                    if (i[1:0] <= idx_count_r) begin
                        bram_en[idx_bank_r[i]] = 1'b1;
                        bram_we[idx_bank_r[i]] = idx_we_r;
                        bram_addr[idx_bank_r[i]] = idx_row_r[i];
                        bram_wdata[idx_bank_r[i]] = idx_wdata_r[i];
                    end
                end
                state_next = IDX_DONE;
            end

            IDX_DONE: begin
                idx_done = 1'b1;
                state_next = IDLE;
            end

            default: state_next = IDLE;
        endcase
    end

endmodule
