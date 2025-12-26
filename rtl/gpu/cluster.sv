`timescale 1ns / 1ps

module cluster #(
    parameter LANES = 8
) (
    input logic clk,
    input logic clk_2x,
    input logic rst
);

    // ========================================================================
    // Mock controller signals (would come from instruction decoder/scheduler)
    // ========================================================================

    // RF addresses
    logic [4:0] rf_rd_addr_a;
    logic [4:0] rf_rd_addr_b;
    logic [4:0] rf_rd_addr_c;
    logic [4:0] rf_wr_addr;
    logic [LANES-1:0] rf_wr_en;

    // WB mux select: 0=ALU, 1=Shifter, 2=FPU, 3=SHMEM
    logic [1:0] wb_sel;

    // ALU opcode (shared across all LANES ALUs)
    logic [4:0] alu_op;

    // Shifter opcode (shared across all LANES/2 shifter_2x units)
    logic [1:0] shifter_op;

    // SHMEM control
    logic shmem_we;
    logic shmem_re;
    logic [8:0] shmem_waddr;
    logic [8:0] shmem_raddr;

    // Mock controller: just toggle things to prevent optimization
    always_ff @(posedge clk) begin
        if (rst) begin
            rf_rd_addr_a <= 5'd0;
            rf_rd_addr_b <= 5'd0;
            rf_rd_addr_c <= 5'd0;
            rf_wr_addr <= 5'd0;
            rf_wr_en <= {LANES{1'b1}};
            wb_sel <= 2'd0;
            alu_op <= 5'd0;
            shifter_op <= 2'd0;
            shmem_we <= 1'b0;
            shmem_re <= 1'b1;
            shmem_waddr <= 9'd0;
            shmem_raddr <= 9'd0;
        end else begin
            rf_rd_addr_a <= rf_rd_addr_a + 1;
            rf_rd_addr_b <= rf_rd_addr_b + 1;
            rf_rd_addr_c <= rf_rd_addr_c + 1;
            rf_wr_addr <= rf_wr_addr + 1;
            wb_sel <= wb_sel + 1;
            alu_op <= alu_op + 1;
            shifter_op <= shifter_op + 1;
            shmem_waddr <= shmem_waddr + 1;
            shmem_raddr <= shmem_raddr + 1;
        end
    end

    // ========================================================================
    // RF read data buses (LANES lanes × 32 bits each)
    // ========================================================================
    logic [31:0] rf_rd_data_a [LANES-1:0];
    logic [31:0] rf_rd_data_b [LANES-1:0];
    logic [31:0] rf_rd_data_c [LANES-1:0];

    // ========================================================================
    // Execution unit outputs
    // ========================================================================
    wire [31:0] alu_out [LANES-1:0];
    wire [31:0] shifter_out [LANES-1:0];
    wire [31:0] fpu_out [LANES-1:0];
    wire fpu_valid_out [LANES-1:0];
    wire [31:0] shmem_out [LANES-1:0];

    // ========================================================================
    // Writeback mux: select which unit writes to RF
    // ========================================================================
    logic [31:0] wb_data [LANES-1:0];

    always_comb begin
        for (int lane = 0; lane < LANES; lane++) begin
            case (wb_sel)
                2'd0: wb_data[lane] = alu_out[lane];
                2'd1: wb_data[lane] = shifter_out[lane];
                2'd2: wb_data[lane] = fpu_out[lane];
                2'd3: wb_data[lane] = shmem_out[lane];
            endcase
        end
    end

    // ========================================================================
    // Register File (3 read ports, 1 write port, LANES lanes)
    // ========================================================================
    (* dont_touch = "true" *)
    rf #(.LANES(LANES)) u_rf (
        .clk(clk),
        .rd_addr_a(rf_rd_addr_a),
        .rd_data_a(rf_rd_data_a),
        .rd_addr_b(rf_rd_addr_b),
        .rd_data_b(rf_rd_data_b),
        .rd_addr_c(rf_rd_addr_c),
        .rd_data_c(rf_rd_data_c),
        .wr_en(rf_wr_en),
        .wr_addr(rf_wr_addr),
        .wr_data(wb_data)
    );

    // ========================================================================
    // LANES ALUs @ 1x clock
    // in1 = rd_data_a, in2 = rd_data_b
    // ========================================================================
    generate
        for (genvar lane = 0; lane < LANES; lane++) begin : g_alu
            (* dont_touch = "true" *)
            alu_DSP48E1 u_alu (
                .clk(clk),
                .rst(rst),
                .op(alu_op),
                .in1(rf_rd_data_a[lane]),
                .in2(rf_rd_data_b[lane]),
                .out(alu_out[lane])
            );
        end
    endgenerate

    // ========================================================================
    // LANES/2 Shifter_2x (double-pumped): each handles lane pair (2i, 2i+1)
    // in1 = rd_data_a (data to shift), shift_amt = rd_data_b[4:0]
    // ========================================================================
    generate
        for (genvar i = 0; i < LANES/2; i++) begin : g_shifter
            (* dont_touch = "true" *)
            shifter_2x u_shifter (
                .clk(clk),
                .clk_2x(clk_2x),
                .rst(rst),
                .op_0(shifter_op),
                .op_1(shifter_op),
                .shift_amt_0(rf_rd_data_b[2*i][4:0]),
                .shift_amt_1(rf_rd_data_b[2*i+1][4:0]),
                .in1_0(rf_rd_data_a[2*i]),
                .in1_1(rf_rd_data_a[2*i+1]),
                .out_0(shifter_out[2*i]),
                .out_1(shifter_out[2*i+1])
            );
        end
    endgenerate

    // ========================================================================
    // LANES/2 FPU_2x (double-pumped): each handles lane pair (2i, 2i+1)
    // in1 = rd_data_a, in2 = rd_data_b, in3 = rd_data_c
    // ========================================================================
    generate
        for (genvar i = 0; i < LANES/2; i++) begin : g_fpu
            (* dont_touch = "true" *)
            fpu_2x u_fpu (
                .clk(clk),
                .clk_2x(clk_2x),
                .rst(rst),
                .in1_0(rf_rd_data_a[2*i]),
                .in1_1(rf_rd_data_a[2*i+1]),
                .in2_0(rf_rd_data_b[2*i]),
                .in2_1(rf_rd_data_b[2*i+1]),
                .in3_0(rf_rd_data_c[2*i]),
                .in3_1(rf_rd_data_c[2*i+1]),
                .valid_in_0(1'b1),
                .valid_in_1(1'b1),
                .out_0(fpu_out[2*i]),
                .out_1(fpu_out[2*i+1]),
                .valid_out_0(fpu_valid_out[2*i]),
                .valid_out_1(fpu_valid_out[2*i+1])
            );
        end
    endgenerate

    // ========================================================================
    // Shared Memory (LANES lanes)
    // Read from rd_data_a address bits, output to WB mux
    // ========================================================================
    (* dont_touch = "true" *)
    shmem #(.LANES(LANES)) u_shmem (
        .clk(clk),
        .we(shmem_we),
        .waddr(shmem_waddr),
        .wdata(rf_rd_data_a),
        .re(shmem_re),
        .raddr(shmem_raddr),
        .rdata(shmem_out)
    );

endmodule
