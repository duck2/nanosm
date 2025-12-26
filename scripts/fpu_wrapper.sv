module fpu_wrapper #(
    parameter NUM_FPUS = 1
) (
    input  wire clk,
    input  wire rst
);

    // INTERNAL REG INPUTS (not top-level ports)
    (* keep = "true" *) reg [31:0] in1_reg = 32'h3F800000; // 1.0
    (* keep = "true" *) reg [31:0] in2_reg = 32'h40000000; // 2.0
    (* keep = "true" *) reg [31:0] in3_reg = 32'h40400000; // 3.0
    (* keep = "true" *) reg        valid_reg = 1'b1;

    // Make the regs toggle slowly so they aren't constants
    always @(posedge clk) begin
        if (!rst) begin
            in1_reg   <= {in1_reg[30:0], in1_reg[31]}; // rotate bits
            in2_reg   <= in2_reg + 32'h01010101;
            in3_reg   <= {in3_reg[0], in3_reg[31:1]};  // another rotate
            valid_reg <= ~valid_reg;
        end
    end

    // FPU outputs kept internal
    (* keep = "true" *) wire [31:0] out_wire [NUM_FPUS];
    (* keep = "true" *) wire        valid_out_wire [NUM_FPUS];

    // Instantiate FPUs
    genvar i;
    generate
        for (i = 0; i < NUM_FPUS; i = i + 1) begin : fpu_gen
            (* dont_touch = "true" *)
            fpu_DSP48E1 fpu_inst (
                .clk(clk),
                .rst(rst),
                .in1(in1_reg),
                .in2(in2_reg),
                .in3(in3_reg),
                .valid_in(valid_reg),
                .out(out_wire[i]),
                .valid_out(valid_out_wire[i])
            );
        end
    endgenerate

    // Keep outputs alive
    (* keep = "true" *) reg [31:0] sink_reg [NUM_FPUS];
    (* keep = "true" *) reg        sink_valid_reg [NUM_FPUS];

    generate
        for (i = 0; i < NUM_FPUS; i = i + 1) begin : sink_gen
            always @(posedge clk) begin
                sink_reg[i]       <= out_wire[i];
                sink_valid_reg[i] <= valid_out_wire[i];
            end
        end
    endgenerate

endmodule
