module shifter_wrapper #(
    parameter NUM_SHIFTERS = 1
) (
    input  wire clk,
    input  wire rst
);

    // INTERNAL REG INPUTS (not top-level ports)
    (* keep = "true" *) reg [1:0]  op_reg = 2'b00;
    (* keep = "true" *) reg [4:0]  shift_amt_reg = 5'd1;
    (* keep = "true" *) reg [31:0] in1_reg = 32'hDEADBEEF;

    // Make the regs toggle slowly so they aren't constants
    always @(posedge clk) begin
        if (!rst) begin
            op_reg        <= op_reg + 2'b01;
            shift_amt_reg <= shift_amt_reg + 5'd1;
            in1_reg       <= {in1_reg[30:0], in1_reg[31]}; // rotate bits
        end
    end

    // Shifter outputs kept internal
    (* keep = "true" *) wire [31:0] out_wire [NUM_SHIFTERS];

    // Instantiate shifters
    genvar i;
    generate
        for (i = 0; i < NUM_SHIFTERS; i = i + 1) begin : shifter_gen
            (* dont_touch = "true" *)
            shifter shifter_inst (
                .clk(clk),
                .rst(rst),
                .op(op_reg),
                .shift_amt(shift_amt_reg),
                .in1(in1_reg),
                .out(out_wire[i])
            );
        end
    endgenerate

    // Keep outputs alive
    (* keep = "true" *) reg [31:0] sink_reg [NUM_SHIFTERS];

    generate
        for (i = 0; i < NUM_SHIFTERS; i = i + 1) begin : sink_gen
            always @(posedge clk) begin
                sink_reg[i] <= out_wire[i];
            end
        end
    endgenerate

endmodule

