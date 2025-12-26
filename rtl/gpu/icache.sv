`timescale 1ns / 1ps

module icache (
    input  logic        clk,
    input  logic        rst,
    
    // Instruction fetch interface
    input  logic [31:0] pc,           // Program counter (byte-addressed)
    output logic [31:0] instruction,  // Fetched instruction
    output logic        valid         // Instruction valid
);
    // 1024 x 32-bit instructions = 4KB
    localparam DEPTH = 1024;
    localparam ADDR_WIDTH = $clog2(DEPTH);
    
    // BRAM storage
    logic [31:0] mem [0:DEPTH-1];
    
    // Word address (convert from byte address)
    logic [ADDR_WIDTH-1:0] word_addr;
    assign word_addr = pc[ADDR_WIDTH+1:2];
    
    // Address range check
    logic addr_in_range;
    assign addr_in_range = (pc[31:ADDR_WIDTH+2] == '0);
    
    // Synchronous read with registered output
    always_ff @(posedge clk) begin
        if (rst) begin
            instruction <= 32'h0000_0000;  // NOP or invalid instruction
            valid <= 1'b0;
        end else begin
            instruction <= mem[word_addr];
            valid <= addr_in_range;
        end
    end
    
    // Initialize from hex file
    initial begin
        // Initialize to NOPs (or zeros)
        for (int i = 0; i < DEPTH; i = i + 1) begin
            mem[i] = 32'h0000_0000;
        end
        
        // Load kernel if file exists
        // Use $readmemh to load from kernel.hex
        // Format: one 32-bit hex value per line (without 0x prefix)
        if ($test$plusargs("KERNEL_HEX")) begin
            $readmemh("kernel.hex", mem);
            $display("Loaded kernel from kernel.hex");
        end
    end

endmodule
