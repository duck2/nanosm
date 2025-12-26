# Open synthesized design and explore resource usage
# Usage: vivado -mode gui -source scripts/vivado/explore_netlist.tcl

# Check if we have a checkpoint, if not guide to create one
if {[file exists "build/dsp_test/post_synth.dcp"]} {
    open_checkpoint build/dsp_test/post_synth.dcp
    puts "Opened post_synth.dcp"
} else {
    puts "ERROR: No design checkpoint found. Please run synthesis first."
    puts "Expected: build/dsp_test/post_synth.dcp"
    exit 1
}

# Select the ALU instance
puts "\n=== Finding ALU instance ==="
set alu_cells [get_cells -hierarchical -filter {REF_NAME == alu_DSP48E1}]
if {[llength $alu_cells] > 0} {
    puts "Found ALU instances: $alu_cells"
} else {
    puts "ALU might be at top level, checking..."
}

# Show schematic - this will open the schematic viewer
puts "\n=== Opening Schematic Viewer ==="
puts "Opening schematic... (GUI will show the design)"
show_schematic [get_nets -hierarchical]

puts "\n=== INSTRUCTIONS ==="
puts "1. In the Netlist window, expand the hierarchy to find your module"
puts "2. Right-click on alu_DSP48E1 and select 'Schematic'"
puts "3. You'll see:"
puts "   - LUT6 cells (the 25 LUTs)"
puts "   - DSP48E1 cell (the DSP block)"
puts "   - FDRE/FDCE cells (flip-flops)"
puts "4. Click on any cell to see its properties and connections"
puts "5. Trace signals by clicking on nets (wires)"
