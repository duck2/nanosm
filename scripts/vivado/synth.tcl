# Parse command line arguments
set options {
    {-part "FPGA part number"}
    {-top "Top module name"}
    {-project_dir "Project directory"}
    {-rtl_dir "RTL source directory"}
    {-constraints_dir "Constraints directory"}
}

set usage "Usage: vivado -mode batch -source synth.tcl \[options\]"
array set opts [::cmdline::getoptions argv $options $usage]

# Create project
create_project -force project $opts(-project_dir)
set_property part $opts(-part) [current_project]

# Add source files
add_files [glob -nocomplain [file join $opts(-rtl_dir) "*.v"]]
add_files [glob -nocomplain [file join $opts(-rtl_dir) "*.sv"]]

# Add constraint files
add_files -fileset constrs_1 [glob -nocomplain [file join $opts(-constraints_dir) "*.xdc"]]

# Set top module
set_property top $opts(-top) [current_fileset]

# Run synthesis
synth_design -top $opts(-top) -part $opts(-part)

# Write checkpoint
write_checkpoint -force [file join $opts(-project_dir) "post_synth"]

# Generate reports
report_timing_summary -file [file join $opts(-project_dir) "post_synth_timing.rpt"]
report_utilization -file [file join $opts(-project_dir) "post_synth_utilization.rpt"]

puts "Synthesis completed successfully" 