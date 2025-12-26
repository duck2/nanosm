# Parse command line arguments
set options {
    {-project_dir "Project directory"}
}

set usage "Usage: vivado -mode batch -source impl.tcl \[options\]"
array set opts [::cmdline::getoptions argv $options $usage]

# Open checkpoint
open_checkpoint [file join $opts(-project_dir) "post_synth.dcp"]

# Run implementation
opt_design
place_design
phys_opt_design
route_design

# Write checkpoint
write_checkpoint -force [file join $opts(-project_dir) "post_route"]

# Generate reports
report_timing_summary -file [file join $opts(-project_dir) "post_route_timing.rpt"]
report_utilization -file [file join $opts(-project_dir) "post_route_utilization.rpt"]
report_drc -file [file join $opts(-project_dir) "post_route_drc.rpt"]

puts "Implementation completed successfully" 