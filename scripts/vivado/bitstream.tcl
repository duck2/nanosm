# Parse command line arguments
set options {
    {-project_dir "Project directory"}
}

set usage "Usage: vivado -mode batch -source bitstream.tcl \[options\]"
array set opts [::cmdline::getoptions argv $options $usage]

# Open checkpoint
open_checkpoint [file join $opts(-project_dir) "post_route.dcp"]

# Generate bitstream
write_bitstream -force [file join $opts(-project_dir) "project.bit"]

puts "Bitstream generation completed successfully" 