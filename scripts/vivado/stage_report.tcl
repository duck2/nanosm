proc stage_report {stages} {
    puts "Stage-to-stage timing summary:"
    puts "------------------------------------------------------"
    puts "from  -> to      slack   levels   datapath(ns)"
    puts "------------------------------------------------------"

    for {set i 0} {$i < [expr {[llength $stages] - 1}]} {incr i} {
        set from [lindex $stages $i]
        set to   [lindex $stages [expr {$i+1}]]

        set p [get_timing_paths \
                  -from [get_cells -hier -regexp ".*${from}_.*"] \
                  -to   [get_cells -hier -regexp ".*${to}_.*"] \
                  -max_paths 1]

        if {[llength $p] == 0} {
            puts [format "%-5s -> %-5s   (no path)" $from $to]
            continue
        }

        set p [lindex $p 0]

        set slack   [get_property SLACK         $p]
        set levels  [get_property LOGIC_LEVELS  $p]
        set ddelay  [get_property DATAPATH_DELAY $p]

        puts [format "%-5s -> %-5s  %7.3f   %3d      %7.3f" \
                      $from $to $slack $levels $ddelay]
    }
    puts "------------------------------------------------------"
}

# usage
# stage_report {s1 s1b s2 s3 s4a s4b s4c s5a s5b s6 s6b s7 s8 s8b s9 s9b s10}
