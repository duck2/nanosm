; Test scratchpad memory (lane-interleaved shared memory)
;
; Expected results (per lane):
;   r1 = lane_id
;   r2 = lane_id * 2
;   r3 = lane_id * 2 (loaded back from scratchpad)
;
; Lane-interleaved: each lane writes to different addr to avoid bank conflicts

    sread r1, LANE_ID   ; lane_id
    shli r2, r1, 1      ; r2 = lane_id * 2
    sts [r1], r2      ; scratchpad[lane_id] = lane_id * 2 (conflict-free)
    lds r3, [r1]      ; r3 = scratchpad[lane_id]
    halt
