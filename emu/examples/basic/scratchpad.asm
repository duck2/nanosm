; Test scratchpad memory (per-lane local storage)
;
; Expected results (per lane):
;   r1 = lane_id
;   r2 = lane_id * 2
;   r3 = lane_id * 2 (loaded back from scratchpad)

    lid r1              ; lane_id
    shli r2, r1, 1      ; r2 = lane_id * 2
    sts [r0+0], r2      ; scratchpad[0] = lane_id * 2
    lds r3, [r0+0]      ; r3 = scratchpad[0]
    halt
