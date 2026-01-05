; Test memory operations
;
; Stores lane_id * 4 at address 0x100 + lane_id*4, then loads it back.
;
; Expected results (per lane):
;   r1 = lane_id
;   r2 = lane_id * 4 (0,4,8,12,16,20,24,28)
;   r4 = 0x100 + lane_id*4
;   r5 = lane_id * 4 (loaded back)

    sread r1, LANE_ID   ; lane_id
    shli r2, r1, 2      ; r2 = lane_id * 4
    addi r3, r0, 0x100  ; base = 0x100
    add r4, r3, r2      ; addr = 0x100 + lane_id*4
    stg [r4], r2        ; store lane_id*4 at addr
    ldg r5, [r4]        ; load it back
    halt
