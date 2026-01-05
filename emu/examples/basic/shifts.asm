; Test shift operations
;
; Expected results (per lane):
;   r1 = 0x00FF00FF
;   r2 = lane_id
;   r3 = r1 << lane_id
;   r4 = 0x0000FF00 (r1 >> 8)

    lui r1, 0x00FF0     ; r1 = 0x00FF0000
    addi r1, r1, 0xFF   ; r1 = 0x00FF00FF
    sread r2, LANE_ID   ; r2 = lane_id
    shl r3, r1, r2      ; r3 = r1 << lane_id
    shri r4, r1, 8      ; r4 = r1 >> 8
    halt




