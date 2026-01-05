; Test basic ALU operations
;
; Expected results (per lane):
;   r1 = lane_id (0,1,2,3,4,5,6,7)
;   r2 = 10
;   r3 = lane_id + 10 (10,11,12,13,14,15,16,17)
;   r4 = 10 - lane_id (10,9,8,7,6,5,4,3)
;   r5 = lane_id^2 (0,1,4,9,16,25,36,49)

    sread r1, LANE_ID   ; r1 = lane_id
    addi r2, r0, 10     ; r2 = 10
    add r3, r1, r2      ; r3 = lane_id + 10
    sub r4, r2, r1      ; r4 = 10 - lane_id
    muls r5, r1, r1     ; r5 = lane_id^2 (signed 16x16)
    halt
