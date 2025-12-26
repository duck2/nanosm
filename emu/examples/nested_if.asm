; Nested if example using SSY/Branch/.S model
;
; if (lane_id < 4) {
;     r5 = 10
;     if (lane_id < 2) {
;         r6 = 100
;     } else {
;         r6 = 200
;     }
; } else {
;     r5 = 20
;     r6 = 300
; }

    lid r1                  ; r1 = lane_id

    ; Outer if
    ssy outer_join          ; push (0xFF, outer_join)
    addi r2, r0, 4
    blt r1, r2, outer_then  ; push (0x0F, outer_then), fall through with 0xF0

outer_else:
    ; Lanes 4-7
    addi r5, r0, 20         ; r5 = 20
    addi r6, r0, 300        ; r6 = 300
    nop.s                   ; pop -> go to outer_then

outer_then:
    ; Lanes 0-3
    addi r5, r0, 10         ; r5 = 10

    ; Nested if (only lanes 0-3 are active here)
    ssy inner_join          ; push (0x0F, inner_join) - only active lanes
    addi r2, r0, 2
    blt r1, r2, inner_then  ; push (0x03, inner_then), fall through with 0x0C

inner_else:
    ; Lanes 2-3
    addi r6, r0, 200        ; r6 = 200
    nop.s                   ; pop -> go to inner_then

inner_then:
    ; Lanes 0-1
    addi r6, r0, 100        ; r6 = 100
    nop.s                   ; pop -> go to inner_join

inner_join:
    ; Lanes 0-3 active again
    nop.s                   ; pop -> go to outer_join

outer_join:
    ; All lanes active (0xFF)
    ; Expected results:
    ;   Lane 0: r5=10, r6=100
    ;   Lane 1: r5=10, r6=100
    ;   Lane 2: r5=10, r6=200
    ;   Lane 3: r5=10, r6=200
    ;   Lane 4-7: r5=20, r6=300
    halt

