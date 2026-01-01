; Divergent control flow example using SSY/setp/@p bra/.S model
;
; Control flow model:
;   SSY label           - push (active_mask, label_pc) for reconvergence
;   setp.cmp.type pN    - set predicate based on comparison
;   @pN bra label       - push (taken_mask, label_pc), fall through with not_taken_mask
;   *.S                 - pop (mask, pc) and switch to it
;
; if (lane_id < 4) r5 = 100 else r5 = 200

    lid r1                  ; r1 = lane_id (0,1,2,3,4,5,6,7)
    addi r2, r0, 4          ; r2 = threshold = 4
    addi r3, r0, 100        ; value for taken path (lane_id < 4)
    addi r4, r0, 200        ; value for not-taken path (lane_id >= 4)

    ssy reconv              ; push (0xFF, reconv) - all lanes rejoin here
    setp.lt.i32 p0, r1, r2  ; p0 = lane_id < 4
    @p0 bra taken           ; push (0x0F, taken), mask=0xF0, fall through

not_taken:
    ; Lanes 4-7 execute this path
    mov r5, r4              ; r5 = 200
    nop.s                   ; pop (0x0F, taken) -> switch to taken path

taken:
    ; Lanes 0-3 execute this path
    mov r5, r3              ; r5 = 100
    nop.s                   ; pop (0xFF, reconv) -> switch to reconv

reconv:
    ; All lanes active again (mask = 0xFF)
    ; r5 now contains 100 for lanes 0-3, 200 for lanes 4-7

    ; Double the result to show all lanes work
    add r6, r5, r5          ; r6 = r5 * 2

    halt
