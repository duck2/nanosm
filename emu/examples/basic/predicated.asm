; Test predicated execution (operations guarded by predicate)
;
; Only lanes where predicate is true should execute the instruction.
;
; Expected results:
;   r2: lanes 0-3 = 150, lanes 4-7 = 100
;   r4: lanes 0-3 = lane_id, lanes 4-7 = 999
;   r7: lanes 0-3 = 3.0f, lanes 4-7 = 0

    lid r1              ; r1 = lane_id (0,1,2,3,4,5,6,7)
    addi r2, r0, 100    ; r2 = 100 (initial value for all lanes)
    addi r3, r0, 4

    ; Set predicate p0 = (lane_id < 4)
    setp.lt.i32 p0, r1, r3

    ; Predicated add: only lanes where p0=true will execute
    @p0 addi r2, r2, 50     ; r2 += 50 for lanes 0-3 only

    ; Now test predicated move
    addi r4, r0, 999
    @p0 mov r4, r1          ; r4 = lane_id for lanes 0-3, stays 999 for lanes 4-7

    ; Test predicated FPU op
    lui r5, 0x3f80          ; r5 = 1.0f (all lanes)
    lui r6, 0x4000          ; r6 = 2.0f (all lanes)
    @p0 fadd r7, r5, r6     ; r7 = 3.0f for lanes 0-3 only

    halt

