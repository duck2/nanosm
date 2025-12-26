; Simple loop example (uniform loop - all lanes exit together)
;
; Computes sum = 1 + 2 + ... + 10 for all lanes.
; This is a uniform loop where all lanes iterate the same number of times.

    addi r2, r0, 10         ; r2 = counter = 10
    addi r3, r0, 0          ; r3 = sum = 0
    addi r4, r0, 1          ; r4 = constant 1

loop:
    ; Add current counter to sum
    add r3, r3, r2          ; sum += counter

    ; Decrement counter
    sub r2, r2, r4          ; counter -= 1

    ; Loop while counter > 0 (uniform branch - all lanes same condition)
    bne r2, r0, loop        ; if counter != 0, all lanes loop

done:
    ; r3 = 10 + 9 + 8 + 7 + 6 + 5 + 4 + 3 + 2 + 1 = 55 for all lanes
    halt
