; SAXPY: y[i] = a * x[i] + y[i]
; 
; Memory layout:
;   x vector at address 0x0000
;   y vector at address 0x0100
;   a (scalar) loaded via immediate
;
; This demonstrates:
;   - Lane-parallel execution
;   - Global memory load/store
;   - FMA operation

    ; Get lane ID (0-7)
    sread r1, LANE_ID

    ; Calculate byte offset for this lane
    shli r2, r1, 2          ; offset = lane_id * 4 (sizeof float)

    ; Load x[lane_id]
    ldg r3, [r2]          ; r3 = mem[offset] = x[lane_id]

    ; Load y[lane_id]
    addi r4, r0, 0x100      ; r4 = 0x100 (y base address)
    add r5, r4, r2          ; r5 = y_base + offset
    ldg r6, [r5]          ; r6 = y[lane_id]

    ; Load scalar a = 2.0 (float bit pattern 0x40000000)
    lui r10, 0x40000        ; r10 = 2.0f

    ; Compute y = a * x + y using FMA
    fma r7, r10, r3, r6     ; r7 = a * x[i] + y[i]

    ; Store result back to y[lane_id]
    stg [r5], r7          ; y[lane_id] = r7

    halt
