; Test FPU operations
;
; Expected results (all lanes uniform):
;   r1 = 0x3F800000 (1.0f)
;   r2 = 0x40000000 (2.0f)
;   r3 = 0x40400000 (3.0f = 1.0 + 2.0)
;   r4 = 0x40000000 (2.0f = 1.0 * 2.0)
;   r5 = 0x40A00000 (5.0f = 2.0 * 2.0 + 1.0)

    lui r1, 0x3f80      ; r1 = 1.0f
    lui r2, 0x4000      ; r2 = 2.0f
    fadd r3, r1, r2     ; r3 = 1.0 + 2.0 = 3.0
    fmul r4, r1, r2     ; r4 = 1.0 * 2.0 = 2.0
    fma r5, r2, r2, r1  ; r5 = 2.0 * 2.0 + 1.0 = 5.0
    halt




