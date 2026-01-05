; Shaded triangle rasterization kernel
; Barycentric interpolation of vertex colors (RGB at corners)
; Uses SSY/divergence to skip shading when all lanes are outside
; Kernel args:
;   R20 = render target descriptor pointer
;   R21 = num_tiles_x
;   R22 = num_tiles_y
; Vertices (CCW):
;   V0 = (160, 40)  - red
;   V1 = (240, 200) - green
;   V2 = (80, 200)  - blue

    ; Load vertices
    addi  r23, r0, 160      ; v0.x
    addi  r24, r0, 40       ; v0.y
    addi  r25, r0, 240      ; v1.x
    addi  r26, r0, 200      ; v1.y
    addi  r27, r0, 80       ; v2.x
    addi  r28, r0, 200      ; v2.y

    ; Compute edge deltas
    sub   r2, r25, r23      ; dx01 = v1.x - v0.x
    sub   r3, r26, r24      ; dy01 = v1.y - v0.y
    sub   r4, r27, r25      ; dx12 = v2.x - v1.x
    sub   r29, r28, r26     ; dy12 = v2.y - v1.y
    sub   r30, r23, r27     ; dx20 = v0.x - v2.x
    sub   r31, r24, r28     ; dy20 = v0.y - v2.y

    ; Vertex colors as floats (0-255 range)
    lui   r1, 0x437F0       ; 255.0 = 0x437F0000

    ; Lane ID
    sread r5, LANE_ID

    ; Tile loop
    addi  r10, r0, 0        ; tile_y = 0

tile_y_loop:
    addi  r11, r0, 0        ; tile_x = 0

tile_x_loop:
    shli  r6, r11, 5        ; tile_x * 32
    shli  r7, r10, 5        ; tile_y * 32

    addi  r12, r0, 0        ; row = 0

row_loop:
    add   r8, r7, r12       ; py

    addi  r13, r0, 0        ; group = 0

group_loop:
    ; px = tile_origin_x + group*8 + lane_id
    shli  r9, r13, 3
    add   r9, r9, r6
    add   r9, r9, r5

    ; Edge 0->1: E01
    sub   r14, r8, r24
    sub   r15, r9, r23
    muls  r14, r2, r14
    muls  r15, r3, r15
    sub   r14, r14, r15     ; r14 = E01 (weight for v2)

    ; Edge 1->2: E12
    sub   r15, r8, r26
    sub   r16, r9, r25
    muls  r15, r4, r15
    muls  r16, r29, r16
    sub   r15, r15, r16     ; r15 = E12 (weight for v0)

    ; Edge 2->0: E20
    sub   r16, r8, r28
    sub   r17, r9, r27
    muls  r16, r30, r16
    muls  r17, r31, r17
    sub   r16, r16, r17     ; r16 = E20 (weight for v1)

    ; Inside test
    setp.ge.i32 p0, r14, r0
    setp.ge.i32 p1, r15, r0
    setp.ge.i32 p2, r16, r0
    pand  p0, p0, p1
    pand  p0, p0, p2        ; p0 = inside

    ; Default to black
    addi  r19, r0, 0

    ; Divergent shading: inside lanes shade, outside lanes skip
    ssy   shade_done
    @p0 bra do_shade
    ; Not-taken path (outside): r19 already black, just sync
    sync

do_shade:
    ; Inside lanes only: compute shaded color
    itof  r17, r14          ; E01 as float
    itof  r18, r15          ; E12 as float
    itof  r19, r16          ; E20 as float

    fadd  r14, r17, r18
    fadd  r14, r14, r19     ; total

    frcp  r14, r14          ; 1/total

    fmul  r15, r18, r14     ; w0 (red weight)
    fmul  r16, r19, r14     ; w1 (green weight)
    fmul  r17, r17, r14     ; w2 (blue weight)

    fmul  r15, r15, r1      ; red
    fmul  r16, r16, r1      ; green
    fmul  r17, r17, r1      ; blue

    ftofx.0.clamp r15, p3, r15
    ftofx.0.clamp r16, p3, r16
    ftofx.0.clamp r17, p3, r17

    andi  r15, r15, 0xFF
    andi  r16, r16, 0xFF
    andi  r17, r17, 0xFF
    shli  r16, r16, 8
    shli  r17, r17, 16
    or    r19, r15, r16
    or    r19, r19, r17
    lui   r18, 0xFF000
    or  r19, r19, r18
    sync

shade_done:
    ; Store to scratchpad: row * 32 + px (lane-interleaved, conflict-free)
    shli  r17, r12, 5       ; row * 32
    add   r17, r17, r9      ; + px (already = group*8 + lane_id)
    sub   r17, r17, r6      ; - tile_origin_x (get px within tile)
    sts   [r17], r19

    ; Next group
    addi  r13, r13, 1
    addi  r17, r0, 4
    setp.lt.u32 p0, r13, r17
    @p0 bra.uni group_loop

    ; Next row
    addi  r12, r12, 1
    addi  r17, r0, 32
    setp.lt.u32 p0, r12, r17
    @p0 bra.uni row_loop

    ; Commit tile
    tile_commit r11, r10, r20

    ; Next tile_x
    addi  r11, r11, 1
    setp.lt.u32 p0, r11, r21
    @p0 bra.uni tile_x_loop

    ; Next tile_y
    addi  r10, r10, 1
    setp.lt.u32 p0, r10, r22
    @p0 bra.uni tile_y_loop

    tile_wait
    halt
