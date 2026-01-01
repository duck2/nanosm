; Triangle rasterization kernel
; Rasterizes a single white triangle on black background using edge functions
; Kernel args:
;   R20 = render target descriptor pointer
;   R21 = num_tiles_x
;   R22 = num_tiles_y
; Triangle vertices passed via registers (screen space 320x240, CCW)

    ; Load triangle vertices
    addi  r23, r0, 160      ; v0.x = 160
    addi  r24, r0, 40       ; v0.y = 40
    addi  r25, r0, 240       ; v1.x = 240
    addi  r26, r0, 200      ; v1.y = 200
    addi  r27, r0, 80      ; v2.x = 80
    addi  r28, r0, 200      ; v2.y = 200

    ; Compute edge deltas
    sub   r2, r25, r23      ; dx01 = v1.x - v0.x
    sub   r3, r26, r24      ; dy01 = v1.y - v0.y
    sub   r4, r27, r25      ; dx12 = v2.x - v1.x
    sub   r29, r28, r26     ; dy12 = v2.y - v1.y
    sub   r30, r23, r27     ; dx20 = v0.x - v2.x
    sub   r31, r24, r28     ; dy20 = v0.y - v2.y

    ; Colors
    lui   r1, 0xFFFF
    ori   r1, r1, 0xFFFF    ; white = 0xFFFFFFFF

    ; Lane ID
    lid   r5

    ; Tile loop
    addi  r10, r0, 0        ; tile_y = 0

tile_y_loop:
    addi  r11, r0, 0        ; tile_x = 0

tile_x_loop:
    ; Tile origin
    shli  r6, r11, 5        ; tile_x * 32
    shli  r7, r10, 5        ; tile_y * 32

    addi  r12, r0, 0        ; row = 0

row_loop:
    add   r8, r7, r12       ; py = tile_origin_y + row

    addi  r13, r0, 0        ; group = 0

group_loop:
    ; px = tile_origin_x + group*8 + lane_id
    shli  r9, r13, 3
    add   r9, r9, r6
    add   r9, r9, r5        ; r9 = px

    ; Edge 0->1: E01 = dx01*(py - v0.y) - dy01*(px - v0.x)
    sub   r14, r8, r24      ; py - v0.y
    sub   r15, r9, r23      ; px - v0.x
    muls  r14, r2, r14      ; dx01 * (py - v0.y)
    muls  r15, r3, r15      ; dy01 * (px - v0.x)
    sub   r14, r14, r15     ; E01

    ; Edge 1->2: E12 = dx12*(py - v1.y) - dy12*(px - v1.x)
    sub   r15, r8, r26      ; py - v1.y
    sub   r16, r9, r25      ; px - v1.x
    muls  r15, r4, r15      ; dx12 * (py - v1.y)
    muls  r16, r29, r16     ; dy12 * (px - v1.x)
    sub   r15, r15, r16     ; E12

    ; Edge 2->0: E20 = dx20*(py - v2.y) - dy20*(px - v2.x)
    sub   r16, r8, r28      ; py - v2.y
    sub   r17, r9, r27      ; px - v2.x
    muls  r16, r30, r16     ; dx20 * (py - v2.y)
    muls  r17, r31, r17     ; dy20 * (px - v2.x)
    sub   r16, r16, r17     ; E20

    ; Inside test: all edges >= 0
    setp.ge.i32 p0, r14, r0
    setp.ge.i32 p1, r15, r0
    setp.ge.i32 p2, r16, r0
    pand  p0, p0, p1
    pand  p0, p0, p2

    ; Conditional move: black default, white if inside
    addi  r19, r0, 0
    @p0 add r19, r1, r0

    ; Store to scratchpad: row * 4 + group
    shli  r17, r12, 2
    add   r17, r17, r13
    sts   [r17+0], r19

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
