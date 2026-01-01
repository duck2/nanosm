; Clear screen to solid color
; Kernel args (in registers):
;   R20 = render target descriptor pointer
;   R21 = num_tiles_x
;   R22 = num_tiles_y
; Tile buffer layout (lane-interleaved):
;   scratchpad[0..1023] = color (32x32 tile, 1 pixel per address)

    ; Clear color: cornflower blue (0xFF6495ED) - ABGR packed
    lui   r1, 0xFF64        ; upper 16 bits
    ori   r1, r1, 0x95ED    ; lower 16 bits

    ; Lane ID for conflict-free access
    lid   r5

    ; Constants
    addi  r4, r0, 128       ; loop iterations (1024/8 = 128 per lane)

    ; Tile loop counters
    addi  r10, r0, 0        ; tile_y = 0

tile_y_loop:
    addi  r11, r0, 0        ; tile_x = 0

tile_x_loop:
    ; Fill tile buffer with clear color (128 iterations, 8 pixels each)
    mov   r3, r5            ; scratch_addr = lane_id (each lane starts at different bank)

fill_loop:
    sts   [r3+0], r1        ; store clear color
    addi  r3, r3, 8         ; next addr += 8 (each lane covers every 8th address)
    addi  r6, r0, 1024
    setp.lt.u32 p0, r3, r6  ; p0 = scratch_addr < 1024
    @p0 bra.uni fill_loop

    ; Commit tile to framebuffer
    tile_commit r11, r10, r20

    ; Next tile_x
    addi  r11, r11, 1
    setp.lt.u32 p0, r11, r21 ; p0 = tile_x < num_tiles_x
    @p0 bra.uni tile_x_loop

    ; Next tile_y
    addi  r10, r10, 1
    setp.lt.u32 p0, r10, r22 ; p0 = tile_y < num_tiles_y
    @p0 bra.uni tile_y_loop

    tile_wait
    halt
