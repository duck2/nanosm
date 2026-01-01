; Clear screen to solid color
; Kernel args (in registers):
;   R20 = render target descriptor pointer
;   R21 = num_tiles_x
;   R22 = num_tiles_y
; Tile buffer layout:
;   scratchpad[0..127] = color (32x32 tile, 8 pixels per address)

    ; Clear color: cornflower blue (0xFF6495ED) - ABGR packed
    lui   r1, 0xFF64        ; upper 16 bits
    ori   r1, r1, 0x95ED    ; lower 16 bits

    ; Constants
    addi  r4, r0, 128       ; loop end (32*32/8 = 128)

    ; Tile loop counters
    addi  r10, r0, 0        ; tile_y = 0

tile_y_loop:
    addi  r11, r0, 0        ; tile_x = 0

tile_x_loop:
    ; Fill tile buffer with clear color (128 addresses for 32x32 tile)
    addi  r3, r0, 0         ; scratch_addr = 0

fill_loop:
    sts   [r3+0], r1        ; store clear color to scratchpad[r3]
    addi  r3, r3, 1
    setp.lt.u32 p0, r3, r4  ; p0 = scratch_addr < 128
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
