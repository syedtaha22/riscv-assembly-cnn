.section .text
.global flatten

# Function: flatten
# Arguments:
#   a0 - input base address
# Returns:
#   a0 - output base address (flatten_output)
flatten:
    # Save return address
    addi sp, sp, -4
    sw ra, 0(sp)

    la a1, flatten_output          # a1 = output base address

    la t0, FL_IN_DIM               # t0 = FL_IN_DIM
    la t1, FL_IN_CHANNELS          # t1 = FL_IN_CHANNELS

    lw t0, 0(t0)                   # t0 = FL_IN_DIM (e.g. 12)
    lw t1, 0(t1)                   # t1 = FL_IN_CHANNELS (e.g. 8)
    mul t2, t0, t0                 # t2 = FL_IN_DIM * FL_IN_DIM = 144
    slli s2, t2, 1                 # stride in bytes between channels = 144 * 2

    vsetvli s3, t1, e16            # VL = channels, e16

    li t3, 0                       # flat_index = 0

    li t4, 0                       # i = 0
    loop_i:
        bge t4, t0, loop_i_end

        # Precompute value
        mul t2, t4, t0                 # t2 = i * FL_IN_DIM

        li t5, 0                       # j = 0
        loop_j:
            bge t5, t0, loop_j_end

            # input_patch_index = i * FL_IN_DIM + j
            add t6, t2, t5                 # t6 = i * W + j
            slli t6, t6, 1                 # offset in bytes
            add s5, a0, t6                 # input + offset

            vlse16.v v5, (s5), s2          # load strided values into v5

            slli s6, t3, 1                 # output offset = flat_index * 2
            add s7, a1, s6                 # output address
            vse16.v v5, (s7)               # store result

            add t3, t3, t1                 # flat_index += channels
            addi t5, t5, 1
            j loop_j

        loop_j_end:
            addi t4, t4, 1
            j loop_i

    loop_i_end:
        mv a0, a1                      # Return address of flatten_output

        lw ra, 0(sp)
        addi sp, sp, 4
        ret

.data
.align 4

FL_IN_DIM:              .word 12
FL_IN_CHANNELS:         .word 8

# Output of flatten layer (12x12x8 = 1152 elements) * 2 bytes = 2304 bytes
flatten_output:         .space 2304                # Reserve space for flattened output
