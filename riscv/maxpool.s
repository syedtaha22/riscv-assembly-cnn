
.section .text
.global maxpool

# Function: maxpool
# Argument:
#   a0 - input base address
# Returns:
#   a0 - output base address
maxpool:
    addi sp, sp, -4
    sw ra, 0(sp)            # Save return address

    la a1, output           # Load output base address

    la t5, MP_IN_DIM       # Load input dimensions address
    lw t0, 0(t5)            # Load input dimensions
    lw t1, 4(t5)            # Load Kernel dimensions
    lw t2, 8(t5)            # Load stride
    lw t3, 12(t5)           # Load output dimensions
    lw t4, 16(t5)           # Load output channels

    mul s1, t0, t0          # s1 = MP_IN_DIM * MP_IN_DIM
    mul s2, t3, t3          # s2 = MP_OUT_DIM * MP_OUT_DIM
 
    li s0, 0                        # c = 0
    loop_c:
        bge s0, t4, loop_c_end          # if c >= MP_OUT_CHANNELS, exit

        mul s4, s0, s1                  # input channel offset = c * IN_DIM²
        mul s6, s0, s2                  # output channel offset = c * OUT_DIM²

        li s5, 0                        # i = 0
        loop_i:
            bge s5, t3, loop_i_end

            # Precompute input_channel_offset + i * MP_STRIDE * MP_IN_DIM
            mul s10, s5, t2                 # s10 = i * MP_STRIDE
            mul s10, s10, t0                # s10 *= MP_IN_DIM
            add s10, s10, s4                # s10 = (i * MP_STRIDE * IN_DIM) + input_channel_offset

            li s9, 0                        # j = 0
            loop_j:
                bge s9, t3, loop_j_end

                li t5, 0xff800000        # Bit pattern for -inf in IEEE 754
                fmv.w.x ft0, t5
                vfmv.v.f v6, ft0         # Broadcast -inf to all elements of v

                # Precompute s10 + j * MP_STRIDE
                mul s7, s9, t2                # s7 = j * MP_STRIDE
                add s7, s7, s10               # s7 = (i * MP_STRIDE * IN_DIM) + j * MP_STRIDE + input_channel_offset

                li t6, 0                       # patch row = 0
                patch_loop_ki:
                    bge t6, t1, patch_done         # for each row in kernel

                    # Calculate Base Address of Input Patch: s7 + ki * MP_IN_DIM
                    mul t5, t6, t0                # t5 = ki * MP_IN_DIM
                    add t5, s7, t5                

                    slli t5, t5, 1                # Convert to byte offset
                    add t5, a0, t5                # t5 = base address of input patch row

                    vsetvli s3, t1, e16, m1       # VL = kernel_dim
                    vle16.v v5, (t5)              # Load current row of patch
                    vfredmax.vs v6, v5, v6        # Reduce into v6

                    addi t6, t6, 1
                    j patch_loop_ki

                patch_done:
                    mul s8, s5, t3                 # i * MP_OUT_DIM
                    add s8, s8, s9                 # + j
                    add s8, s8, s6                 # + output channel offset
                    slli s8, s8, 1                 # Convert to byte offset
                    add s8, a1, s8                 # final output address
                    
                    vse16.v v6, (s8)               # Store max-pooled value

                    addi s9, s9, 1                 # j++
                    j loop_j

            loop_j_end:
                addi s5, s5, 1                 # i++
                j loop_i

        loop_i_end:
            addi s0, s0, 1                 # c++
            j loop_c

    loop_c_end:
        mv a0, a1                      # Return output pointer

    lw ra, 0(sp)
    addi sp, sp, 4
    ret


.section .data
.align 4

MP_IN_DIM:          .word 24     # Dimension of the input feature map
MP_KERNEL_DIM:      .word  2     # Dimension of the pooling kernel
MP_STRIDE:          .word  2     # Stride of the pooling operation
MP_OUT_DIM:         .word 12     # Dimension of the output feature map
MP_CHANNELS:        .word  8     # Number of channels

output:
    .space 8 * 12 * 12 * 4        # Output data space: channels × height × width × 4 bytes
