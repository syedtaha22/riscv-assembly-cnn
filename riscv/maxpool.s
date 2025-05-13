#define STDOUT 0xd0580000

.section .text
.global maxpool

# Function: maxpool
# Argument:
#   a0 - input base address
# Returns:
#   a0 - output base address
maxpool:
    addi sp, sp, -4
    sw ra, 0(sp)

    la a1, output               # Load output base address

    la t0, MP_IN_DIM
    la t1, MP_KERNEL_DIM
    la t2, MP_STRIDE
    la t3, MP_OUT_DIM
    la t4, MP_OUT_CHANNELS

    lw t0, 0(t0)                # MP_IN_DIM
    lw t1, 0(t1)                # MP_KERNEL_DIM
    lw t2, 0(t2)                # MP_STRIDE
    lw t3, 0(t3)                # MP_OUT_DIM
    lw t4, 0(t4)                # MP_OUT_CHANNELS

    mul s1, t0, t0              # s1 = MP_IN_DIM * MP_IN_DIM
    mul s2, t3, t3              # s2 = MP_OUT_DIM * MP_OUT_DIM

    li s0, 0                    # c = 0
    loop_c:
        bge s0, t4, loop_c_end

        mul s4, s0, s1              # input channel offset = c * IN_DIM²
        mul s6, s0, s2              # output channel offset = c * OUT_DIM²

        li s5, 0                    # i = 0
        loop_i:
            bge s5, t3, loop_i_end

            li s9, 0                    # j = 0
            loop_j:
                bge s9, t3, loop_j_end

                vsetvli s10, t1, e32, m1    # Set VL to kernel size

                # Calculate input address
                mul s7, s5, t0              # s7 = i * MP_IN_DIM
                add s7, s7, s9              # s7 += j
                mul s7, s7, t2              # s7 *= MP_STRIDE
                add s7, s7, s4              # add input channel offset
                slli s7, s7, 2              # byte offset
                add s7, a0, s7              # final input address

                slli s3, t0, 2              # MP_IN_DIM * 4 (byte stride)
                vlsseg2e32.v v0, (s7), s3   # Load 2 rows of kernel into v0

                fmv.w.x ft0, zero
                vfmv.s.f v1, ft0
                vfredmax.vs v1, v0, v1      # Reduce max

                # Compute output address
                mul s8, s5, t3              # i * MP_OUT_DIM
                add s8, s8, s9              # + j
                add s8, s8, s6              # + channel offset
                slli s8, s8, 2              # byte offset
                add s8, a1, s8              # final output address

                vse32.v v1, (s8)            # Store result

                addi s9, s9, 1
                j loop_j

            loop_j_end:
                addi s5, s5, 1
                j loop_i

        loop_i_end:
            addi s0, s0, 1
            j loop_c

    loop_c_end:
        mv a0, a1                   # Return output base address

    lw ra, 0(sp)
    addi sp, sp, 4
    ret

.section .data
.align 4

MP_IN_DIM:          .word 24     # Dimension of the input feature map
MP_OUT_DIM:         .word 12     # Dimension of the output feature map
MP_IN_CHANNELS:     .word  8     # Number of input channels
MP_OUT_CHANNELS:    .word  8     # Number of output channels
MP_KERNEL_DIM:      .word  2     # Dimension of the pooling kernel
MP_STRIDE:          .word  2     # Stride of the pooling operation

output:
    .space 8 * 12 * 12 * 4        # Output data space: channels × height × width × 4 bytes
