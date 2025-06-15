.section .text
.global conv2d

# Function: conv2d
# Argument:
#   a0 - address of the input image (input image pointer)
# Return:
#   result pointer (a0 will hold the address of the result)
conv2d:
    # Load addresses for global variables
    la a1, conv_filters         # Load address of filters
    la a2, conv_biases          # Load address of biases

    la t0, CL_IN_DIM            # Load input dimension address
    lw a3, 0(t0)                # Load input dimension
    lw a4, 4(t0)                # Load output dimension
    lw a5, 12(t0)               # Load number of filters
    lw a6, 16(t0)               # Load filter dimension
    lw a7, 20(t0)               # Load stride

    # Save registers that will be used
    addi sp, sp, -16            # Make space on the stack for saved registers
    sw ra, 0(sp)                # Save return address
    sw s3, 4(sp)                # Save result pointer (s3)

    # Set vector lenght, only once
    vsetvli t0, a6, e32, m1  # vector length = FILTER_DIM

    # Loop over the CL_NUM_FILTERS
    li t1, 0                     # k = 0
    loop_k:
        bge t1, a5, loop_k_end   # if k >= CL_NUM_FILTERS, exit

        # Precompute offsets values
        mul s1, t1, a6         # s1 = k * CL_FILTER_DIM (used for filter row offset)

        # Precompute bias address for the current filter
        # Calculate bias index: bias address + k
        slli t0, t1, 2            # byte offset = *4
        add t0, a2, t0           # &bias[k]
        flw ft6, 0(t0)             # Load bias value

        # Loop over the output dimensions
        li t2, 0                 # i = 0
        loop_i:
            bge t2, a4, loop_i_end   # if i >= CL_OUT_DIM, exit

            li t3, 0                 # j = 0
            loop_j:
                bge t3, a4, loop_j_end   # if j >= CL_OUT_DIM, exit

                vmv.v.x v5, x0       # Set all elements of v5 to zero

                li t4, 0                 # fi = 0
                conv_row:
                    bge t4, a6, conv_end     # if fi >= FILTER_DIM, break

                    # input_row = (i + fi) * CL_IN_DIM + j                             (since stride = 1)
                    add  t5, t2, t4           # input_row = i + fi
                    mul  t5, t5, a3           # input_row *= CL_IN_DIM
                    add  t5, t5, t3           # input_row += input_row + j
                    
                    slli t5, t5, 2            # byte offset = *4
                    add  t5, a0, t5           # &image[row_offset]

                    # Load input row with stride = 29
                    vle32.v v1, 0(t5)      # input patch row

                    # filter_row = k * (CL_FILTER_DIM * CL_FILTER_DIM) + fi * CL_FILTER_DIM
                    # filter_row = (s1 + fi) * CL_FILTER_DIM                 (take CL_FILTER_DIM common)
                    add t6, s1, t4         # + fi
                    mul t6, t6, a6         # * FD

                    slli t6, t6, 2            # byte offset = *4
                    add  t6, a1, t6           # &filter_row

                    # Load filter row (contiguous memory)
                    vle32.v v2, 0(t6)         # filter row

                    # Accumulate dot product
                    vfmacc.vv v5, v1, v2

                    addi t4, t4, 1           # fi++
                    j conv_row
                conv_end:
                    # Sum reduction
                    vfmv.s.f v3, ft6           # Initialize v3 with bias
                    
                    vfredsum.vs v3, v5, v3

                    # Store result
                    vfmv.f.s ft0, v3         # Extract scalar from v3[0]
                    fsw ft0, 0(s3)           # Store single float
                    addi s3, s3, 4          # Increment result pointer

                    addi t3, t3, 1           # j++
                    j loop_j
            loop_j_end:
                addi t2, t2, 1               # i++
                j loop_i
        loop_i_end:
            addi t1, t1, 1               # k++
            j loop_k
    loop_k_end:

    # Restore registers and return from the function
    lw ra, 0(sp)                  # Restore return address
    lw a0, 4(sp)                  # Restore result pointer
    addi sp, sp, 16               # Restore stack pointer
    ret                           # Return from the function


.data
.align 4

# Definitions from #define (from the C code)
CL_IN_DIM:         .word 28              # Input dimension (28x28)
CL_OUT_DIM:        .word 24              # Output dimension (24x24)
CL_IN_CHANNELS:    .word 01              # Input channels (1)
CL_NUM_FILTERS:    .word 8              # Number of filters (8)
CL_FILTER_DIM:     .word 05              # Filter dimension (5x5)
CL_STRIDE:         .word 01              # Stride (1)

# Convolution parameters
conv_filters:      
    .float  -0.212320,  0.145384, -0.037321,  0.123764,  0.056655
    .float  -0.202739,  0.079783,  0.070867, -0.042695, -0.259383
    .float   0.049911,  0.145552, -0.020537, -0.187723, -0.131252
    .float   0.163713,  0.224705,  0.140069, -0.009085,  0.073234
    .float  -0.080659,  0.010824, -0.086331,  0.203459,  0.244409

    .float  -1.054972, -0.891024, -1.066550, -0.828543, -0.654060
    .float  -0.786513, -0.413009, -0.401602, -0.432393, -0.849205
    .float  -0.190523,  0.045326,  0.216727, -0.030382, -0.359319
    .float   0.350181,  0.340316,  0.487858,  0.364874,  0.259409
    .float   0.482986,  0.353294,  0.225865,  0.342781,  0.341477

    .float  -0.178495, -0.472485, -0.508939,  0.146645,  0.305071
    .float  -1.057735, -0.355156,  0.490745,  0.463817,  0.170792
    .float  -0.433517,  0.564238,  0.552975,  0.197741, -0.279824
    .float   0.272791,  0.584378,  0.121418, -0.373568, -0.231887
    .float   0.299926, -0.031639, -0.407140, -0.156032, -0.061682

    .float   0.273750,  0.310541,  0.470391, -0.055891, -0.402251
    .float   0.379908,  0.387477,  0.225295, -0.402449, -0.661950
    .float   0.408006,  0.495547, -0.003231, -0.497584, -0.559818
    .float   0.426277,  0.293883, -0.450071, -0.411655, -0.463014
    .float   0.308907, -0.218843, -0.678969, -0.654918, -0.878282

    .float   0.192461, -0.125563, -0.226632,  0.006193,  0.225292
    .float   0.208990,  0.043345, -0.004417, -0.264719, -0.372079
    .float   0.035871,  0.160617,  0.313575, -0.121912, -0.594081
    .float  -0.505590,  0.142334,  0.606456, -0.054825, -0.319844
    .float  -0.684762,  0.098203,  0.452869,  0.051462,  0.096053

    .float  -0.120633,  0.164645,  0.282223,  0.475194,  0.291666
    .float  -0.414444, -0.384494,  0.024740,  0.403475,  0.424486
    .float  -0.747294, -0.332456, -0.170921,  0.316539,  0.347904
    .float  -0.782172, -0.574138, -0.243847,  0.141671,  0.409410
    .float  -0.651157, -0.432430, -0.185768,  0.121586,  0.198998

    .float   0.168427, -0.004475, -0.116341, -0.465146, -0.472895
    .float   0.317201,  0.250817,  0.226554, -0.456673, -0.602053
    .float   0.078209,  0.507722,  0.514219, -0.068863, -0.439256
    .float  -0.231946,  0.160473,  0.504466,  0.311824, -0.176354
    .float  -0.263749, -0.127226,  0.223549,  0.458108,  0.486485

    .float   0.137269,  0.230076,  0.218404,  0.414103,  0.389986
    .float   0.270464,  0.570013,  0.722101,  0.660840,  0.469928
    .float  -0.033906,  0.095202,  0.032985,  0.161803,  0.184437
    .float  -0.835555, -0.592377, -0.858343, -0.693995, -0.071462
    .float  -0.532673, -0.534382, -0.728421, -0.574272, -0.164224

conv_biases:
    .float  -0.459817,  0.177505, -0.325453, -0.026671, -0.118034, -0.147214, -0.541557, -0.005194

result:
    .space  24 * 24 * 8 * 4        # Space for output (24x24x8) in bytes
