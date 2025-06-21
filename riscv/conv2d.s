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

    addi sp, sp, -16            # Make space on the stack for saved registers
    sw ra, 0(sp)                # Save return address
    sw s3, 4(sp)                # Save result pointer (s3)

    # Set vector lenght, only once
    vsetvli t0, a6, e16, m1  # vector length = FILTER_DIM

    # Loop over the CL_NUM_FILTERS
    li t1, 0                     # k = 0
    loop_k:
        bge t1, a5, loop_k_end   # if k >= CL_NUM_FILTERS, exit

        # Precompute offsets values
        mul s1, t1, a6         # s1 = k * CL_FILTER_DIM (used for filter row offset)

        # Precompute bias address for the current filter
        # Calculate bias index: bias address + k
        slli t0, t1, 1            # byte offset = *2
        add t0, a2, t0           # &bias[k]
        lh s10, 0(t0)             # Load bias value

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
                    
                    slli t5, t5, 1            # byte offset = *2
                    add  t5, a0, t5           # &image[row_offset]

                    # filter_row = k * (CL_FILTER_DIM * CL_FILTER_DIM) + fi * CL_FILTER_DIM
                    # filter_row = (s1 + fi) * CL_FILTER_DIM                 (take CL_FILTER_DIM common)
                    add t6, s1, t4            # + fi
                    mul t6, t6, a6            # * FD
                    slli t6, t6, 1            # byte offset = *2
                    add  t6, a1, t6           # &filter_row

                    # Load input row with stride = 29
                    vle16.v v1, 0(t5)         # input patch row

                    # Load filter row (contiguous memory)
                    vle16.v v2, 0(t6)         # filter row

                    # Accumulate dot product
                    vfmacc.vv v5, v1, v2

                    addi t4, t4, 1           # fi++
                    j conv_row
                conv_end:
                    # Sum reduction
                    vmv.s.x v3, s10           # Initialize v3 with bias
                    
                    vfredsum.vs v3, v5, v3

                    # Store result
                    vmv.x.s s11, v3         # Extract scalar from v3[0]
                    sh s11, 0(s3)           # Store single float
                    addi s3, s3, 2           # Increment result pointer

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
    .half 0x3687, 0x3231, 0x2eef, 0x2e60, 0xb2d9
    .half 0x1ff3, 0x206c, 0xb437, 0x2e28, 0x3032
    .half 0xb443, 0xb4fa, 0xa2e0, 0x312f, 0x33ff
    .half 0xac70, 0x308c, 0x2c6f, 0x2e06, 0xac17
    .half 0x356f, 0x30a9, 0xade5, 0xb37a, 0x2c46

    .half 0x2931, 0x3335, 0x3343, 0x23bd, 0xb717
    .half 0x2871, 0x3465, 0x3904, 0x3735, 0xaa99
    .half 0xae44, 0x2d4b, 0x35be, 0x3a16, 0x380f
    .half 0xb855, 0xb8d6, 0xb864, 0x30ab, 0x36e5
    .half 0xb3e3, 0xb8dc, 0xbb2a, 0xb933, 0x31c2

    .half 0xb747, 0xb4a6, 0xb13e, 0xb295, 0x2529
    .half 0xba12, 0xb7ec, 0xb708, 0xb367, 0x2dd9
    .half 0xb88a, 0xaf74, 0x31ba, 0x3621, 0x362d
    .half 0x32ce, 0x3887, 0x3603, 0x3607, 0x315d
    .half 0x34f2, 0x2c86, 0x2e0c, 0xb0c3, 0xb1e7

    .half 0x343c, 0xae89, 0xb6f5, 0xb9b8, 0xba6d
    .half 0x3559, 0x362b, 0x34a2, 0xb529, 0xb89f
    .half 0x1137, 0x3620, 0x39b4, 0x33eb, 0xb42e
    .half 0xb0d1, 0xa074, 0x3823, 0x35a3, 0x3274
    .half 0xaed0, 0xae1b, 0x3290, 0x3353, 0x3440

    .half 0x348b, 0x33bb, 0x34cc, 0x336e, 0x38c0
    .half 0x30e0, 0x3633, 0x3743, 0x3807, 0x35ce
    .half 0xb43d, 0x3663, 0x3814, 0x32eb, 0xb742
    .half 0xbad0, 0xb819, 0xb4ad, 0xba68, 0xbc53
    .half 0xbc95, 0xbd89, 0xbe80, 0xba93, 0xafd8

    .half 0xb91a, 0x34a8, 0x36d8, 0xb089, 0xb0b7
    .half 0xbc37, 0x3218, 0x36b4, 0x2e84, 0xb043
    .half 0xbacb, 0x3138, 0x37d3, 0x29de, 0xab5c
    .half 0xb6cd, 0x363b, 0x3252, 0xb727, 0x30b3
    .half 0xb10e, 0x3433, 0xa89c, 0xb4ca, 0x337c

    .half 0xb68e, 0xb7ad, 0xb454, 0x3563, 0x35d3
    .half 0xb594, 0xb2d3, 0x321d, 0x383f, 0xa821
    .half 0xade3, 0x35c3, 0x3879, 0x35b7, 0xb83d
    .half 0x3482, 0x3119, 0x2fda, 0xb1d3, 0xba2f
    .half 0x35e9, 0x285d, 0xb087, 0xb5c7, 0xb705

    .half 0x30e5, 0x2c86, 0x2f9f, 0x311a, 0xacf5
    .half 0x3207, 0x34a2, 0x3428, 0xa4bd, 0xb70f
    .half 0x305b, 0x2d6c, 0x3621, 0xb21d, 0xb935
    .half 0x2f19, 0x3376, 0x35ae, 0xb1d1, 0xbb24
    .half 0x25c1, 0x35af, 0x35b5, 0xb901, 0xbd26



conv_biases:
    .half 0xb6ab, 0xb225, 0xb0ba, 0xb71a, 0x3379, 0xa19d, 0xb46f, 0xae9e

result:
    .space  24 * 24 * 8 * 2        # Space for output (24x24x8) in bytes (half precision)
