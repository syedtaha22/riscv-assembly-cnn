.section .text
.global conv2d
.extern im2col

# Function: conv2d
# Argument:
#   a0 - address of the input image (input image pointer)
# Return:
#   result pointer (a0 will hold the address of the result)
conv2d:
    # Save stack
    addi sp, sp, -8            # Make space on the stack for saved registers
    sw ra, 0(sp)                # Save return address

    # Load addresses for global variables
    la t0, CL_IN_DIM            # Load input dimension address
    lw a1, 0(t0)                # Load input dimension
    lw a2, 4(t0)                # Load output dimension
    lw a3, 8(t0)                # Load filter dimension
    lw a4, 12(t0)               # Load stride
    lw a7, 20(t0)  # Load number of filters (CL_NUM_FILTERS)
    
    mul a5, a3, a3        # a5 = CL_FILTER_DIM * CL_FILTER_DIM = PATCH_SIZE
    mul a6, a2, a2        # a6 = CL_OUT_DIM * CL_OUT_DIM = NUM_PATCHES

    call im2col  # Call im2col function to prepare input patches

    la a1, conv_filters  # Load address of convolution filters
    la a2, conv_biases   # Load address of convolution biases
    la a4, result  # Load address of result

    slli s4, a5, 1  # s4 = PATCH_SIZE * 2 (for half precision)

    li t1, 0  # Initialize filter index (f = 0)
    loop_f:
        bge t1, a7, loop_f_end  # If f >= CL_NUM_FILTERS, exit

        mv s2, a0  # iptr = im2col_matrix

        li t2, 0  # Initialize patch index (p = 0)
        loop_p:
            bge t2, a6, loop_p_end  # If p >= NUM_PATCH

            # broadcast bias to v1
            vsetivli t4, 1, e16  # Set vector length to 1
            vle16.v v1, (a2)  # Load bias value

            mv s3, a1  # s3 = address of conv_filters[f]
            mv t6, a5   # t6 = PATCH_SIZE

            patch_product:
                vsetvli t5, t6, e16  # Set vector length to PATCH_SIZE
                sub t6, t6, t5       # Decrement patch size
                slli s1, t5, 1       # vector byte length

                vle16.v v2, (s2)  # Load input patch into v2
                vle16.v v3, (s3)  # Load filter weights into v3

                add s2, s2, s1    # Move to the next patch in im2col_matrix
                add s3, s3, s1    # Move to the next filter weights

                vfmul.vv v2, v2, v3  # Multiply filter weights with input patch
                vfredosum.vs v1, v2, v1  # Reduce the result and accumulate in v1

                bnez t6, patch_product  # If there are more elements in the patch, continue

            vmv.x.s t3, v1  # Move the result from v1 to t3
            sh t3, 0(a4)    # Store the result in the output

            addi a4, a4, 2  # Move to the next output element
            
            addi t2, t2, 1  # Increment patch index (p += 1)
            j loop_p  # Repeat for the next patch
        loop_p_end:

        add  a1, a1, s4  # Move to the next filter weights
        addi a2, a2, 2  # Move to the next bias

        addi t1, t1, 1  # Increment filter index (f += 1)
        j loop_f  # Repeat for the next filter
    loop_f_end:
        la a0, result  # Set a0 to the address of the result

    lw ra, 0(sp)  # Restore return address
    addi sp, sp, 8  # Restore stack pointer
    ret  # Return from the function
    # a0 contains the address of the im2col matrix


.data
.align 4

# Definitions from #define (from the C code)
CL_IN_DIM:         .word 28              # Input dimension (28x28)
CL_OUT_DIM:        .word 24              # Output dimension (24x24)
CL_FILTER_DIM:     .word 05              # Filter dimension (5x5)
CL_STRIDE:         .word 01              # Stride (1)
CL_IN_CHANNELS:    .word 01              # Input channels (1)
CL_NUM_FILTERS:    .word 8              # Number of filters (8)



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


# im2col matrix
im2col_matrix:
    .space  24 * 24 * 25 * 2        # Space for im2col matrix (24x24 patches, each 5x5 = 25 elements)


