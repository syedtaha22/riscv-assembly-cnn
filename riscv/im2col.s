.section .text
.global im2col


# Function: im2col
# Argument:
#   a0 - address of the input image (input image pointer)
#
#   a1 - Input dimension (CL_IN_DIM)
#   a2 - Output dimension (CL_OUT_DIM)
#   a3 - Filter dimension (CL_FILTER_DIM)
#   a4 - Stride (CL_STRIDE)
#   a5 - Patch size (CL_FILTER_DIM * CL_FILTER_DIM)
#   a6 - Number of patches (CL_OUT_DIM * CL_OUT_DIM)
# Return:
#   result pointer (a0 will hold the address of the result)
im2col:
    # Save stack
    addi sp, sp, -16            # Make space on the stack for saved registers
    sw ra, 0(sp)                # Save return address
    sw t0, 4(sp)                # Save t0

    la s6, im2col_matrix  # Load address of im2col matrix

    # Precompute Values
    mul s1, a2, a5             # odps = CL_OUT_DIM * PATCH_SIZE
    mul s2, a1, a4             # idcs = CL_IN_DIM * CL_STRIDE

    vsetvli t0, a3, e16,m1,tu,mu  # Set vector length to CL_FILTER_DIM

    # Load addresses for global variables
    li t1, 0    # i = 0
    loop_i:
        bge t1, a2, loop_i_end  # if i >= CL_OUT_DIM, exit

        # Precompute values
        mul s3, t1, s1    # s3 = i * odps = iodps
        mul s4, t1, s2    # s4 = i * idcs = iidcs


        li t2, 0    # j = 0
        loop_j:
            bge t2, a2, loop_j_end  # if j >= CL_OUT_DIM, exit

            # Calculate patch_col
            mul t3, t2, a5  # t3 = j * PATCH_SIZE
            add t3, t3, s3  # t3 = j * PATCH_SIZE + iodps = patch_col = (i * CL_OUT_DIM + j) * PATCH_SIZE

            # Calculate input_row_base
            mul t4, t2, a4  # t4 = j * CL_STRIDE
            add t4, t4, s4  # t4 = j * CL_STRIDE + iidcs = input_row_base = (i * CL_IN_DIM + j) * CL_STRIDE

            li t5, 0 # Initialize fi = 0
            loop_fi:
                bge t5, a3, loop_fi_end  # if fi >= CL_FILTER_DIM, exit

                # Calculate patch_row
                mul s5, t5, a3  # s5 = fi * CL_FILTER_DIM
                add s5, s5, t3  # s5 = patch_row = (i * CL_OUT_DIM + j) * PATCH_SIZE + fi * CL_FILTER_DIM
                slli s5, s5, 1  # Convert to byte offset (half precision)
                add s5, s6, s5  # s5 = address in im2col matrix

                # Calculate input_row
                mul t6, t5, a1  # t6 = fi * CL_IN_DIM
                add t6, t6, t4  # t6 = input_row = (i * CL_IN_DIM + j) * CL_STRIDE + fi * CL_IN_DIM
                slli t6, t6, 1  # Convert to byte offset (half precision)
                add t6, a0, t6  # t6 = address of input_row in input image

                vle16.v v1, (t6)  # Load input patch from input image into v1
                vse16.v v1, (s5)  # Store the input patch in im2col matrix

                addi t5, t5, 1  # Increment fi
                j loop_fi  # Repeat for the next fi
            loop_fi_end:
            addi t2, t2, 1  # Increment j
            j loop_j  # Repeat for the next j
        loop_j_end:
        addi t1, t1, 1  # Increment i
        j loop_i  # Repeat for the next i
    loop_i_end:
        # At this point, all patches have been processed and stored in im2col_matrix
        # a0 will hold the address of the im2col matrix
        la a0, im2col_matrix  # Set a0 to the address of the im2col matrix
    # Restore registers and return from the function
    lw t0, 4(sp)                  # Restore t0
    lw ra, 0(sp)                  # Restore return address  
    addi sp, sp, 16               # Restore stack pointer
    ret                           # Return from the function


.data
.align 4

# im2col matrix
im2col_matrix:
    .space  24 * 24 * 25 * 2        # Space for im2col matrix (24x24 patches, each 5x5 = 25 elements)


