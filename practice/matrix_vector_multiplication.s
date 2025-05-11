# ============================================================
# void matvec_mul(int* mat, int* vec, int* out, int rows, int cols) {
#     for (int i = 0; i < rows; i++) {
#         int sum = 0;
#         for (int j = 0; j < cols; j++) {
#             sum += mat[i * cols + j] * vec[j];
#         }
#         out[i] = sum;
#     }
# }
#
# Observation:
#    Inner loop can be vectorized
# ============================================================
#define STDOUT 0xd0580000

.section .text
.global _start
_start:

    li x1, 0xd0580000            # Load address for output

    la a1, a                                # Load address of the Matrix (stored in row-major order)
    la a2, b                                # Load address of the vector
    la a3, result                           # Load address for result
    la a4, MATRIX_DIMS                      # Load address of matrix dimensions

    lw a0, 0(a4)                            # Load number of rows

    # Initliaze counter for outer loop (i.e. rows)
    li t1, 0                                # i = 0

    outer_loop:
        bge t1, a0, outer_loop_end          # if t1 > a0 then outer_loop_end
        lw t2, 0(a4)                        # Load number of columns

        # Initialize the accumulator to 0
        vsetvli t0, t2, e32, m1             # Set vector length (e32 for 32-bit elements)
        vmv.v.x v3, x0                      # Set all elements of v3 to zero

        matrix_vector_multiply:
            # Set vector length
            vsetvli t0, t2, e32, ta, ma     # Set vector length based on 32-bit floats
            sub t2, t2, t0                  # Decrement number of elements
            slli t0, t0, 2                  # Multiply number of elements by 4 bytes

            vle32.v v0, (a1)                # Load first vector (row of matrix)
            add a1, a1, t0                  # Increment pointer for first vector

            vle32.v v1, (a2)                # Load second vector to be multiplied
            add a2, a2, t0                  # Increment pointer for second vector

            # Element-wise multiplication
            vfmul.vv v2, v0, v1             # Multiply vectors

            # Reduce the vector to a single sum
            vfredosum.vs v3, v2, v3         # Reduce the vector to a single sum

            # Store the result
            vse32.v v3, (a3)                # Store the result
            add a3, a3, t0                  # Increment pointer for result
            bnez t2, matrix_vector_multiply # Loop back if not done
        
        # Increment the row pointer
        addi t1, t1, 1                # Increment row counter
        la a2, b              # Reset vector pointer
        j outer_loop          # Jump to outer loop
    outer_loop_end:

_finish:
    li x3, 0xd0580000
    addi x5, x0, 0xff
    sb x5, 0(x3)
    beq x0, x0, _finish
.rept 100
    nop
.endr


.data
MATRIX_DIMS: .word 3
a:      
    .float 01.0, 02.0, 03.0
    .float 04.0, 05.0, 06.0
    .float 07.0, 08.0, 09.0
b:      
    .float 01.0, 02.0, 03.0

result: .space 12 # Allocate space for 3 floats, i.e 4x3 bytes
# Expected output: 14.0, 32.0, 50.0
# In Hex (IEEE 754 float32): 0x41600000, 0x42000000, 0x42480000
