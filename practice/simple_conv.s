# ==============================================================================
# #define INPUT_DIM 3
# #define FILTER_DIM 2
# #define STRIDE 1
#
# Assuming square filter and square input, and that filter and bias is globally
# declared
#
# void conv2d(float* input) {
#     int output_dim = INPUT_DIM - FILTER_DIM + 1;
#
#     for (int i = 0; i < output_dim; ++i) {
#         for (int j = 0; j < output_dim; ++j) {
#             float sum = 0.0f;
#
#             for (int fi = 0; fi < FILTER_DIM; ++fi) {
#                 for (int fj = 0; fj < FILTER_DIM; ++fj) {
#                     int input_index = (i + fi) * INPUT_DIM + (j + fj);
#                     int filter_index = fi * FILTER_DIM + fj;
#                     sum += input[input_index] * filter[filter_index];
#                 }
#             }
#
#             int output_index = i * output_dim + j;
#             output[output_index] = sum + bias;
#         }
#     }
# }
# ==============================================================================
#define STDOUT 0xd0580000

.section .text
.global _start
_start:

    addi a0, x0, 20              # Number of elements
    li x1, 0xd0580000            # Load address for output
    la a1, image                 # Load address of Image
    la a2, filter                # Load address of filter
    la a3, result                # Load address for result
    la a4, INPUT_DIM             # Load address of input dimensions
    la a6, bias                  # Load address of bias

    lw a5, 4(a4)                 # Load filter dimension
    lw a7, 12(a4)                # Load output dimension
    lw a4, 0(a4)                 # Load input dimension

    lw a6, 0(a6)                 # Load bias

    mul s0, a5, a5               # s0 = FILTER_DIM * FILTER_DIM = Number of elements to process

    # Initialize the counter for outer loop (i.e. output_dim)
    # TODO: Reduce the number of loops, by leveraging the fact that everything is a 1d array
    li t1, 0                     # i = 0
    loop_i:
        bge t1, a7, loop_i_end   # if i >= output_dim, exit
        li t2, 0                 # j = 0

        loop_j:
            bge t2, a7, loop_j_end   # if j >= output_dim, exit

            vsetvli t0, s0, e32, m1  # vector length = FILTER_DIM²
            vmv.v.x v0, x0       # Set all elements of v0 to zero

            li t3, 0                 # fi = 0 (filter row)
            conv_row:
                bge t3, a5, conv_end     # if fi >= FILTER_DIM, break

                # row_offset = (i + fi) * INPUT_DIM + j
                add t4, t1, t3           # row = i + fi
                mul t5, t4, a4           # row_offset = row * INPUT_DIM
                add t5, t5, t2           # + col offset (j)
                slli t5, t5, 2           # byte offset = *4
                add t6, a1, t5           # &image[row_offset]

                # Load input row with stride = 4
                li s1, 4
                vsetvli t0, a5, e32, m1
                vlse32.v v1, 0(t6), s1      # input patch row

                # Load filter row (contiguous memory)
                mul s2, t3, a5           # fi * FILTER_DIM
                slli s2, s2, 2           # byte offset
                add s3, a2, s2           # &filter[fi * FILTER_DIM]
                vle32.v v2, 0(s3)           # filter row

                # Accumulate dot product
                vfmacc.vv v0, v1, v2

                addi t3, t3, 1           # fi++
                j conv_row

            conv_end:
                # Sum reduction
                vsetvli t0, zero, e32, m1
                vmv.s.x v3, a6           # Initialize v3 with bias
                vfredsum.vs v3, v0, v3

                # Store result
                vse32.v v3, 0(a3)
                addi a3, a3, 4

                addi t2, t2, 1           # j++
                j loop_j

        loop_j_end:
            addi t1, t1, 1               # i++
            j loop_i
    loop_i_end:

_finish:
    li x3, 0xd0580000
    addi x5, x0, 0xff
    sb x5, 0(x3)
    beq x0, x0, _finish
.rept 100
    nop
.endr

.data
INPUT_DIM:      .word 3
FILTER_DIM:     .word 2
STRIDE:         .word 1
OUT_DIM:        .word 2

image:
    .float  1.0, 2.0, 3.0
    .float  4.0, 5.0, 6.0
    .float  7.0, 8.0, 9.0

filter:
    .float  1.0, 2.0
    .float  3.0, 4.0

bias:
    .float  1.0

result:
    .space 16
# Expected output: 38.0, 48.0, 68.0, 78.0