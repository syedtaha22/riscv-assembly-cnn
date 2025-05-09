# ===================================================================
# void vector_max(const int* a, const int* b, int* result, int n) {
#     for (int i = 0; i < n; i++) {
#         result[i] = (a[i] > b[i]) ? a[i] : b[i];
#     }
# }
# ===================================================================
#define STDOUT 0xd0580000

.section .text
.global _start
_start:

    addi a0, x0, 4               # Number of elements
    li x1, 0xd0580000            # Load address for output
    la a1, a                     # Load address of first vector
    la a2, b                     # Load address of second vector
    la a3, result                # Load address for result

elementwise_max:
    vsetvli t0, a0, e32, ta, ma  # Set vector length based on 32-bit floats
    sub a0, a0, t0               # Decrement number of elements
    slli t0, t0, 2               # Multiply number of elements by 4 bytes

    vle32.v v0, (a1)             # Load first vector
    add a1, a1, t0               # Increment pointer for first vector

    vle32.v v1, (a2)             # Load second vector
    add a2, a2, t0               # Increment pointer for second vector


    vfmax.vv v2, v0, v1          # Element-wise maximum
    vse32.v v2, (a3)             # Store result
    add a3, a3, t0               # Increment pointer for result
    bnez a0, elementwise_max     # Loop back if not done

_finish:
    li x3, 0xd0580000
    addi x5, x0, 0xff
    sb x5, 0(x3)
    beq x0, x0, _finish
.rept 100
    nop
.endr


.data
a:      
.float 01.0, 02.0, 03.0, 04.0

b:      
.float 05.0, 05.0, 00.1, 0.2

result: .space 816 # Allocate space for 4 floats, i.e 4x4 bytes
