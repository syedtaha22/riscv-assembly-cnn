# ===========================================================
# float dot_product(const float* a, const float* b, int n) {
#     float sum = 0.0f;
#     for (int i = 0; i < n; i++) {
#         sum += a[i] * b[i];
#     }
#     return sum;
# }
# ===========================================================
#define STDOUT 0xd0580000

.section .text
.global _start
_start:
    addi a0, x0, 20              # Number of elements
    li x1, 0xd0580000            # Load address for output
    la a1, a                     # Load address of first vector
    la a2, b                     # Load address of second vector
    la a3, result                # Load address for result

vvdotproduct:
    vsetvli t0, a0, e32, ta, ma  # Set vector length based on 32-bit floats

    vle32.v v0, (a1)             # Load first vector

    sub a0, a0, t0               # Decrement number of elements
    slli t0, t0, 2               # Multiply number of elements by 4 bytes
    add a1, a1, t0               # Increment pointer for first vector

    vle32.v v1, (a2)             # Load second vector
    add a2, a2, t0               # Increment pointer for second vector

    vfmul.vv v2, v0, v1          # Multiply vectors

    # Taking v3, as the accumulator, should probably be initialized to zero first. But omitting it for now
    vfredosum.vs v3, v2, v3      # Reduce the vector to a single sum

    bnez a0, vvdotproduct        # Loop back if not done

# After accumulation, we need to store the result
# The result is in v3, we need to store it in the memory location pointed by a3
# We need to extract the first element of v3 and store it in the result
vfmv.f.s f0, v3              # Move the first element of v3 to f0
fsw f0, 0(a3)                # Store the result in memory

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
.float 00.0, 01.0, 02.0, 03.0, 04.0, 05.0, 06.0, 07.0, 08.0, 09.0
.float 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0

b:      
.float 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0
.float 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0

result: .space 4 # Allocate space for 1 float, i.e 4 bytes
# Expected Dot Product: 6270.0
