.section .text
.global relu


# Function: relu
# Arguments:
#   a0 - address of the array (float*)
#   a1 - number of elements (int)
# Behavior:
#   Applies ReLU in-place (modifies the array)
relu:
    # Save return address
    addi sp, sp, -8
    sw ra, 0(sp)
    sw a0, 4(sp)                # Save input array pointer

    la a1, RELU_IN_ELEMENTS   # Load number of elements
    lw a1, 0(a1)              # Load number of elements from memory

    vsetvli t0, zero, e32, m1, ta, ma 
    vmv.v.i v2, 0             # v5 = 0.0 (ReLU threshold)

    relu_loop:
        vsetvli t0, a1, e32, m1, ta, ma     # Set vector length
        sub a1, a1, t0              # Decrease remaining count
             
        vle32.v v1, (a0)            # Load input
        vmfgt.vv v0, v1, v2         # v0 = (v1 > 0.0) ? 1 : 0 (mask)
        vmerge.vvm v1, v2, v1, v0   # v1 = (v1 > 0.0) ? v1 : 0.0
        vse32.v v1, (a0)            # Store result back

        slli t0, t0, 2              # Convert elements to bytes
        add a0, a0, t0              # Advance array pointer

        bnez a1, relu_loop          # Loop if elements remain

    # Restore return address and return
    lw ra, 0(sp)
    lw a0, 4(sp)                # Restore input array pointer
    addi sp, sp, 8
    ret

.section .data

RELU_IN_ELEMENTS:    .word 4608
