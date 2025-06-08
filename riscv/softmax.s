.section .text
.global softmax
# Function: softmax
# Arguments:
#   a0 - address of the input array 
# Returns:
#   a0 - modifies the input array in place.
softmax:
    # Save return address
    addi sp, sp, -8
    sw ra, 0(sp)
    sw a0, 4(sp)                # Save input array pointer

    la a1, array_size           # Load number of elements
    lw a1, 0(a1)                # Load number of elements from memory

    # Initialize accumulator for sum
    addi t0, zero, 1
    vsetvli t0, t0, e32, ta, ma     # Set vector config, VL into t0
    vmv.v.i v4, 0                   # v4 = 0.0

    exponentiation:
        vsetvli t0, a1, e32, ta, ma         # Set vector config, VL into t0
        sub a1, a1, t0                      # Decrement number of elements
        slli t0, t0, 2                      # Multiply number of elements by 4 bytes

        vle32.v v0, (a0)                   # Load input[] into v0 (x)

        # --- Taylor series: result = 1.0, term = 1.0 ---
        addi t3, zero, 1                # t0 = 1
        fcvt.s.w f0, t3                 # f0 = float(1)
        vfmv.v.f v1, f0                 # v1 = broadcast 1.0
        vfmv.v.f v2, f0                 # v2 = term vector

        li t2, 1000                  # terms = 1000

        li t1, 1                     # i = 1
        exp_loop:
            bge t1, t2, exp_done        # while (i < 1000)

            vfmul.vv v2, v2, v0         # term *= x
            fcvt.s.w f0, t1             # f0 = (float)i
            vfmv.v.f v3, f0             # v3 = float(i) broadcast
            vfdiv.vv v2, v2, v3         # term /= i!
            vfadd.vv v1, v1, v2         # result += term

            addi t1, t1, 1              # i++
            j exp_loop
        exp_done:
            vmv.v.v v0, v1               # v0 = exp(x)

            vse32.v v0, (a0)            # Store exp(x) back to input[]
            add a0, a0, t0              # Increment pointer for input

            vfredosum.vs v4, v0, v4     # v4[0] = sum(exp(x))
            
            bnez a1, exponentiation     # if (a1 != 0 i.e more elements) continue
    exponentiation_done:
        lw a0, 4(sp)                # Restore input array pointer

        la a1, array_size           # Load number of elements
        lw a1, 0(a1)                # Load number of elements

        # Load the sum of exponentials
        vfmv.f.s f0, v4                    # f0 = sum(exp(x))

    # Normalize the exponentials
    normalize:
        vsetvli t0, a1, e32, ta, ma     # Set vector config, VL into t0
        sub a1, a1, t0                  # Decrement number of elements
        slli t0, t0, 2                  # Multiply number of elements by 4 bytes
        
        vfmv.v.f v1, f0                 # v1 = broadcast(sum)

        vle32.v v0, (a0)                # Load input[] into v0 (x)
        vfdiv.vv v0, v0, v1             # v0 = exp(x) / sum(exp(x))

        # Store back result
        vse32.v v0, (a0)
        add a0, a0, t0                  # Increment pointer for input

        bnez a1, normalize              # if (a1 != 0 i.e more elements) continue
    
    # Restore return address and return    
    lw ra, 0(sp)
    lw a0, 4(sp)                # Restore input array pointer
    addi sp, sp, 8
    ret 

.section .data

array_size:    .word 10
