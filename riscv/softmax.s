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
    lw a2, 0(a1)                # Load number of elements from memory

    la t4, LN2          # Load ln(2) address
    flw f4, 0(t4)        # Load ln(2) from memory
    flw f5, 4(t4)        # Load 1/ln(2) from memory

    # Initialize accumulator for sum(exp(x))
    addi t0, zero, 1
    vsetvli t0, t0, e32, ta, ma     # Set vector config, VL into t0
    vmv.v.i v4, 0                   # v4 = 0.0

    reduce_range:
        vsetvli t6, a2, e32, ta, ma     # Set vector config, VL into t6
        sub a2, a2, t6                  # Decrement number of elements
        slli t6, t6, 2                  # Multiply number of elements by 4 bytes


        # Do this first so that we can use v1 for 1/2^n later
        # --- Taylor series: result = 1.0, term = 1.0 ---
        addi t3, zero, 1                # t3 = 1
        fcvt.s.w f0, t3                 # f0 = float(1)
        vfmv.v.f v1, f0                 # v1 = broadcast 1.0
        vfmv.v.f v2, f0                 # v2 = term vector

        vle32.v v8, (a0)                # Load input[] into v8 (x)

        # Calculate n = (int)(x / ln(2)) ---> v6
        vfmul.vf v6, v8, f5              # v6 = x * (1/ln(2))
        vfcvt.x.f.v v6, v6              # Convert to integer

        # Broad cast 2 in to v7
        vmv.v.i v7, 2                     # Initialize v7 with 2 for base of exponentiation
        vmslt.vi v0, v6, 0                # v0 = mask where v6 < 0 (n is negative)
        vneg.v v9, v6                     # v9 = -v6 (absolute value of n for negative cases)
        vsll.vv v7, v7, v6                # v7 = 2^n (works correctly only for non-negative n)
        vmv.v.i v11, 2                    # Initialize v11 with 2
        vsll.vv v11, v11, v9              # v11 = 2^|n| for negative n
        vfcvt.f.x.v v11, v11              # Convert 2^|n| to float
        vfdiv.vv v11, v1, v11             # v11 = 1 / (2^|n|) for negative n
        vmerge.vvm v7, v11, v7, v0        # For negative n (mask true), use 1/2^|n|; else keep 2^n


        # Calculate r = x - n * ln(2) ---> v5
        vfcvt.f.x.v v6, v6              # Convert n back to float
        vfmul.vf v5, v6, f4              # v5 = n * ln(2)
        vfsub.vv v5, v8, v5              # v5 = x - n * ln(2)

        vmv.v.i v3, 1                   # v3 = 1.0 (for i=1)

        li t2, 10                  # terms = 10

        li t1, 1                     # i = 1
        exp_loop:
            bge t1, t2, exp_done        # while (i < 10)

            vfcvt.f.x.v v9, v3          # Convert integer i to float
            vadd.vi v3, v3, 1           # i++

            vfmul.vv v2, v2, v5         # term *= x
            vfdiv.vv v2, v2, v9         # term /= i!
            vfadd.vv v1, v1, v2         # result += term

            addi t1, t1, 1              # i++
            j exp_loop
        exp_done:

        # Now we have exp(r) in v1, we need to multiply it by 2^n
        vfmul.vv v1, v1, v7            # v1 = exp(r) * 2^n
        # v1 now contains exp(x) for the reduced range
        # Store exp(x) back to input[]
        vse32.v v1, (a0)            # Store exp(x) back to input[]
        add a0, a0, t6              # Increment pointer for input
        vfredosum.vs v4, v1, v4     # v4[0] = sum(exp(x))
        
        bnez a2, reduce_range       # if (a2 != 0 i.e more elements) continue
        
    lw a0, 4(sp)                # Restore input array pointer
    lw a2, 0(a1)                # Load number of elements

    # Load the sum of exponentials
    vfmv.f.s f0, v4                    # f0 = sum(exp(x))

    # Normalize the exponentials
    normalize:
        vsetvli t0, a2, e32, ta, ma     # Set vector config, VL into t0
        sub a2, a2, t0                  # Decrement number of elements
        slli t0, t0, 2                  # Multiply number of elements by 4 bytes
        
        vfmv.v.f v1, f0                 # v1 = broadcast(sum)

        vle32.v v8, (a0)                # Load input[] into v8 (x)
        vfdiv.vv v8, v8, v1             # v8 = exp(x) / sum(exp(x))

        # Store back result
        vse32.v v8, (a0)
        add a0, a0, t0                  # Increment pointer for input

        bnez a2, normalize              # if (a2 != 0 i.e more elements) continue
    
    # Restore return address and return    
    lw ra, 0(sp)
    lw a0, 4(sp)                # Restore input array pointer
    addi sp, sp, 8
    ret 

.section .data

array_size:    .word 10

LN2:      .word 0x3f317218    # ln(2) ≈ 0.69314718
LN2_INV:  .word 0x3fb8aa3b    # 1/ln(2) ≈ 1.44269504
