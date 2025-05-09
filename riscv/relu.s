
// Assembly code for ReLU function
.text
.globl ReLU_Vector

# Function: ReLU_Vector
# Applies ReLU activation (max(x, 0.0)) on an array of 32-bit floats using RISC-V vector instructions.
# Arguments:
#   a0 - pointer to input float array
#   a1 - number of elements in the array

ReLU_Vector:
    mv t0, a0               # t0 = address of input array
    mv t1, a1               # t1 = number of elements to process

    li t3, 0x00000000       # t3 = integer 0 (bit pattern for float 0.0)
    fmv.w.x f0, t3          # f0 = floating-point 0.0 (to be used for ReLU threshold)

Loop:
    beqz t1, Done           # If no elements left, exit loop

    vsetvli t2, t1, e32     # Set vector length (VL) for 32-bit floats based on remaining elements
    vle32.v v1, (t0)        # Load a chunk of input floats from memory into vector register v1

    vmv.v.f v0, f0          # Fill vector register v0 with float 0.0 (used to compare against input)

    vfmax.vv v2, v1, v0     # Compute element-wise max between input vector and zero: v2[i] = max(v1[i], 0.0)

    vse32.v v2, (t0)        # Store result vector back to memory at the input location

    slli t4, t2, 2          # t4 = number of bytes processed this iteration (t2 * 4 bytes per float)
    add t0, t0, t4          # Move input pointer forward by number of bytes processed
    sub t1, t1, t2          # Decrease number of elements left to process
    j Loop                  

Done:
    ret                     
