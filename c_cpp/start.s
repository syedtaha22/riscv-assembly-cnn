#define STDOUT 0xd0580000

.section .text
.global _start

.extern main

_start:
    call main        # Jumps into C code's main()
    j _finish        # After main, jump to halt routine

_finish:
    li x3, 0xd0580000
    addi x5, x0, 1
    sb x5, 0(x3)     # Write 1 to STDOUT to signal "done"
    beq x0, x0, _finish  # Infinite loop
.rept 100
    nop
.endr
