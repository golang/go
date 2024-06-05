// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

.file "gcc_arm.S"

/*
 * void crosscall1(void (*fn)(void), void (*setg_gcc)(void *g), void *g)
 *
 * Calling into the gc tool chain, where all registers are caller save.
 * Called from standard ARM EABI, where r4-r11 are callee-save, so they
 * must be saved explicitly.
 */
.globl crosscall1
crosscall1:
	push {r4, r5, r6, r7, r8, r9, r10, r11, ip, lr}
	mov r4, r0
	mov r5, r1
	mov r0, r2

	// Because the assembler might target an earlier revision of the ISA
	// by default, we encode BLX as a .word.
	.word 0xe12fff35 // blx r5 // setg(g)
	.word 0xe12fff34 // blx r4 // fn()

	pop {r4, r5, r6, r7, r8, r9, r10, r11, ip, pc}


#ifdef __ELF__
.section .note.GNU-stack,"",%progbits
#endif
