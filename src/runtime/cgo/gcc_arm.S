// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 * Apple still insists on underscore prefixes for C function names.
 */
#if defined(__APPLE__)
#define EXT(s) _##s
#else
#define EXT(s) s
#endif

/*
 * void crosscall_arm1(void (*fn)(void), void (*setg_gcc)(void *g), void *g)
 *
 * Calling into the 5c tool chain, where all registers are caller save.
 * Called from standard ARM EABI, where r4-r11 are callee-save, so they
 * must be saved explicitly.
 */
.globl EXT(crosscall_arm1)
EXT(crosscall_arm1):
	push {r4, r5, r6, r7, r8, r9, r10, r11, ip, lr}
	mov r4, r0
	mov r5, r1
	mov r0, r2

	// Because the assembler might target an earlier revision of the ISA
	// by default, we encode BLX as a .word.
	.word 0xe12fff35 // blx r5 // setg(g)
	.word 0xe12fff34 // blx r4 // fn()

	pop {r4, r5, r6, r7, r8, r9, r10, r11, ip, pc}

.globl EXT(__stack_chk_fail_local)
EXT(__stack_chk_fail_local):
1:
	b 1b

#ifdef __ELF__
.section .note.GNU-stack,"",%progbits
#endif
