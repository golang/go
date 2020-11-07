// Copyright 2015 The Go Authors. All rights reserved.
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

// Apple's ld64 wants 4-byte alignment for ARM code sections.
// .align in both Apple as and GNU as treat n as aligning to 2**n bytes.
.align	2

/*
 * void crosscall1(void (*fn)(void), void (*setg_gcc)(void *g), void *g)
 *
 * Calling into the gc tool chain, where all registers are caller save.
 * Called from standard ARM EABI, where x19-x29 are callee-save, so they
 * must be saved explicitly, along with x30 (LR).
 */
.globl EXT(crosscall1)
EXT(crosscall1):
	.cfi_startproc
	stp x29, x30, [sp, #-96]!
	.cfi_def_cfa_offset 96
	.cfi_offset 29, -96
	.cfi_offset 30, -88
	mov x29, sp
	.cfi_def_cfa_register 29
	stp x19, x20, [sp, #80]
	.cfi_offset 19, -16
	.cfi_offset 20, -8
	stp x21, x22, [sp, #64]
	.cfi_offset 21, -32
	.cfi_offset 22, -24
	stp x23, x24, [sp, #48]
	.cfi_offset 23, -48
	.cfi_offset 24, -40
	stp x25, x26, [sp, #32]
	.cfi_offset 25, -64
	.cfi_offset 26, -56
	stp x27, x28, [sp, #16]
	.cfi_offset 27, -80
	.cfi_offset 28, -72

	mov x19, x0
	mov x20, x1
	mov x0, x2

	blr x20
	blr x19

	ldp x27, x28, [sp, #16]
	.cfi_restore 27
	.cfi_restore 28
	ldp x25, x26, [sp, #32]
	.cfi_restore 25
	.cfi_restore 26
	ldp x23, x24, [sp, #48]
	.cfi_restore 23
	.cfi_restore 24
	ldp x21, x22, [sp, #64]
	.cfi_restore 21
	.cfi_restore 22
	ldp x19, x20, [sp, #80]
	.cfi_restore 19
	.cfi_restore 20
	ldp x29, x30, [sp], #96
	.cfi_restore 29
	.cfi_restore 30
	.cfi_def_cfa 31, 0
	ret
	.cfi_endproc


#ifdef __ELF__
.section .note.GNU-stack,"",%progbits
#endif
