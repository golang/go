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
	stp x19, x20, [sp, #-16]!
	stp x21, x22, [sp, #-16]!
	stp x23, x24, [sp, #-16]!
	stp x25, x26, [sp, #-16]!
	stp x27, x28, [sp, #-16]!
	stp x29, x30, [sp, #-16]!
	mov x29, sp

	mov x19, x0
	mov x20, x1
	mov x0, x2

	blr x20
	blr x19

	ldp x29, x30, [sp], #16
	ldp x27, x28, [sp], #16
	ldp x25, x26, [sp], #16
	ldp x23, x24, [sp], #16
	ldp x21, x22, [sp], #16
	ldp x19, x20, [sp], #16
	ret


#ifdef __ELF__
.section .note.GNU-stack,"",%progbits
#endif
