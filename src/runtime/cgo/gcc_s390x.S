// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

.file "gcc_s390x.S"

/*
 * void crosscall1(void (*fn)(void), void (*setg_gcc)(void*), void *g)
 *
 * Calling into the go tool chain, where all registers are caller save.
 * Called from standard s390x C ABI, where r6-r13, r15, and f8-f15 are
 * callee-save, so they must be saved explicitly.
 * ABI: r2=fn, r3=setg_gcc, r4=g.
 */
.globl crosscall1
crosscall1:
	/* save r6-r15 in the register save area of the calling function */
	stmg    %r6, %r15, 48(%r15)

	/* allocate 64 bytes of stack space to save f8-f15 */
	lay     %r15, -64(%r15)

	/* save callee-saved floating point registers */
	std     %f8, 0(%r15)
	std     %f9, 8(%r15)
	std     %f10, 16(%r15)
	std     %f11, 24(%r15)
	std     %f12, 32(%r15)
	std     %f13, 40(%r15)
	std     %f14, 48(%r15)
	std     %f15, 56(%r15)

	/* save fn pointer in r6 (already saved to stack) */
	lgr     %r6, %r2
	/* restore g pointer */
	lgr     %r13, %r4

	/* prepare arg for setg_gcc: first argument in r2 */
	lgr     %r2, %r4      /* r2 = g */
	/* call setg_gcc(g) using function pointer in r3 */
	basr    %r14, %r3

	/* call fn */
	lgr     %r2, %r6
	basr    %r14, %r2

	/* restore floating point registers */
	ld      %f8, 0(%r15)
	ld      %f9, 8(%r15)
	ld      %f10, 16(%r15)
	ld      %f11, 24(%r15)
	ld      %f12, 32(%r15)
	ld      %f13, 40(%r15)
	ld      %f14, 48(%r15)
	ld      %f15, 56(%r15)

	/* de-allocate stack frame */
	la      %r15, 64(%r15)

	/* restore general purpose registers */
	lmg     %r6, %r15, 48(%r15)

	br      %r14 /* restored by lmg */

#ifdef __ELF__
.section .note.GNU-stack,"",%progbits
#endif
