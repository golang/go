// Copyright 2016 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 * void crosscall_s390x(void (*fn)(void), void *g)
 *
 * Calling into the go tool chain, where all registers are caller save.
 * Called from standard s390x C ABI, where r6-r13, r15, and f0, f2, f4 and f6 are
 * callee-save, so they must be saved explicitly.
 */
.globl crosscall_s390x
crosscall_s390x:
	/*
	 * save r6-r15, f0, f2, f4 and f6 in the
	 * register save area of the calling function
	 */
	stmg	%r6, %r15, 48(%r15)
	stdy	%f0, 128(%r15)
	stdy	%f2, 136(%r15)
	stdy	%f4, 144(%r15)
	stdy	%f6, 152(%r15)

	/* set r0 to 0 */
	xgr	%r0, %r0

	/* restore g pointer */
	lgr	%r13, %r3

	/* grow stack 8 bytes and call fn */
	agfi    %r15, -8
	basr    %r14, %r2
	agfi	%r15, 8

	/* restore registers */
	lmg	%r6, %r15, 48(%r15)
	ldy	%f0, 128(%r15)
	ldy	%f2, 136(%r15)
	ldy	%f4, 144(%r15)
	ldy	%f6, 152(%r15)

	br      %r14 /* restored by lmg */

#ifdef __ELF__
.section .note.GNU-stack,"",%progbits
#endif
