// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ppc64 ppc64le

/*
 * Apple still insists on underscore prefixes for C function names.
 */
#if defined(__APPLE__)
#define EXT(s) _##s
#else
#define EXT(s) s
#endif

/*
 * void crosscall_ppc64(void (*fn)(void), void *g)
 *
 * Calling into the gc tool chain, where all registers are caller save.
 * Called from standard ppc64 C ABI, where r2, r14-r31, f14-f31 are
 * callee-save, so they must be saved explicitly.
 */
.globl EXT(crosscall_ppc64)
EXT(crosscall_ppc64):
	// Start with standard C stack frame layout and linkage
	mflr	%r0
	std	%r0, 16(%r1)	// Save LR in caller's frame
	std	%r2, 24(%r1)	// Save TOC in caller's frame
	bl	saveregs
	stdu	%r1, -296(%r1)

	// Set up Go ABI constant registers
	bl	_cgo_reginit
	nop

	// Restore g pointer (r30 in Go ABI, which may have been clobbered by C)
	mr	%r30, %r4

	// Call fn
	mr	%r12, %r3
	mtctr	%r3
	bctrl

	addi	%r1, %r1, 296
	bl	restoreregs
	ld	%r2, 24(%r1)
	ld	%r0, 16(%r1)
	mtlr	%r0
	blr

saveregs:
	// Save callee-save registers
	// O=-288; for R in %r{14..31}; do echo "\tstd\t$R, $O(%r1)"; ((O+=8)); done; for F in f{14..31}; do echo "\tstfd\t$F, $O(%r1)"; ((O+=8)); done
	std	%r14, -288(%r1)
	std	%r15, -280(%r1)
	std	%r16, -272(%r1)
	std	%r17, -264(%r1)
	std	%r18, -256(%r1)
	std	%r19, -248(%r1)
	std	%r20, -240(%r1)
	std	%r21, -232(%r1)
	std	%r22, -224(%r1)
	std	%r23, -216(%r1)
	std	%r24, -208(%r1)
	std	%r25, -200(%r1)
	std	%r26, -192(%r1)
	std	%r27, -184(%r1)
	std	%r28, -176(%r1)
	std	%r29, -168(%r1)
	std	%r30, -160(%r1)
	std	%r31, -152(%r1)
	stfd	%f14, -144(%r1)
	stfd	%f15, -136(%r1)
	stfd	%f16, -128(%r1)
	stfd	%f17, -120(%r1)
	stfd	%f18, -112(%r1)
	stfd	%f19, -104(%r1)
	stfd	%f20, -96(%r1)
	stfd	%f21, -88(%r1)
	stfd	%f22, -80(%r1)
	stfd	%f23, -72(%r1)
	stfd	%f24, -64(%r1)
	stfd	%f25, -56(%r1)
	stfd	%f26, -48(%r1)
	stfd	%f27, -40(%r1)
	stfd	%f28, -32(%r1)
	stfd	%f29, -24(%r1)
	stfd	%f30, -16(%r1)
	stfd	%f31, -8(%r1)

	blr

restoreregs:
	// O=-288; for R in %r{14..31}; do echo "\tld\t$R, $O(%r1)"; ((O+=8)); done; for F in %f{14..31}; do echo "\tlfd\t$F, $O(%r1)"; ((O+=8)); done
	ld	%r14, -288(%r1)
	ld	%r15, -280(%r1)
	ld	%r16, -272(%r1)
	ld	%r17, -264(%r1)
	ld	%r18, -256(%r1)
	ld	%r19, -248(%r1)
	ld	%r20, -240(%r1)
	ld	%r21, -232(%r1)
	ld	%r22, -224(%r1)
	ld	%r23, -216(%r1)
	ld	%r24, -208(%r1)
	ld	%r25, -200(%r1)
	ld	%r26, -192(%r1)
	ld	%r27, -184(%r1)
	ld	%r28, -176(%r1)
	ld	%r29, -168(%r1)
	ld	%r30, -160(%r1)
	ld	%r31, -152(%r1)
	lfd	%f14, -144(%r1)
	lfd	%f15, -136(%r1)
	lfd	%f16, -128(%r1)
	lfd	%f17, -120(%r1)
	lfd	%f18, -112(%r1)
	lfd	%f19, -104(%r1)
	lfd	%f20, -96(%r1)
	lfd	%f21, -88(%r1)
	lfd	%f22, -80(%r1)
	lfd	%f23, -72(%r1)
	lfd	%f24, -64(%r1)
	lfd	%f25, -56(%r1)
	lfd	%f26, -48(%r1)
	lfd	%f27, -40(%r1)
	lfd	%f28, -32(%r1)
	lfd	%f29, -24(%r1)
	lfd	%f30, -16(%r1)
	lfd	%f31, -8(%r1)

	blr


#ifdef __ELF__
.section .note.GNU-stack,"",%progbits
#endif
