// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && (ppc64 || ppc64le)

.file "gcc_linux_ppc64x.S"

// Define a frame which has no argument space, but is compatible with
// a call into a Go ABI. We allocate 32B to match FIXED_FRAME with
// similar semantics, except we store the backchain pointer, not the
// LR at offset 0. R2 is stored in the Go TOC save slot (offset 24).
.set GPR_OFFSET, 32
.set FPR_OFFSET, GPR_OFFSET + 18*8
.set VR_OFFSET, FPR_OFFSET + 18*8
.set FRAME_SIZE, VR_OFFSET + 12*16

.macro FOR_EACH_GPR opcode r=14
.ifge 31 - \r
	\opcode \r, GPR_OFFSET + 8*(\r-14)(1)
	FOR_EACH_GPR \opcode "(\r+1)"
.endif
.endm

.macro FOR_EACH_FPR opcode fr=14
.ifge 31 - \fr
	\opcode \fr, FPR_OFFSET + 8*(\fr-14)(1)
	FOR_EACH_FPR \opcode "(\fr+1)"
.endif
.endm

.macro FOR_EACH_VR opcode vr=20
.ifge 31 - \vr
	li 0, VR_OFFSET + 16*(\vr-20)
	\opcode \vr, 1, 0
	FOR_EACH_VR \opcode "(\vr+1)"
.endif
.endm

/*
 * void crosscall_ppc64(void (*fn)(void), void *g)
 *
 * Calling into the gc tool chain, where all registers are caller save.
 * Called from standard ppc64 C ABI, where r2, r14-r31, f14-f31 are
 * callee-save, so they must be saved explicitly.
 */
.globl crosscall_ppc64
crosscall_ppc64:
	// Start with standard C stack frame layout and linkage
	mflr	%r0
	std	%r0, 16(%r1)	// Save LR in caller's frame
	mfcr	%r0
	std	%r0, 8(%r1)	// Save CR in caller's frame
	stdu	%r1, -FRAME_SIZE(%r1)
	std	%r2, 24(%r1)

	FOR_EACH_GPR std
	FOR_EACH_FPR stfd
	FOR_EACH_VR stvx

	// Set up Go ABI constant registers
	li	%r0, 0

	// Restore g pointer (r30 in Go ABI, which may have been clobbered by C)
	mr	%r30, %r4

	// Call fn
	mr	%r12, %r3
	mtctr	%r3
	bctrl

	FOR_EACH_GPR ld
	FOR_EACH_FPR lfd
	FOR_EACH_VR lvx

	ld	%r2, 24(%r1)
	addi	%r1, %r1, FRAME_SIZE
	ld	%r0, 16(%r1)
	mtlr	%r0
	ld	%r0, 8(%r1)
	mtcr	%r0
	blr

#ifdef __ELF__
.section .note.GNU-stack,"",%progbits
#endif
