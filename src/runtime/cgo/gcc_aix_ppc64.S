// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ppc64
// +build aix

/*
 * void crosscall_ppc64(void (*fn)(void), void *g)
 *
 * Calling into the gc tool chain, where all registers are caller save.
 * Called from standard ppc64 C ABI, where r2, r14-r31, f14-f31 are
 * callee-save, so they must be saved explicitly.
 * AIX has a special assembly syntax and keywords that can be mixed with
 * Linux assembly.
 */
  .toc
  .csect .text[PR]
  .globl crosscall_ppc64
  .globl .crosscall_ppc64
  .csect crosscall_ppc64[DS]
crosscall_ppc64:
  .llong .crosscall_ppc64, TOC[tc0], 0
  .csect .text[PR]
.crosscall_ppc64:
	// Start with standard C stack frame layout and linkage
	mflr	0
	std	0, 16(1)	// Save LR in caller's frame
	std	2, 40(1)	// Save TOC in caller's frame
	bl	saveregs
	stdu	1, -296(1)

	// Set up Go ABI constant registers
	// Must match _cgo_reginit in runtime package.
	xor 0, 0, 0

	// Restore g pointer (r30 in Go ABI, which may have been clobbered by C)
	mr	30, 4

	// Call fn
	mr	12, 3
	mtctr	12
	bctrl

	addi	1, 1, 296
	bl	restoreregs
	ld	2, 40(1)
	ld	0, 16(1)
	mtlr	0
	blr

saveregs:
	// Save callee-save registers
	// O=-288; for R in {14..31}; do echo "\tstd\t$R, $O(1)"; ((O+=8)); done; for F in f{14..31}; do echo "\tstfd\t$F, $O(1)"; ((O+=8)); done
	std	14, -288(1)
	std	15, -280(1)
	std	16, -272(1)
	std	17, -264(1)
	std	18, -256(1)
	std	19, -248(1)
	std	20, -240(1)
	std	21, -232(1)
	std	22, -224(1)
	std	23, -216(1)
	std	24, -208(1)
	std	25, -200(1)
	std	26, -192(1)
	std	27, -184(1)
	std	28, -176(1)
	std	29, -168(1)
	std	30, -160(1)
	std	31, -152(1)
	stfd	14, -144(1)
	stfd	15, -136(1)
	stfd	16, -128(1)
	stfd	17, -120(1)
	stfd	18, -112(1)
	stfd	19, -104(1)
	stfd	20, -96(1)
	stfd	21, -88(1)
	stfd	22, -80(1)
	stfd	23, -72(1)
	stfd	24, -64(1)
	stfd	25, -56(1)
	stfd	26, -48(1)
	stfd	27, -40(1)
	stfd	28, -32(1)
	stfd	29, -24(1)
	stfd	30, -16(1)
	stfd	31, -8(1)

	blr

restoreregs:
	// O=-288; for R in {14..31}; do echo "\tld\t$R, $O(1)"; ((O+=8)); done; for F in {14..31}; do echo "\tlfd\t$F, $O(1)"; ((O+=8)); done
	ld	14, -288(1)
	ld	15, -280(1)
	ld	16, -272(1)
	ld	17, -264(1)
	ld	18, -256(1)
	ld	19, -248(1)
	ld	20, -240(1)
	ld	21, -232(1)
	ld	22, -224(1)
	ld	23, -216(1)
	ld	24, -208(1)
	ld	25, -200(1)
	ld	26, -192(1)
	ld	27, -184(1)
	ld	28, -176(1)
	ld	29, -168(1)
	ld	30, -160(1)
	ld	31, -152(1)
	lfd	14, -144(1)
	lfd	15, -136(1)
	lfd	16, -128(1)
	lfd	17, -120(1)
	lfd	18, -112(1)
	lfd	19, -104(1)
	lfd	20, -96(1)
	lfd	21, -88(1)
	lfd	22, -80(1)
	lfd	23, -72(1)
	lfd	24, -64(1)
	lfd	25, -56(1)
	lfd	26, -48(1)
	lfd	27, -40(1)
	lfd	28, -32(1)
	lfd	29, -24(1)
	lfd	30, -16(1)
	lfd	31, -8(1)

	blr
