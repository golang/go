// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// When linking C ELFv2 objects, the Go linker may need to insert calling stubs.
// A call stub is usually needed when the ELFv2 st_other attribute is different
// between caller and callee.
//
// The type of call stub inserted will vary depending on GOPPC64 and the
// buildmode (e.g pie builds shared code, default builds fixed-position code).
// CI is set up to run for P8 and P10 machines, and this test is run in both
// pie and default modes.
//
// Several functions are written with interesting st_other attributes, and
// call each other to test various calling combinations which require stubs.
//
// The call tree is as follows, starting from TestPPC64Stubs (A C function):
// TestPPC64Stubs (compiled PIC by default by Go)
//   notoc_func          [called TOC -> NOTOC (but R2 is preserved)]
//     toc_func          [called NOTOC -> TOC]
//       notoc_nor2_func [called TOC -> NOTOC]
//       random          [dynamic TOC call]
//     random		 [dynamic NOTOC call]
//
// Depending on the GOPPC64/buildmode used, and type of call, one of 7 stubs may need inserted:
//
// TOC   -> NOTOC:     Save R2, call global entry. (valid for any GOPPC64)
//                      TOC save slot is rewrittent to restore TOC.
// NOTOC -> TOC [P10]: A PIC call stub using P10 instructions to call the global entry
// NOTOC -> TOC [P8]:  A PIC call stub using P8 instructions to call the global entry
//
// TOC   -> dynamic:              A PLT call stub is generated which saves R2.
//                                 TOC save slot is rewritten to restore TOC.
// NOTOC -> dynamic [P10]:        A stub using pcrel instructions is generated.
// NOTOC -> dynamic [P8/default]: A P8 compatible, non-PIC stub is generated
// NOTOC -> dynamic [P8/pie]:     A P8 compatible, PIC stub is generated
//
//
// Some notes about other cases:
//   TOC -> TOC, NOTOC -> NOTOC, NOTOC -> TOC  local calls do not require require call stubs.
//   TOC -> NOTOC (R2 is preserved, st_other==0): A special case where a call stub is not needed.

// This test requires a binutils with power10 and ELFv2 1.5 support. This is earliest verified version.
.if .gasversion. >= 23500

// A function which does not guarantee R2 is preserved.
// R2 is clobbered here to ensure the stubs preserve it.
	.globl	notoc_nor2_func
	.type	notoc_nor2_func, @function
notoc_nor2_func:
	.localentry notoc_nor2_func,1
	li	2,0
	blr

// A function which expects R2 to hold TOC, and has a distinct local entry.
	.globl	toc_func
	.type	toc_func, @function
toc_func:
	addis	2,12,.TOC.-toc_func@ha
	addi	2,2,.TOC.-toc_func@l
	.localentry toc_func, .-toc_func
	mflr	0
	std	0,16(1)
	stdu	1,-32(1)

	// Call a NOTOC function which clobbers R2.
	bl	notoc_nor2_func
	nop

	// Call libc random. This should generate a TOC relative plt stub.
	bl	random
	nop

	addi	1,1,32
	ld 	0,16(1)
	mtlr	0
	blr

// An ELFv2 st_other==0 function. It preserves R2 (TOC), but does not use it.
	.globl	notoc_func
	.type	notoc_func, @function
notoc_func:
	// Save R2 and LR and stack a frame.
	mflr	0
	std	0,16(1)
	stdu	1,-32(1)

	// Save R2 in TOC save slot.
	std	2,24(1)

	// clobber R2
	li	2,0

	// Call type2_func. A call stub from notoc to toc should be inserted.
	bl	toc_func@notoc

	// Call libc random. A notoc plt stub should be inserted.
	bl	random@notoc

	// Return 0 to indicate the test ran.
	li	3,0

	// Restore R2
	ld	2,24(1)

	// Restore LR and pop stack
	addi	1,1,32
	ld 	0,16(1)
	mtlr	0
	blr

.else

// A stub for older binutils
	.globl	notoc_func
	.type	notoc_func, @function
notoc_func:
	// Return 1 to indicate the test was skipped.
	li	3,1
	blr

.endif
