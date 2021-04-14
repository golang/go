// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build mips64 mips64le

/*
 * void crosscall1(void (*fn)(void), void (*setg_gcc)(void *g), void *g)
 *
 * Calling into the gc tool chain, where all registers are caller save.
 * Called from standard MIPS N64 ABI, where $16-$23, $28, $30, and $f24-$f31
 * are callee-save, so they must be saved explicitly, along with $31 (LR).
 */
.globl crosscall1
.set noat
crosscall1:
#ifndef __mips_soft_float
	daddiu	$29, $29, -160
#else
	daddiu	$29, $29, -96 // For soft-float, no need to make room for FP registers
#endif
	sd	$31, 0($29)
	sd	$16, 8($29)
	sd	$17, 16($29)
	sd	$18, 24($29)
	sd	$19, 32($29)
	sd	$20, 40($29)
	sd	$21, 48($29)
	sd	$22, 56($29)
	sd	$23, 64($29)
	sd	$28, 72($29)
	sd	$30, 80($29)
#ifndef __mips_soft_float
	sdc1	$f24, 88($29)
	sdc1	$f25, 96($29)
	sdc1	$f26, 104($29)
	sdc1	$f27, 112($29)
	sdc1	$f28, 120($29)
	sdc1	$f29, 128($29)
	sdc1	$f30, 136($29)
	sdc1	$f31, 144($29)
#endif

	// prepare SB register = pc & 0xffffffff00000000
	bal	1f
1:
	dsrl	$28, $31, 32
	dsll	$28, $28, 32

	move	$20, $4 // save R4
	move	$1, $6
	jalr	$5	// call setg_gcc (clobbers R4)
	jalr	$20	// call fn

	ld	$16, 8($29)
	ld	$17, 16($29)
	ld	$18, 24($29)
	ld	$19, 32($29)
	ld	$20, 40($29)
	ld	$21, 48($29)
	ld	$22, 56($29)
	ld	$23, 64($29)
	ld	$28, 72($29)
	ld	$30, 80($29)
#ifndef __mips_soft_float
	ldc1	$f24, 88($29)
	ldc1	$f25, 96($29)
	ldc1	$f26, 104($29)
	ldc1	$f27, 112($29)
	ldc1	$f28, 120($29)
	ldc1	$f29, 128($29)
	ldc1	$f30, 136($29)
	ldc1	$f31, 144($29)
#endif
	ld	$31, 0($29)
#ifndef __mips_soft_float
	daddiu	$29, $29, 160
#else
	daddiu	$29, $29, 96
#endif
	jr	$31

.set at

#ifdef __ELF__
.section .note.GNU-stack,"",%progbits
#endif
