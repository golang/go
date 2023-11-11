// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build mips || mipsle

.file "gcc_mipsx.S"

/*
 * void crosscall1(void (*fn)(void), void (*setg_gcc)(void *g), void *g)
 *
 * Calling into the gc tool chain, where all registers are caller save.
 * Called from standard MIPS O32 ABI, where $16-$23, $30, and $f20-$f31
 * are callee-save, so they must be saved explicitly, along with $31 (LR).
 */
.globl crosscall1
.set noat
crosscall1:
#ifndef __mips_soft_float
	addiu	$29, $29, -88
#else
	addiu	$29, $29, -40 // For soft-float, no need to make room for FP registers
#endif
	sw	$31, 0($29)
	sw	$16, 4($29)
	sw	$17, 8($29)
	sw	$18, 12($29)
	sw	$19, 16($29)
	sw	$20, 20($29)
	sw	$21, 24($29)
	sw	$22, 28($29)
	sw	$23, 32($29)
	sw	$30, 36($29)

#ifndef __mips_soft_float
	sdc1	$f20, 40($29)
	sdc1	$f22, 48($29)
	sdc1	$f24, 56($29)
	sdc1	$f26, 64($29)
	sdc1	$f28, 72($29)
	sdc1	$f30, 80($29)
#endif
	move	$20, $4 // save R4
	move	$4, $6
	jalr	$5	// call setg_gcc
	jalr	$20	// call fn

	lw	$16, 4($29)
	lw	$17, 8($29)
	lw	$18, 12($29)
	lw	$19, 16($29)
	lw	$20, 20($29)
	lw	$21, 24($29)
	lw	$22, 28($29)
	lw	$23, 32($29)
	lw	$30, 36($29)
#ifndef __mips_soft_float
	ldc1	$f20, 40($29)
	ldc1	$f22, 48($29)
	ldc1	$f24, 56($29)
	ldc1	$f26, 64($29)
	ldc1	$f28, 72($29)
	ldc1	$f30, 80($29)
#endif
	lw	$31, 0($29)
#ifndef __mips_soft_float
	addiu	$29, $29, 88
#else
	addiu	$29, $29, 40
#endif
	jr	$31

.set at

#ifdef __ELF__
.section .note.GNU-stack,"",%progbits
#endif
