// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"
#include "time_windows.h"

TEXT ·StdCall<ABIInternal>(SB),NOSPLIT,$0
	B	·asmstdcall(SB)

TEXT ·asmstdcall(SB),NOSPLIT|NOFRAME,$0
	MOVM.DB.W [R4, R5, R14], (R13)	// push {r4, r5, lr}
	MOVW	R0, R4			// put fn * in r4
	MOVW	R13, R5			// save stack pointer in r5

	// SetLastError(0)
	MOVW	$0, R0
	MRC	15, 0, R1, C13, C0, 2
	MOVW	R0, 0x34(R1)

	MOVW	8(R4), R12	// fn->Args

	// Do we have more than 4 arguments?
	MOVW	4(R4), R0	// fn->n
	SUB.S	$4, R0, R2
	BLE	loadregs

	// Reserve stack space for remaining args
	SUB	R2<<2, R13
	BIC	$0x7, R13	// alignment for ABI

	// R0: count of arguments
	// R1:
	// R2: loop counter, from 0 to (n-4)
	// R3: scratch
	// R4: pointer to StdCallInfo struct
	// R12: fn->args
	MOVW	$0, R2
stackargs:
	ADD	$4, R2, R3		// r3 = args[4 + i]
	MOVW	R3<<2(R12), R3
	MOVW	R3, R2<<2(R13)		// stack[i] = r3

	ADD	$1, R2			// i++
	SUB	$4, R0, R3		// while (i < (n - 4))
	CMP	R3, R2
	BLT	stackargs

loadregs:
	CMP	$3, R0
	MOVW.GT 12(R12), R3

	CMP	$2, R0
	MOVW.GT 8(R12), R2

	CMP	$1, R0
	MOVW.GT 4(R12), R1

	CMP	$0, R0
	MOVW.GT 0(R12), R0

	BIC	$0x7, R13		// alignment for ABI
	MOVW	0(R4), R12		// branch to fn->fn
	BL	(R12)

	MOVW	R5, R13			// free stack space
	MOVW	R0, 12(R4)		// save return value to fn->r1
	MOVW	R1, 16(R4)

	// GetLastError
	MRC	15, 0, R1, C13, C0, 2
	MOVW	0x34(R1), R0
	MOVW	R0, 20(R4)		// store in fn->err

	MOVM.IA.W (R13), [R4, R5, R15]
