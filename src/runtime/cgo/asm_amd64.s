// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

/*
 * void crosscall2(void (*fn)(void*, int32), void*, int32)
 * Save registers and call fn with two arguments.
 */
TEXT crosscall2(SB),NOSPLIT,$0
	SUBQ	$0x58, SP	/* keeps stack pointer 32-byte aligned */
	MOVQ	BX, 0x10(SP)
	MOVQ	BP, 0x18(SP)
	MOVQ	R12, 0x20(SP)
	MOVQ	R13, 0x28(SP)
	MOVQ	R14, 0x30(SP)
	MOVQ	R15, 0x38(SP)

#ifdef GOOS_windows
	// Win64 save RBX, RBP, RDI, RSI, RSP, R12, R13, R14, and R15
	MOVQ	DI, 0x40(SP)
	MOVQ	SI, 0x48(SP)

	MOVQ	DX, 0(SP)	/* arg */
	MOVQ	R8, 8(SP)	/* argsize (includes padding) */
	
	CALL	CX	/* fn */
	
	MOVQ	0x40(SP), DI
	MOVQ	0x48(SP), SI
#else
	MOVQ	SI, 0(SP)	/* arg */
	MOVQ	DX, 8(SP)	/* argsize (includes padding) */

	CALL	DI	/* fn */
#endif

	MOVQ	0x10(SP), BX
	MOVQ	0x18(SP), BP
	MOVQ	0x20(SP), R12
	MOVQ	0x28(SP), R13
	MOVQ	0x30(SP), R14
	MOVQ	0x38(SP), R15
	
	ADDQ	$0x58, SP
	RET
