// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

/*
 * void crosscall2(void (*fn)(void*, int32), void*, int32)
 * Save registers and call fn with two arguments.
 */
TEXT crosscall2(SB),NOSPLIT,$0
#ifndef GOOS_windows
	SUBQ	$0x58, SP	/* keeps stack pointer 32-byte aligned */
#else
	SUBQ	$0xf8, SP	/* also need to save xmm6 - xmm15 */
#endif
	MOVQ	BX, 0x10(SP)
	MOVQ	BP, 0x18(SP)
	MOVQ	R12, 0x20(SP)
	MOVQ	R13, 0x28(SP)
	MOVQ	R14, 0x30(SP)
	MOVQ	R15, 0x38(SP)

#ifdef GOOS_windows
	// Win64 save RBX, RBP, RDI, RSI, RSP, R12, R13, R14, R15 and XMM6 -- XMM15.
	MOVQ	DI, 0x40(SP)
	MOVQ	SI, 0x48(SP)
	MOVUPS	X6, 0x50(SP)
	MOVUPS	X7, 0x60(SP)
	MOVUPS	X8, 0x70(SP)
	MOVUPS	X9, 0x80(SP)
	MOVUPS	X10, 0x90(SP)
	MOVUPS	X11, 0xa0(SP)
	MOVUPS	X12, 0xb0(SP)
	MOVUPS	X13, 0xc0(SP)
	MOVUPS	X14, 0xd0(SP)
	MOVUPS	X15, 0xe0(SP)

	MOVQ	DX, 0(SP)	/* arg */
	MOVQ	R8, 8(SP)	/* argsize (includes padding) */
	
	CALL	CX	/* fn */
	
	MOVQ	0x40(SP), DI
	MOVQ	0x48(SP), SI
	MOVUPS	0x50(SP), X6
	MOVUPS	0x60(SP), X7
	MOVUPS	0x70(SP), X8
	MOVUPS	0x80(SP), X9
	MOVUPS	0x90(SP), X10
	MOVUPS	0xa0(SP), X11
	MOVUPS	0xb0(SP), X12
	MOVUPS	0xc0(SP), X13
	MOVUPS	0xd0(SP), X14
	MOVUPS	0xe0(SP), X15
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
	
#ifndef GOOS_windows
	ADDQ	$0x58, SP
#else
	ADDQ	$0xf8, SP
#endif
	RET
