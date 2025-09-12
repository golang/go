// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

TEXT ·StdCall<ABIInternal>(SB),NOSPLIT,$0
	MOVQ	AX, CX
	JMP	·asmstdcall(SB)

TEXT ·asmstdcall(SB),NOSPLIT,$16
	MOVQ	SP, AX
	ANDQ	$~15, SP	// alignment as per Windows requirement
	MOVQ	AX, 8(SP)
	MOVQ	CX, 0(SP)	// asmcgocall will put first argument into CX.

	MOVQ	StdCallInfo_Fn(CX), AX
	MOVQ	StdCallInfo_Args(CX), SI
	MOVQ	StdCallInfo_N(CX), CX

	// SetLastError(0).
	MOVQ	0x30(GS), DI
	MOVL	$0, 0x68(DI)

	SUBQ	$(const_MaxArgs*8), SP	// room for args

	// Fast version, do not store args on the stack.
	CMPL	CX, $0;	JE	_0args
	CMPL	CX, $1;	JE	_1args
	CMPL	CX, $2;	JE	_2args
	CMPL	CX, $3;	JE	_3args
	CMPL	CX, $4;	JE	_4args

	// Check we have enough room for args.
	CMPL	CX, $const_MaxArgs
	JLE	2(PC)
	INT	$3			// not enough room -> crash

	// Copy args to the stack.
	MOVQ	SP, DI
	CLD
	REP; MOVSQ
	MOVQ	SP, SI

	// Load first 4 args into correspondent registers.
	// Floating point arguments are passed in the XMM
	// registers. Set them here in case any of the arguments
	// are floating point values. For details see
	//	https://learn.microsoft.com/en-us/cpp/build/x64-calling-convention?view=msvc-170
_4args:
	MOVQ	24(SI), R9
	MOVQ	R9, X3
_3args:
	MOVQ	16(SI), R8
	MOVQ	R8, X2
_2args:
	MOVQ	8(SI), DX
	MOVQ	DX, X1
_1args:
	MOVQ	0(SI), CX
	MOVQ	CX, X0
_0args:

	// Call stdcall function.
	CALL	AX

	ADDQ	$(const_MaxArgs*8), SP

	// Return result.
	MOVQ	0(SP), CX
	MOVQ	8(SP), SP
	MOVQ	AX, StdCallInfo_R1(CX)
	// Floating point return values are returned in XMM0. Setting r2 to this
	// value in case this call returned a floating point value. For details,
	// see https://docs.microsoft.com/en-us/cpp/build/x64-calling-convention
	MOVQ    X0, StdCallInfo_R2(CX)

	// GetLastError().
	MOVQ	0x30(GS), DI
	MOVL	0x68(DI), AX
	MOVQ	AX, StdCallInfo_Err(CX)

	RET
