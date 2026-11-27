// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

TEXT ·StdCall<ABIInternal>(SB),NOSPLIT,$4-8
	MOVL	fn+0(FP), AX
	MOVL	AX, 0(SP)
	CALL	·asmstdcall(SB)
	MOVL	AX, ret+4(FP)
	RET

TEXT ·asmstdcall(SB),NOSPLIT,$0-4
	MOVL	fn+0(FP), BX
	MOVL	SP, BP	// save stack pointer

	// SetLastError(0).
	MOVL	$0, 0x34(FS)

	MOVL	StdCallInfo_N(BX), CX

	// Fast version, do not store args on the stack.
	CMPL	CX, $0
	JE	docall

	// Copy args to the stack.
	MOVL	CX, AX
	SALL	$2, AX
	SUBL	AX, SP			// room for args
	MOVL	SP, DI
	MOVL	StdCallInfo_Args(BX), SI
	CLD
	REP; MOVSL

docall:
	// Call stdcall or cdecl function.
	// DI SI BP BX are preserved, SP is not
	CALL	StdCallInfo_Fn(BX)
	MOVL	BP, SP

	// Return result.
	MOVL	fn+0(FP), BX
	MOVL	AX, StdCallInfo_R1(BX)
	MOVL	DX, StdCallInfo_R2(BX)

	// Floating point return values are returned in ST(0), the top of the
	// x87 stack, which is empty otherwise. Pop it as a float64 into R1 and
	// R2, low word first, so that the x87 stack is left empty.
	FXAM
	FSTSW	AX
	ANDL	$0x4500, AX	// C3, C2 and C0 classify ST(0)
	CMPL	AX, $0x4100	// empty
	JEQ	nofloat
	FMOVDP	F0, StdCallInfo_R1(BX)
nofloat:

	// GetLastError().
	MOVL	0x34(FS), AX

	RET
