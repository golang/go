// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "amd64/asm.h"

#define maxargs 12

// void runtime·asmstdcall(void *c);
TEXT runtime·asmstdcall(SB),7,$0
	// asmcgocall will put first argument into CX.
	PUSHQ	CX			// save for later
	MOVQ	wincall_fn(CX), AX
	MOVQ	wincall_args(CX), SI
	MOVQ	wincall_n(CX), CX

	// SetLastError(0).
	MOVQ	0x30(GS), DI
	MOVL	$0, 0x68(DI)

	SUBQ	$(maxargs*8), SP	// room for args

	// Fast version, do not store args on the stack.
	CMPL	CX, $4
	JLE	loadregs

	// Check we have enough room for args.
	CMPL	CX, $maxargs
	JLE	2(PC)
	INT	$3			// not enough room -> crash

	// Copy args to the stack.
	MOVQ	SP, DI
	CLD
	REP; MOVSQ
	MOVQ	SP, SI

loadregs:
	// Load first 4 args into correspondent registers.
	MOVQ	0(SI), CX
	MOVQ	8(SI), DX
	MOVQ	16(SI), R8
	MOVQ	24(SI), R9

	// Call stdcall function.
	CALL	AX

	ADDQ	$(maxargs*8), SP

	// Return result.
	POPQ	CX
	MOVQ	AX, wincall_r(CX)

	// GetLastError().
	MOVQ	0x30(GS), DI
	MOVL	0x68(DI), AX
	MOVQ	AX, wincall_err(CX)

	RET

// faster get/set last error
TEXT runtime·getlasterror(SB),7,$0
	MOVQ	0x30(GS), AX
	MOVL	0x68(AX), AX
	RET

TEXT runtime·setlasterror(SB),7,$0
	MOVL	err+0(FP), AX
	MOVQ	0x30(GS),	CX
	MOVL	AX, 0x68(CX)
	RET

// Windows runs the ctrl handler in a new thread.
TEXT runtime·ctrlhandler(SB),7,$0
	PUSHQ	BP
	MOVQ	SP, BP
	PUSHQ	BX
	PUSHQ	SI
	PUSHQ	DI
	PUSHQ	0x58(GS)
	MOVQ	SP, BX

	// setup dummy m, g
	SUBQ	$(m_fflag+4), SP	// at least space for m_fflag
	LEAQ	m_tls(SP), CX
	MOVQ	CX, 0x58(GS)
	MOVQ	SP, m(CX)
	MOVQ	SP, DX
	SUBQ	$16, SP			// space for g_stack{guard,base}
	MOVQ	SP, g(CX)
	MOVQ	SP, m_g0(DX)
	LEAQ	-8192(SP), CX
	MOVQ	CX, g_stackguard(SP)
	MOVQ	BX, g_stackbase(SP)

	PUSHQ	16(BP)
	CALL	runtime·ctrlhandler1(SB)
	POPQ	CX

	get_tls(CX)
	MOVQ	g(CX), CX
	MOVQ	g_stackbase(CX), SP
	POPQ	0x58(GS)
	POPQ	DI
	POPQ	SI
	POPQ	BX
	POPQ	BP
	MOVQ	0(SP), CX
	ADDQ	$16, SP
	JMP	CX
	
TEXT runtime·callbackasm(SB),7,$0
	// TODO
	RET

// void tstart(M *newm);
TEXT runtime·tstart(SB),7,$0
	MOVQ	newm+8(SP), CX		// m
	MOVQ	m_g0(CX), DX		// g

	MOVQ	SP, DI			// remember stack

	// Layout new m scheduler stack on os stack.
	MOVQ	SP, AX
	MOVQ	AX, g_stackbase(DX)
	SUBQ	$(64*1024), AX	// stack size
	MOVQ	AX, g_stackguard(DX)

	// Set up tls.
	LEAQ	m_tls(CX), SI
	MOVQ	SI, 0x58(GS)
	MOVQ	CX, m(SI)
	MOVQ	DX, g(SI)

	// Someday the convention will be D is always cleared.
	CLD

	PUSHQ	DI			// original stack

	CALL	runtime·stackcheck(SB)		// clobbers AX,CX

	CALL	runtime·mstart(SB)

	POPQ	DI			// original stack
	MOVQ	DI, SP
	
	RET

// uint32 tstart_stdcall(M *newm);
TEXT runtime·tstart_stdcall(SB),7,$0
	MOVQ CX, BX // stdcall first arg in RCX

	PUSHQ	BX
	CALL	runtime·tstart+0(SB)
	POPQ	BX

	// Adjust stack for stdcall to return properly.
	MOVQ	(SP), AX		// save return address
	ADDQ	$8, SP			// remove single parameter
	MOVQ	AX, (SP)		// restore return address

	XORL	AX, AX			// return 0 == success

	RET

TEXT runtime·notok(SB),7,$0
	MOVQ	$0xf1, BP
	MOVQ	BP, (BP)
	RET

// set tls base to DI
TEXT runtime·settls(SB),7,$0
	MOVQ	DI, 0x58(GS)
	RET

