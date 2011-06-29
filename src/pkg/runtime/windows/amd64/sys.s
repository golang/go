// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "amd64/asm.h"

// void *stdcall_raw(void *fn, uintptr nargs, void *args)
TEXT runtime·stdcall_raw(SB),7,$8
	MOVQ	fn+0(FP), AX
	MOVQ	nargs+8(FP), CX
	MOVQ	args+16(FP), R11

	// Switch to m->g0 if needed.
	get_tls(DI)
	MOVQ	m(DI), DX
	MOVQ	g(DI), SI
	MOVQ	SI, 0(SP)		// save g
	MOVQ	SP, m_gostack(DX)	// save SP
	MOVQ	m_g0(DX), SI
	CMPQ	g(DI), SI
	JEQ 3(PC)
	MOVQ	(g_sched+gobuf_sp)(SI), SP
	MOVQ	SI, g(DI)
	
	SUBQ	$0x60, SP
	
	// Copy args to new stack.
	MOVQ	SP, DI
	MOVQ	R11, SI
	CLD
	REP; MOVSQ
	MOVQ	0(R11), CX
	MOVQ	8(R11), DX
	MOVQ	16(R11), R8
	MOVQ	24(R11), R9

	// Call stdcall function.
	CALL	AX
	
	// Restore original SP, g.
	get_tls(DI)
	MOVQ	m(DI), DX
	MOVQ	m_gostack(DX), SP	// restore SP
	MOVQ	0(SP), SI		// restore g
	MOVQ	SI, g(DI)

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
	// TODO
	RET
	
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

