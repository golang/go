// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "386/asm.h"

// void *stdcall_raw(void *fn, int32 count, uintptr *args)
TEXT runtime·stdcall_raw(SB),7,$4
	// Copy arguments from stack.
	MOVL	fn+0(FP), AX
	MOVL	count+4(FP), CX		// words
	MOVL	args+8(FP), BP

	// Switch to m->g0 if needed.
	get_tls(DI)
	MOVL	m(DI), DX
	MOVL	g(DI), SI
	MOVL	SI, 0(SP)		// save g
	MOVL	SP, m_gostack(DX)	// save SP
	MOVL	m_g0(DX), SI
	CMPL	g(DI), SI
	JEQ 3(PC)
	MOVL	(m_sched+gobuf_sp)(DX), SP
	MOVL	SI, g(DI)

	// Copy args to new stack.
	SUBL	$(10*4), SP		// padding
	MOVL	CX, BX
	SALL	$2, BX
	SUBL	BX, SP			// room for args
	MOVL	SP, DI
	MOVL	BP, SI
	CLD
	REP; MOVSL

	// Call stdcall function.
	CALL	AX

	// Restore original SP, g.
	get_tls(DI)
	MOVL	m(DI), DX
	MOVL	m_gostack(DX), SP	// restore SP
	MOVL	0(SP), SI		// restore g
	MOVL	SI, g(DI)

	// Someday the convention will be D is always cleared.
	CLD

	RET 

// void tstart(M *newm);
TEXT runtime·tstart(SB),7,$0
	MOVL	newm+4(SP), CX		// m
	MOVL	m_g0(CX), DX		// g

	MOVL	SP, DI			// remember stack

	// Layout new m scheduler stack on os stack.
	MOVL	SP, AX
	SUBL	$256, AX		// just some space for ourselves
	MOVL	AX, g_stackbase(DX)
	SUBL	$8192, AX		// stack size
	MOVL	AX, g_stackguard(DX)

	// Set up tls.
	LEAL	m_tls(CX), SI
	MOVL	SI, 0x2c(FS)
	MOVL	CX, m(SI)
	MOVL	DX, g(SI)

	// Use scheduler stack now.
	MOVL	g_stackbase(DX), SP

	// Someday the convention will be D is always cleared.
	CLD

	PUSHL	DI			// original stack

	CALL	runtime·stackcheck(SB)		// clobbers AX,CX

	CALL	runtime·mstart(SB)

	POPL	DI			// original stack
	MOVL	DI, SP

	RET

// uint32 tstart_stdcall(M *newm);
TEXT runtime·tstart_stdcall(SB),7,$0
	MOVL	newm+4(SP), BX

	PUSHL	BX
	CALL	runtime·tstart(SB)
	POPL	BX

	// Adjust stack for stdcall to return properly.
	MOVL	(SP), AX		// save return address
	ADDL	$4, SP			// remove single parameter
	MOVL	AX, (SP)		// restore return address

	XORL	AX, AX			// return 0 == success

	RET

// setldt(int entry, int address, int limit)
TEXT runtime·setldt(SB),7,$0
	MOVL	address+4(FP), CX
	MOVL	CX, 0x2c(FS)
	RET

// for now, return 0,0.  only used for internal performance monitoring.
TEXT runtime·gettime(SB),7,$0
	MOVL	sec+0(FP), DI
	MOVL	$0, (DI)
	MOVL	$0, 4(DI)	// zero extend 32 -> 64 bits
	MOVL	usec+4(FP), DI
	MOVL	$0, (DI)
	RET
