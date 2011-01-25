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

TEXT runtime·sigtramp(SB),7,$0
	PUSHL	BP			// cdecl
	PUSHL	0(FS)
	CALL	runtime·sigtramp1(SB)
	POPL	0(FS)
	POPL	BP
	RET

TEXT runtime·sigtramp1(SB),0,$16-28
	// unwinding?
	MOVL	info+12(FP), BX
	MOVL	4(BX), CX		// exception flags
	ANDL	$6, CX
	MOVL	$1, AX
	JNZ	sigdone

	// place ourselves at the top of the SEH chain to
	// ensure SEH frames lie within thread stack bounds
	MOVL	frame+16(FP), CX	// our SEH frame
	MOVL	CX, 0(FS)

	// copy arguments for call to sighandler
	MOVL	BX, 0(SP)
	MOVL	CX, 4(SP)
	MOVL	context+20(FP), BX
	MOVL	BX, 8(SP)
	MOVL	dispatcher+24(FP), BX
	MOVL	BX, 12(SP)

	CALL	runtime·sighandler(SB)
	TESTL	AX, AX
	JZ	sigdone

	// call windows default handler early
	MOVL	4(SP), BX		// our SEH frame
	MOVL	0(BX), BX		// SEH frame of default handler
	MOVL	BX, 4(SP)		// set establisher frame
	CALL	4(BX)

sigdone:
	RET

// Called from dynamic function created by ../thread.c compilecallback,
// running on Windows stack (not Go stack).
// Returns straight to DLL.
// EBX, EBP, ESI, EDI registers and DF flag are preserved
// as required by windows callback convention.
// On entry to the function the stack looks like:
//
// 0(SP)  - return address to callback
// 4(SP)  - address of go func we need to call
// 8(SP)  - total size of arguments
// 12(SP) - room to save BX register
// 16(SP) - room to save BP
// 20(SP) - room to save SI
// 24(SP) - room to save DI
// 28(SP) - return address to DLL
// 32(SP) - beginning of arguments
//
TEXT runtime·callbackasm+0(SB),7,$0
	MOVL	BX, 12(SP)		// save registers as required for windows callback
	MOVL	BP, 16(SP)
	MOVL	SI, 20(SP)
	MOVL	DI, 24(SP)

	LEAL	args+32(SP), AX
	MOVL	AX, 0(SP)

	CLD

	CALL	runtime·callback(SB)

	MOVL	12(SP), BX		// restore registers as required for windows callback
	MOVL	16(SP), BP
	MOVL	20(SP), SI
	MOVL	24(SP), DI
	CLD

	MOVL	ret+28(SP), CX
	MOVL	size+8(SP), DX
	ADDL	$32, DX
	ADDL	DX, SP
	JMP	CX

// void tstart(M *newm);
TEXT runtime·tstart(SB),7,$0
	MOVL	newm+4(SP), CX		// m
	MOVL	m_g0(CX), DX		// g

	// Set up SEH frame
	PUSHL	$runtime·sigtramp(SB)
	PUSHL	0(FS)
	MOVL	SP, 0(FS)

	// Layout new m scheduler stack on os stack.
	MOVL	SP, AX
	SUBL	$256, AX		// just some space for ourselves
	MOVL	AX, g_stackbase(DX)
	SUBL	$(64*1024), AX		// stack size
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

	CALL	runtime·stackcheck(SB)	// clobbers AX,CX

	CALL	runtime·mstart(SB)

	// Pop SEH frame
	MOVL	0(FS), SP
	POPL	0(FS)
	POPL	CX

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
	MOVL	$0, 4(DI)		// zero extend 32 -> 64 bits
	MOVL	usec+4(FP), DI
	MOVL	$0, (DI)
	RET
