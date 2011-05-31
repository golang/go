// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "386/asm.h"

// void *stdcall_raw(void *fn, int32 count, uintptr *args)
TEXT runtime·stdcall_raw(SB),7,$0
	// Copy arguments from stack.
	MOVL	fn+0(FP), AX
	MOVL	count+4(FP), CX		// words
	MOVL	args+8(FP), BP

	// Switch to m->g0 if needed.
	get_tls(DI)
	MOVL	m(DI), DX
	MOVL	0(FS), SI
	MOVL	SI, m_sehframe(DX)
	MOVL	m_g0(DX), SI
	CMPL	g(DI), SI
	MOVL	SP, BX
	JEQ	2(PC)
	MOVL	(g_sched+gobuf_sp)(SI), SP
	PUSHL	BX
	PUSHL	g(DI)
	MOVL	SI, g(DI)

	// Copy args to new stack.
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
	POPL	g(DI)
	POPL	SP

	// Someday the convention will be D is always cleared.
	CLD

	RET

// faster get/set last error
TEXT runtime·getlasterror(SB),7,$0
	MOVL	0x34(FS), AX
	RET

TEXT runtime·setlasterror(SB),7,$0
	MOVL	err+0(FP), AX
	MOVL	AX, 0x34(FS)
	RET

TEXT runtime·sigtramp(SB),7,$0
	PUSHL	BP			// cdecl
	PUSHL	BX
	PUSHL	SI
	PUSHL	DI
	PUSHL	0(FS)
	CALL	runtime·sigtramp1(SB)
	POPL	0(FS)
	POPL	DI
	POPL	SI
	POPL	BX
	POPL	BP
	RET

TEXT runtime·sigtramp1(SB),0,$16-40
	// unwinding?
	MOVL	info+24(FP), BX
	MOVL	4(BX), CX		// exception flags
	ANDL	$6, CX
	MOVL	$1, AX
	JNZ	sigdone

	// place ourselves at the top of the SEH chain to
	// ensure SEH frames lie within thread stack bounds
	MOVL	frame+28(FP), CX	// our SEH frame
	MOVL	CX, 0(FS)

	// copy arguments for call to sighandler
	MOVL	BX, 0(SP)
	MOVL	CX, 4(SP)
	MOVL	context+32(FP), BX
	MOVL	BX, 8(SP)
	MOVL	dispatcher+36(FP), BX
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

// Windows runs the ctrl handler in a new thread.
TEXT runtime·ctrlhandler(SB),7,$0
	PUSHL	BP
	MOVL	SP, BP
	PUSHL	BX
	PUSHL	SI
	PUSHL	DI
	PUSHL	0x2c(FS)
	MOVL	SP, BX

	// setup dummy m, g
	SUBL	$(m_sehframe+4), SP	// at least space for m_sehframe
	LEAL	m_tls(SP), CX
	MOVL	CX, 0x2c(FS)
	MOVL	SP, m(CX)
	MOVL	SP, DX
	SUBL	$8, SP			// space for g_stack{guard,base}
	MOVL	SP, g(CX)
	MOVL	SP, m_g0(DX)
	LEAL	-4096(SP), CX
	MOVL	CX, g_stackguard(SP)
	MOVL	BX, g_stackbase(SP)

	PUSHL	8(BP)
	CALL	runtime·ctrlhandler1(SB)
	POPL	CX

	get_tls(CX)
	MOVL	g(CX), CX
	MOVL	g_stackbase(CX), SP
	POPL	0x2c(FS)
	POPL	DI
	POPL	SI
	POPL	BX
	POPL	BP
	MOVL	0(SP), CX
	ADDL	$8, SP
	JMP	CX

// Called from dynamic function created by ../thread.c compilecallback,
// running on Windows stack (not Go stack).
// BX, BP, SI, DI registers and DF flag are preserved
// as required by windows callback convention.
// AX = address of go func we need to call
// DX = total size of arguments
//
TEXT runtime·callbackasm+0(SB),7,$0
	// preserve whatever's at the memory location that
	// the callback will use to store the return value
	LEAL	8(SP), CX
	PUSHL	0(CX)(DX*1)
	ADDL	$4, DX			// extend argsize by size of return value

	// save registers as required for windows callback
	PUSHL	0(FS)
	PUSHL	DI
	PUSHL	SI
	PUSHL	BP
	PUSHL	BX
	PUSHL	DX
	PUSHL	CX
	PUSHL	AX

	// reinstall our SEH handler
	get_tls(CX)
	MOVL	m(CX), CX
	MOVL	m_sehframe(CX), CX
	MOVL	CX, 0(FS)
	CLD

	CALL	runtime·cgocallback(SB)

	// restore registers as required for windows callback
	POPL	AX
	POPL	CX
	POPL	DX
	POPL	BX
	POPL	BP
	POPL	SI
	POPL	DI
	POPL	0(FS)
	CLD

	MOVL	-4(CX)(DX*1), AX
	POPL	-4(CX)(DX*1)
	RET

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
	MOVL	AX, g_stackbase(DX)
	SUBL	$(64*1024), AX		// stack size
	MOVL	AX, g_stackguard(DX)

	// Set up tls.
	LEAL	m_tls(CX), SI
	MOVL	SI, 0x2c(FS)
	MOVL	CX, m(SI)
	MOVL	DX, g(SI)

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
