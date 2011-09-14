// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "386/asm.h"

// void runtime·asmstdcall(void *c);
TEXT runtime·asmstdcall(SB),7,$0
	MOVL	c+0(FP), BX

	// SetLastError(0).
	MOVL	$0, 0x34(FS)

	// Copy args to the stack.
	MOVL	SP, BP
	MOVL	wincall_n(BX), CX	// words
	MOVL	CX, AX
	SALL	$2, AX
	SUBL	AX, SP			// room for args
	MOVL	SP, DI
	MOVL	wincall_args(BX), SI
	CLD
	REP; MOVSL

	// Call stdcall or cdecl function.
	// DI SI BP BX are preserved, SP is not
	CALL	wincall_fn(BX)
	MOVL	BP, SP

	// Return result.
	MOVL	c+0(FP), BX
	MOVL	AX, wincall_r1(BX)
	MOVL	DX, wincall_r2(BX)

	// GetLastError().
	MOVL	0x34(FS), AX
	MOVL	AX, wincall_err(BX)

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
	SUBL	$(m_fflag+4), SP	// at least space for m_fflag
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
	PUSHL	DI
	PUSHL	SI
	PUSHL	BP
	PUSHL	BX

	// set up SEH frame again
	PUSHL	$runtime·sigtramp(SB)
	PUSHL	0(FS)
	MOVL	SP, 0(FS)

	// callback parameters
	PUSHL	DX
	PUSHL	CX
	PUSHL	AX

	CLD

	CALL	runtime·cgocallback(SB)

	POPL	AX
	POPL	CX
	POPL	DX

	// pop SEH frame
	POPL	0(FS)
	POPL	BX

	// restore registers as required for windows callback
	POPL	BX
	POPL	BP
	POPL	SI
	POPL	DI

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
