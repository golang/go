// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "zasm_GOOS_GOARCH.h"

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

TEXT	runtime·badcallback(SB),7,$24
	// stderr
	MOVL	$-12, 0(SP)
	MOVL	SP, BP
	CALL	*runtime·GetStdHandle(SB)
	MOVL	BP, SP

	MOVL	AX, 0(SP)	// handle
	MOVL	$runtime·badcallbackmsg(SB), DX // pointer
	MOVL	DX, 4(SP)
	MOVL	runtime·badcallbacklen(SB), DX // count
	MOVL	DX, 8(SP)
	LEAL	20(SP), DX  // written count
	MOVL	$0, 0(DX)
	MOVL	DX, 12(SP)
	MOVL	$0, 16(SP) // overlapped
	CALL	*runtime·WriteFile(SB)
	MOVL	BP, SI
	RET

TEXT	runtime·badsignal(SB),7,$24
	// stderr
	MOVL	$-12, 0(SP)
	MOVL	SP, BP
	CALL	*runtime·GetStdHandle(SB)
	MOVL	BP, SP

	MOVL	AX, 0(SP)	// handle
	MOVL	$runtime·badsignalmsg(SB), DX // pointer
	MOVL	DX, 4(SP)
	MOVL	runtime·badsignallen(SB), DX // count
	MOVL	DX, 8(SP)
	LEAL	20(SP), DX  // written count
	MOVL	$0, 0(DX)
	MOVL	DX, 12(SP)
	MOVL	$0, 16(SP) // overlapped
	CALL	*runtime·WriteFile(SB)
	MOVL	BP, SI
	RET

// faster get/set last error
TEXT runtime·getlasterror(SB),7,$0
	MOVL	0x34(FS), AX
	RET

TEXT runtime·setlasterror(SB),7,$0
	MOVL	err+0(FP), AX
	MOVL	AX, 0x34(FS)
	RET

TEXT runtime·sigtramp(SB),7,$28
	// unwinding?
	MOVL	info+0(FP), CX
	TESTL	$6, 4(CX)		// exception flags
	MOVL	$1, AX
	JNZ	sigdone

	// copy arguments for call to sighandler
	MOVL	CX, 0(SP)
	MOVL	context+8(FP), CX
	MOVL	CX, 4(SP)

	get_tls(CX)

	// check that m exists
	MOVL	m(CX), AX
	CMPL	AX, $0
	JNE	2(PC)
	CALL	runtime·badsignal(SB)

	MOVL	g(CX), CX
	MOVL	CX, 8(SP)

	MOVL	BX, 12(SP)
	MOVL	BP, 16(SP)
	MOVL	SI, 20(SP)
	MOVL	DI, 24(SP)

	CALL	runtime·sighandler(SB)
	// AX is set to report result back to Windows

	MOVL	24(SP), DI
	MOVL	20(SP), SI
	MOVL	16(SP), BP
	MOVL	12(SP), BX
sigdone:
	RET

TEXT runtime·ctrlhandler(SB),7,$0
	PUSHL	$runtime·ctrlhandler1(SB)
	CALL	runtime·externalthreadhandler(SB)
	MOVL	4(SP), CX
	ADDL	$12, SP
	JMP	CX

TEXT runtime·profileloop(SB),7,$0
	PUSHL	$runtime·profileloop1(SB)
	CALL	runtime·externalthreadhandler(SB)
	MOVL	4(SP), CX
	ADDL	$12, SP
	JMP	CX

TEXT runtime·externalthreadhandler(SB),7,$0
	PUSHL	BP
	MOVL	SP, BP
	PUSHL	BX
	PUSHL	SI
	PUSHL	DI
	PUSHL	0x14(FS)
	MOVL	SP, DX

	// setup dummy m, g
	SUBL	$m_end, SP		// space for M
	MOVL	SP, 0(SP)
	MOVL	$m_end, 4(SP)
	CALL	runtime·memclr(SB)	// smashes AX,BX,CX

	LEAL	m_tls(SP), CX
	MOVL	CX, 0x14(FS)
	MOVL	SP, m(CX)
	MOVL	SP, BX
	SUBL	$g_end, SP		// space for G
	MOVL	SP, g(CX)
	MOVL	SP, m_g0(BX)

	MOVL	SP, 0(SP)
	MOVL	$g_end, 4(SP)
	CALL	runtime·memclr(SB)	// smashes AX,BX,CX
	LEAL	-4096(SP), CX
	MOVL	CX, g_stackguard(SP)
	MOVL	DX, g_stackbase(SP)

	PUSHL	16(BP)			// arg for handler
	CALL	8(BP)
	POPL	CX

	get_tls(CX)
	MOVL	g(CX), CX
	MOVL	g_stackbase(CX), SP
	POPL	0x14(FS)
	POPL	DI
	POPL	SI
	POPL	BX
	POPL	BP
	RET

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

	CALL	runtime·cgocallback_gofunc(SB)

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

	// Layout new m scheduler stack on os stack.
	MOVL	SP, AX
	MOVL	AX, g_stackbase(DX)
	SUBL	$(64*1024), AX		// stack size
	MOVL	AX, g_stackguard(DX)

	// Set up tls.
	LEAL	m_tls(CX), SI
	MOVL	SI, 0x14(FS)
	MOVL	CX, m(SI)
	MOVL	DX, g(SI)

	// Someday the convention will be D is always cleared.
	CLD

	CALL	runtime·stackcheck(SB)	// clobbers AX,CX
	CALL	runtime·mstart(SB)

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
	MOVL	CX, 0x14(FS)
	RET

// void install_exception_handler()
TEXT runtime·install_exception_handler(SB),7,$0
	get_tls(CX)
	MOVL	m(CX), CX		// m

	// Set up SEH frame
	MOVL	m_seh(CX), DX
	MOVL	$runtime·sigtramp(SB), AX
	MOVL	AX, seh_handler(DX)
	MOVL	0(FS), AX
	MOVL	AX, seh_prev(DX)

	// Install it
	MOVL	DX, 0(FS)

	RET

// void remove_exception_handler()
TEXT runtime·remove_exception_handler(SB),7,$0
	get_tls(CX)
	MOVL	m(CX), CX		// m

	// Remove SEH frame
	MOVL	m_seh(CX), DX
	MOVL	seh_prev(DX), AX
	MOVL	AX, 0(FS)

	RET

TEXT runtime·osyield(SB),7,$20
	// Tried NtYieldExecution but it doesn't yield hard enough.
	// NtWaitForSingleObject being used here as Sleep(0).
	MOVL	runtime·NtWaitForSingleObject(SB), AX
	MOVL	$-1, hi-4(SP)
	MOVL	$-1, lo-8(SP)
	LEAL	lo-8(SP), BX
	MOVL	BX, ptime-12(SP)
	MOVL	$0, alertable-16(SP)
	MOVL	$-1, handle-20(SP)
	MOVL	SP, BP
	CALL	checkstack4<>(SB)
	CALL	AX
	MOVL	BP, SP
	RET

TEXT runtime·usleep(SB),7,$20
	MOVL	runtime·NtWaitForSingleObject(SB), AX 
	// Have 1us units; need negative 100ns units.
	// Assume multiply by 10 will not overflow 32-bit word.
	MOVL	usec+0(FP), BX
	IMULL	$10, BX
	NEGL	BX
	MOVL	$-1, hi-4(SP)
	MOVL	BX, lo-8(SP)
	LEAL	lo-8(SP), BX
	MOVL	BX, ptime-12(SP)
	MOVL	$0, alertable-16(SP)
	MOVL	$-1, handle-20(SP)
	MOVL	SP, BP
	CALL	checkstack4<>(SB)
	CALL	AX
	MOVL	BP, SP
	RET

// This function requires 4 bytes of stack,
// to simulate what calling NtWaitForSingleObject will use.
// (It is just a CALL to the system call dispatch.)
// If the linker okays the call to checkstack4 (a NOSPLIT function)
// then the call to NtWaitForSingleObject is okay too.
TEXT checkstack4<>(SB),7,$4
	RET
