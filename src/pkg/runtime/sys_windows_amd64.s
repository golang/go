// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "zasm_GOOS_GOARCH.h"

// maxargs should be divisible by 2, as Windows stack
// must be kept 16-byte aligned on syscall entry.
#define maxargs 16

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
	MOVQ	AX, wincall_r1(CX)

	// GetLastError().
	MOVQ	0x30(GS), DI
	MOVL	0x68(DI), AX
	MOVQ	AX, wincall_err(CX)

	RET

// This should be called on a system stack,
// so we don't need to concern about split stack.
TEXT runtime·badcallback(SB),7,$0
	SUBQ	$48, SP

	// stderr
	MOVQ	$-12, CX // stderr
	MOVQ	CX, 0(SP)
	MOVQ	runtime·GetStdHandle(SB), AX
	CALL	AX

	MOVQ	AX, CX	// handle
	MOVQ	CX, 0(SP)
	MOVQ	$runtime·badcallbackmsg(SB), DX // pointer
	MOVQ	DX, 8(SP)
	MOVL	$runtime·badcallbacklen(SB), R8 // count
	MOVQ	R8, 16(SP)
	LEAQ	40(SP), R9  // written count
	MOVQ	$0, 0(R9)
	MOVQ	R9, 24(SP)
	MOVQ	$0, 32(SP)	// overlapped
	MOVQ	runtime·WriteFile(SB), AX
	CALL	AX
	
	ADDQ	$48, SP
	RET

TEXT runtime·badsignal(SB),7,$48
	// stderr
	MOVQ	$-12, CX // stderr
	MOVQ	CX, 0(SP)
	MOVQ	runtime·GetStdHandle(SB), AX
	CALL	AX

	MOVQ	AX, CX	// handle
	MOVQ	CX, 0(SP)
	MOVQ	$runtime·badsignalmsg(SB), DX // pointer
	MOVQ	DX, 8(SP)
	MOVL	$runtime·badsignallen(SB), R8 // count
	MOVQ	R8, 16(SP)
	LEAQ	40(SP), R9  // written count
	MOVQ	$0, 0(R9)
	MOVQ	R9, 24(SP)
	MOVQ	$0, 32(SP)	// overlapped
	MOVQ	runtime·WriteFile(SB), AX
	CALL	AX
	
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

TEXT runtime·sigtramp(SB),7,$0
	// CX: exception record
	// R8: context

	// unwinding?
	TESTL	$6, 4(CX)		// exception flags
	MOVL	$1, AX
	JNZ	sigdone

	// copy arguments for call to sighandler.

	// Stack adjustment is here to hide from 6l,
	// which doesn't understand that sigtramp
	// runs on essentially unlimited stack.
	SUBQ	$56, SP
	MOVQ	CX, 0(SP)
	MOVQ	R8, 8(SP)

	get_tls(CX)

	// check that m exists
	MOVQ	m(CX), AX
	CMPQ	AX, $0
	JNE	2(PC)
	CALL	runtime·badsignal(SB)

	MOVQ	g(CX), CX
	MOVQ	CX, 16(SP)

	MOVQ	BX, 24(SP)
	MOVQ	BP, 32(SP)
	MOVQ	SI, 40(SP)
	MOVQ	DI, 48(SP)

	CALL	runtime·sighandler(SB)

	MOVQ	24(SP), BX
	MOVQ	32(SP), BP
	MOVQ	40(SP), SI
	MOVQ	48(SP), DI
	ADDQ	$56, SP

sigdone:
	RET

TEXT runtime·ctrlhandler(SB),7,$8
	MOVQ	CX, 16(SP)		// spill
	MOVQ	$runtime·ctrlhandler1(SB), CX
	MOVQ	CX, 0(SP)
	CALL	runtime·externalthreadhandler(SB)
	RET

TEXT runtime·profileloop(SB),7,$8
	MOVQ	$runtime·profileloop1(SB), CX
	MOVQ	CX, 0(SP)
	CALL	runtime·externalthreadhandler(SB)
	RET

TEXT runtime·externalthreadhandler(SB),7,$0
	PUSHQ	BP
	MOVQ	SP, BP
	PUSHQ	BX
	PUSHQ	SI
	PUSHQ	DI
	PUSHQ	0x28(GS)
	MOVQ	SP, DX

	// setup dummy m, g
	SUBQ	$m_end, SP		// space for M
	MOVQ	SP, 0(SP)
	MOVQ	$m_end, 8(SP)
	CALL	runtime·memclr(SB)	// smashes AX,BX,CX

	LEAQ	m_tls(SP), CX
	MOVQ	CX, 0x28(GS)
	MOVQ	SP, m(CX)
	MOVQ	SP, BX
	SUBQ	$g_end, SP		// space for G
	MOVQ	SP, g(CX)
	MOVQ	SP, m_g0(BX)

	MOVQ	SP, 0(SP)
	MOVQ	$g_end, 8(SP)
	CALL	runtime·memclr(SB)	// smashes AX,BX,CX
	LEAQ	-8192(SP), CX
	MOVQ	CX, g_stackguard(SP)
	MOVQ	DX, g_stackbase(SP)

	PUSHQ	32(BP)			// arg for handler
	CALL	16(BP)
	POPQ	CX

	get_tls(CX)
	MOVQ	g(CX), CX
	MOVQ	g_stackbase(CX), SP
	POPQ	0x28(GS)
	POPQ	DI
	POPQ	SI
	POPQ	BX
	POPQ	BP
	RET

// Continuation of thunk function created for each callback by ../thread.c compilecallback,
// runs on Windows stack (not Go stack).
// Thunk code designed to have minimal size for it is copied many (up to thousands) times.
//
// thunk:
//	MOVQ	$fn, AX
//	PUSHQ	AX
//	MOVQ	$argsize, AX
//	PUSHQ	AX
//	MOVQ	$runtime·callbackasm, AX
//	JMP	AX
TEXT runtime·callbackasm(SB),7,$0
	// Construct args vector for cgocallback().
	// By windows/amd64 calling convention first 4 args are in CX, DX, R8, R9
	// args from the 5th on are on the stack.
	// In any case, even if function has 0,1,2,3,4 args, there is reserved
	// but uninitialized "shadow space" for the first 4 args.
	// The values are in registers.
  	MOVQ	CX, (24+0)(SP)
  	MOVQ	DX, (24+8)(SP)
  	MOVQ	R8, (24+16)(SP)
  	MOVQ	R9, (24+24)(SP)
	// 6l does not accept writing POPQs here issuing a warning "unbalanced PUSH/POP"
  	MOVQ	0(SP), DX	// POPQ DX
  	MOVQ	8(SP), AX	// POPQ AX
	ADDQ	$16, SP

	// preserve whatever's at the memory location that
	// the callback will use to store the return value
	LEAQ	8(SP), CX       // args vector, skip return address
	PUSHQ	0(CX)(DX*1)     // store 8 bytes from just after the args array
	ADDQ	$8, DX          // extend argsize by size of return value

	// DI SI BP BX R12 R13 R14 R15 registers and DF flag are preserved
	// as required by windows callback convention.
	// 6l does not allow writing many PUSHQs here issuing a warning "nosplit stack overflow"
	// the warning has no sense as this code uses os thread stack
	PUSHFQ
	SUBQ	$64, SP
	MOVQ	DI, 56(SP)
	MOVQ	SI, 48(SP)
	MOVQ	BP, 40(SP)
	MOVQ	BX, 32(SP)
	MOVQ	R12, 24(SP)
	MOVQ	R13, 16(SP)
	MOVQ	R14, 8(SP)
	MOVQ	R15, 0(SP)

	// prepare call stack.  use SUBQ to hide from stack frame checks
	// cgocallback(Go func, void *frame, uintptr framesize)
	SUBQ	$24, SP
	MOVQ	DX, 16(SP)	// uintptr framesize
	MOVQ	CX, 8(SP)   // void *frame
	MOVQ	AX, 0(SP)    // Go func
	CLD
	CALL  runtime·cgocallback_gofunc(SB)
	MOVQ	0(SP), AX
	MOVQ	8(SP), CX
	MOVQ	16(SP), DX
	ADDQ	$24, SP

	// restore registers as required for windows callback
	// 6l does not allow writing many POPs here issuing a warning "nosplit stack overflow"
	MOVQ	0(SP), R15
	MOVQ	8(SP), R14
	MOVQ	16(SP), R13
	MOVQ	24(SP), R12
	MOVQ	32(SP), BX
	MOVQ	40(SP), BP
	MOVQ	48(SP), SI
	MOVQ	56(SP), DI
	ADDQ	$64, SP
	POPFQ

	MOVL	-8(CX)(DX*1), AX  // return value
	POPQ	-8(CX)(DX*1)      // restore bytes just after the args
	RET

TEXT runtime·setstacklimits(SB),7,$0
	MOVQ	0x30(GS), CX
	MOVQ	$0, 0x10(CX)
	MOVQ	$0xffffffffffff, AX
	MOVQ	AX, 0x08(CX)
	RET

// uint32 tstart_stdcall(M *newm);
TEXT runtime·tstart_stdcall(SB),7,$0
	// CX contains first arg newm
	MOVQ	m_g0(CX), DX		// g

	// Layout new m scheduler stack on os stack.
	MOVQ	SP, AX
	MOVQ	AX, g_stackbase(DX)
	SUBQ	$(64*1024), AX		// stack size
	MOVQ	AX, g_stackguard(DX)

	// Set up tls.
	LEAQ	m_tls(CX), SI
	MOVQ	SI, 0x28(GS)
	MOVQ	CX, m(SI)
	MOVQ	DX, g(SI)

	// Someday the convention will be D is always cleared.
	CLD

	CALL	runtime·stackcheck(SB)	// clobbers AX,CX
	CALL	runtime·mstart(SB)

	XORL	AX, AX			// return 0 == success
	RET

// set tls base to DI
TEXT runtime·settls(SB),7,$0
	MOVQ	DI, 0x28(GS)
	RET

// void install_exception_handler()
TEXT runtime·install_exception_handler(SB),7,$0
	CALL	runtime·setstacklimits(SB)
	RET

TEXT runtime·remove_exception_handler(SB),7,$0
	RET

TEXT runtime·osyield(SB),7,$8
	// Tried NtYieldExecution but it doesn't yield hard enough.
	// NtWaitForSingleObject being used here as Sleep(0).
	// The CALL is safe because NtXxx is a system call wrapper:
	// it puts the right system call number in AX, then does
	// a SYSENTER and a RET.
	MOVQ	runtime·NtWaitForSingleObject(SB), AX
	MOVQ	$1, BX
	NEGQ	BX
	MOVQ	SP, R8 // ptime
	MOVQ	BX, (R8)
	MOVQ	$-1, CX // handle
	MOVQ	$0, DX // alertable
	CALL	AX
	RET

TEXT runtime·usleep(SB),7,$8
	// The CALL is safe because NtXxx is a system call wrapper:
	// it puts the right system call number in AX, then does
	// a SYSENTER and a RET.
	MOVQ	runtime·NtWaitForSingleObject(SB), AX
	// Have 1us units; want negative 100ns units.
	MOVL	usec+0(FP), BX
	IMULQ	$10, BX
	NEGQ	BX
	MOVQ	SP, R8 // ptime
	MOVQ	BX, (R8)
	MOVQ	$-1, CX // handle
	MOVQ	$0, DX // alertable
	CALL	AX
	RET
