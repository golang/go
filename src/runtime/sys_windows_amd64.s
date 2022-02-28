// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"
#include "time_windows.h"
#include "cgo/abi_amd64.h"

// void runtime·asmstdcall(void *c);
TEXT runtime·asmstdcall(SB),NOSPLIT|NOFRAME,$0
	// asmcgocall will put first argument into CX.
	PUSHQ	CX			// save for later
	MOVQ	libcall_fn(CX), AX
	MOVQ	libcall_args(CX), SI
	MOVQ	libcall_n(CX), CX

	// SetLastError(0).
	MOVQ	0x30(GS), DI
	MOVL	$0, 0x68(DI)

	SUBQ	$(const_maxArgs*8), SP	// room for args

	// Fast version, do not store args on the stack.
	CMPL	CX, $4
	JLE	loadregs

	// Check we have enough room for args.
	CMPL	CX, $const_maxArgs
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
	// Floating point arguments are passed in the XMM
	// registers. Set them here in case any of the arguments
	// are floating point values. For details see
	//	https://msdn.microsoft.com/en-us/library/zthk2dkh.aspx
	MOVQ	CX, X0
	MOVQ	DX, X1
	MOVQ	R8, X2
	MOVQ	R9, X3

	// Call stdcall function.
	CALL	AX

	ADDQ	$(const_maxArgs*8), SP

	// Return result.
	POPQ	CX
	MOVQ	AX, libcall_r1(CX)
	// Floating point return values are returned in XMM0. Setting r2 to this
	// value in case this call returned a floating point value. For details,
	// see https://docs.microsoft.com/en-us/cpp/build/x64-calling-convention
	MOVQ    X0, libcall_r2(CX)

	// GetLastError().
	MOVQ	0x30(GS), DI
	MOVL	0x68(DI), AX
	MOVQ	AX, libcall_err(CX)

	RET

TEXT runtime·badsignal2(SB),NOSPLIT|NOFRAME,$48
	// stderr
	MOVQ	$-12, CX // stderr
	MOVQ	CX, 0(SP)
	MOVQ	runtime·_GetStdHandle(SB), AX
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
	MOVQ	runtime·_WriteFile(SB), AX
	CALL	AX

	// Does not return.
	CALL	runtime·abort(SB)
	RET

// faster get/set last error
TEXT runtime·getlasterror(SB),NOSPLIT,$0
	MOVQ	0x30(GS), AX
	MOVL	0x68(AX), AX
	MOVL	AX, ret+0(FP)
	RET

// Called by Windows as a Vectored Exception Handler (VEH).
// First argument is pointer to struct containing
// exception record and context pointers.
// Handler function is stored in AX.
// Return 0 for 'not handled', -1 for handled.
TEXT sigtramp<>(SB),NOSPLIT|NOFRAME,$0-0
	// CX: PEXCEPTION_POINTERS ExceptionInfo

	// Switch from the host ABI to the Go ABI.
	PUSH_REGS_HOST_TO_ABI0()
	// Make stack space for the rest of the function.
	ADJSP	$48

	MOVQ	AX, R15	// save handler address

	// find g
	get_tls(DX)
	CMPQ	DX, $0
	JNE	3(PC)
	MOVQ	$0, AX // continue
	JMP	done
	MOVQ	g(DX), DX
	CMPQ	DX, $0
	JNE	2(PC)
	CALL	runtime·badsignal2(SB)

	// save g and SP in case of stack switch
	MOVQ	DX, 32(SP) // g
	MOVQ	SP, 40(SP)

	// do we need to switch to the g0 stack?
	MOVQ	g_m(DX), BX
	MOVQ	m_g0(BX), BX
	CMPQ	DX, BX
	JEQ	g0

	// switch to g0 stack
	get_tls(BP)
	MOVQ	BX, g(BP)
	MOVQ	(g_sched+gobuf_sp)(BX), DI
	// make room for sighandler arguments
	// and re-save old SP for restoring later.
	// Adjust g0 stack by the space we're using and
	// save SP at the same place on the g0 stack.
	// The 40(DI) here must match the 40(SP) above.
	SUBQ	$(REGS_HOST_TO_ABI0_STACK + 48), DI
	MOVQ	SP, 40(DI)
	MOVQ	DI, SP

g0:
	MOVQ	0(CX), BX // ExceptionRecord*
	MOVQ	8(CX), CX // Context*
	MOVQ	BX, 0(SP)
	MOVQ	CX, 8(SP)
	MOVQ	DX, 16(SP)
	CALL	R15	// call handler
	// AX is set to report result back to Windows
	MOVL	24(SP), AX

	// switch back to original stack and g
	// no-op if we never left.
	MOVQ	40(SP), SP
	MOVQ	32(SP), DX
	get_tls(BP)
	MOVQ	DX, g(BP)

done:
	ADJSP	$-48
	POP_REGS_HOST_TO_ABI0()

	RET

TEXT runtime·exceptiontramp(SB),NOSPLIT|NOFRAME,$0
	MOVQ	$runtime·exceptionhandler(SB), AX
	JMP	sigtramp<>(SB)

TEXT runtime·firstcontinuetramp(SB),NOSPLIT|NOFRAME,$0-0
	MOVQ	$runtime·firstcontinuehandler(SB), AX
	JMP	sigtramp<>(SB)

TEXT runtime·lastcontinuetramp(SB),NOSPLIT|NOFRAME,$0-0
	MOVQ	$runtime·lastcontinuehandler(SB), AX
	JMP	sigtramp<>(SB)

GLOBL runtime·cbctxts(SB), NOPTR, $8

TEXT runtime·callbackasm1(SB),NOSPLIT,$0
	// Construct args vector for cgocallback().
	// By windows/amd64 calling convention first 4 args are in CX, DX, R8, R9
	// args from the 5th on are on the stack.
	// In any case, even if function has 0,1,2,3,4 args, there is reserved
	// but uninitialized "shadow space" for the first 4 args.
	// The values are in registers.
  	MOVQ	CX, (16+0)(SP)
  	MOVQ	DX, (16+8)(SP)
  	MOVQ	R8, (16+16)(SP)
  	MOVQ	R9, (16+24)(SP)
	// R8 = address of args vector
	LEAQ	(16+0)(SP), R8

	// remove return address from stack, we are not returning to callbackasm, but to its caller.
  	MOVQ	0(SP), AX
	ADDQ	$8, SP

	// determine index into runtime·cbs table
	MOVQ	$runtime·callbackasm(SB), DX
	SUBQ	DX, AX
	MOVQ	$0, DX
	MOVQ	$5, CX	// divide by 5 because each call instruction in runtime·callbacks is 5 bytes long
	DIVL	CX
	SUBQ	$1, AX	// subtract 1 because return PC is to the next slot

	// Switch from the host ABI to the Go ABI.
	PUSH_REGS_HOST_TO_ABI0()

	// Create a struct callbackArgs on our stack to be passed as
	// the "frame" to cgocallback and on to callbackWrap.
	SUBQ	$(24+callbackArgs__size), SP
	MOVQ	AX, (24+callbackArgs_index)(SP) 	// callback index
	MOVQ	R8, (24+callbackArgs_args)(SP)  	// address of args vector
	MOVQ	$0, (24+callbackArgs_result)(SP)	// result
	LEAQ	24(SP), AX
	// Call cgocallback, which will call callbackWrap(frame).
	MOVQ	$0, 16(SP)	// context
	MOVQ	AX, 8(SP)	// frame (address of callbackArgs)
	LEAQ	·callbackWrap<ABIInternal>(SB), BX	// cgocallback takes an ABIInternal entry-point
	MOVQ	BX, 0(SP)	// PC of function value to call (callbackWrap)
	CALL	·cgocallback(SB)
	// Get callback result.
	MOVQ	(24+callbackArgs_result)(SP), AX
	ADDQ	$(24+callbackArgs__size), SP

	POP_REGS_HOST_TO_ABI0()

	// The return value was placed in AX above.
	RET

// uint32 tstart_stdcall(M *newm);
TEXT runtime·tstart_stdcall(SB),NOSPLIT,$0
	// Switch from the host ABI to the Go ABI.
	PUSH_REGS_HOST_TO_ABI0()

	// CX contains first arg newm
	MOVQ	m_g0(CX), DX		// g

	// Layout new m scheduler stack on os stack.
	MOVQ	SP, AX
	MOVQ	AX, (g_stack+stack_hi)(DX)
	SUBQ	$(64*1024), AX		// initial stack size (adjusted later)
	MOVQ	AX, (g_stack+stack_lo)(DX)
	ADDQ	$const__StackGuard, AX
	MOVQ	AX, g_stackguard0(DX)
	MOVQ	AX, g_stackguard1(DX)

	// Set up tls.
	LEAQ	m_tls(CX), SI
	MOVQ	SI, 0x28(GS)
	MOVQ	CX, g_m(DX)
	MOVQ	DX, g(SI)

	CALL	runtime·stackcheck(SB)	// clobbers AX,CX
	CALL	runtime·mstart(SB)

	POP_REGS_HOST_TO_ABI0()

	XORL	AX, AX			// return 0 == success
	RET

// set tls base to DI
TEXT runtime·settls(SB),NOSPLIT,$0
	MOVQ	DI, 0x28(GS)
	RET

// Runs on OS stack.
// duration (in -100ns units) is in dt+0(FP).
// g may be nil.
// The function leaves room for 4 syscall parameters
// (as per windows amd64 calling convention).
TEXT runtime·usleep2(SB),NOSPLIT|NOFRAME,$48-4
	MOVLQSX	dt+0(FP), BX
	MOVQ	SP, AX
	ANDQ	$~15, SP	// alignment as per Windows requirement
	MOVQ	AX, 40(SP)
	LEAQ	32(SP), R8  // ptime
	MOVQ	BX, (R8)
	MOVQ	$-1, CX // handle
	MOVQ	$0, DX // alertable
	MOVQ	runtime·_NtWaitForSingleObject(SB), AX
	CALL	AX
	MOVQ	40(SP), SP
	RET

// Runs on OS stack. duration (in -100ns units) is in dt+0(FP).
// g is valid.
TEXT runtime·usleep2HighRes(SB),NOSPLIT|NOFRAME,$72-4
	MOVLQSX	dt+0(FP), BX
	get_tls(CX)

	MOVQ	SP, AX
	ANDQ	$~15, SP	// alignment as per Windows requirement
	MOVQ	AX, 64(SP)

	MOVQ	g(CX), CX
	MOVQ	g_m(CX), CX
	MOVQ	(m_mOS+mOS_highResTimer)(CX), CX	// hTimer
	MOVQ	CX, 48(SP)				// save hTimer for later
	LEAQ	56(SP), DX				// lpDueTime
	MOVQ	BX, (DX)
	MOVQ	$0, R8					// lPeriod
	MOVQ	$0, R9					// pfnCompletionRoutine
	MOVQ	$0, AX
	MOVQ	AX, 32(SP)				// lpArgToCompletionRoutine
	MOVQ	AX, 40(SP)				// fResume
	MOVQ	runtime·_SetWaitableTimer(SB), AX
	CALL	AX

	MOVQ	48(SP), CX				// handle
	MOVQ	$0, DX					// alertable
	MOVQ	$0, R8					// ptime
	MOVQ	runtime·_NtWaitForSingleObject(SB), AX
	CALL	AX

	MOVQ	64(SP), SP
	RET

// Runs on OS stack.
TEXT runtime·switchtothread(SB),NOSPLIT|NOFRAME,$0
	MOVQ	SP, AX
	ANDQ	$~15, SP	// alignment as per Windows requirement
	SUBQ	$(48), SP	// room for SP and 4 args as per Windows requirement
				// plus one extra word to keep stack 16 bytes aligned
	MOVQ	AX, 32(SP)
	MOVQ	runtime·_SwitchToThread(SB), AX
	CALL	AX
	MOVQ	32(SP), SP
	RET

TEXT runtime·nanotime1(SB),NOSPLIT,$0-8
	CMPB	runtime·useQPCTime(SB), $0
	JNE	useQPC
	MOVQ	$_INTERRUPT_TIME, DI
	MOVQ	time_lo(DI), AX
	IMULQ	$100, AX
	MOVQ	AX, ret+0(FP)
	RET
useQPC:
	JMP	runtime·nanotimeQPC(SB)
	RET

// func osSetupTLS(mp *m)
// Setup TLS. for use by needm on Windows.
TEXT runtime·osSetupTLS(SB),NOSPLIT,$0-8
	MOVQ	mp+0(FP), AX
	LEAQ	m_tls(AX), DI
	CALL	runtime·settls(SB)
	RET
