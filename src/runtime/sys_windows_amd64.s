// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"
#include "time_windows.h"
#include "cgo/abi_amd64.h"

// Offsets into Thread Environment Block (pointer in GS)
#define TEB_TlsSlots 0x1480
#define TEB_ArbitraryPtr 0x28

TEXT runtime·asmstdcall_trampoline<ABIInternal>(SB),NOSPLIT,$0
	MOVQ	AX, CX
	JMP	runtime·asmstdcall(SB)

// void runtime·asmstdcall(void *c);
TEXT runtime·asmstdcall(SB),NOSPLIT,$16
	MOVQ	SP, AX
	ANDQ	$~15, SP	// alignment as per Windows requirement
	MOVQ	AX, 8(SP)
	MOVQ	CX, 0(SP)	// asmcgocall will put first argument into CX.

	MOVQ	libcall_fn(CX), AX
	MOVQ	libcall_args(CX), SI
	MOVQ	libcall_n(CX), CX

	// SetLastError(0).
	MOVQ	0x30(GS), DI
	MOVL	$0, 0x68(DI)

	SUBQ	$(const_maxArgs*8), SP	// room for args

	// Fast version, do not store args on the stack nor
	// load them into registers.
	CMPL	CX, $0
	JE	docall

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
	//	https://learn.microsoft.com/en-us/cpp/build/x64-calling-convention?view=msvc-170
	MOVQ	CX, X0
	MOVQ	DX, X1
	MOVQ	R8, X2
	MOVQ	R9, X3

docall:
	// Call stdcall function.
	CALL	AX

	ADDQ	$(const_maxArgs*8), SP

	// Return result.
	MOVQ	0(SP), CX
	MOVQ	8(SP), SP
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

// faster get/set last error
TEXT runtime·getlasterror(SB),NOSPLIT,$0
	MOVQ	0x30(GS), AX
	MOVL	0x68(AX), AX
	MOVL	AX, ret+0(FP)
	RET

// Called by Windows as a Vectored Exception Handler (VEH).
// CX is pointer to struct containing
// exception record and context pointers.
// DX is the kind of sigtramp function.
// Return value of sigtrampgo is stored in AX.
TEXT sigtramp<>(SB),NOSPLIT,$0-0
	// Switch from the host ABI to the Go ABI.
	PUSH_REGS_HOST_TO_ABI0()

	// Set up ABIInternal environment: cleared X15 and R14.
	// R14 is cleared in case there's a non-zero value in there
	// if called from a non-go thread.
	XORPS	X15, X15
	XORQ	R14, R14

	get_tls(AX)
	CMPQ	AX, $0
	JE	2(PC)
	// Exception from Go thread, set R14.
	MOVQ	g(AX), R14

	// Reserve space for spill slots.
	ADJSP	$16
	MOVQ	CX, AX
	MOVQ	DX, BX
	// Calling ABIInternal because TLS might be nil.
	CALL	runtime·sigtrampgo<ABIInternal>(SB)
	// Return value is already stored in AX.

	ADJSP	$-16

	POP_REGS_HOST_TO_ABI0()
	RET

// Trampoline to resume execution from exception handler.
// This is part of the control flow guard workaround.
// It switches stacks and jumps to the continuation address.
// R8 and R9 are set above at the end of sigtrampgo
// in the context that starts executing at sigresume.
TEXT runtime·sigresume(SB),NOSPLIT|NOFRAME,$0
	MOVQ	R8, SP
	JMP	R9

TEXT runtime·exceptiontramp(SB),NOSPLIT|NOFRAME,$0
	// PExceptionPointers already on CX
	MOVQ	$const_callbackVEH, DX
	JMP	sigtramp<>(SB)

TEXT runtime·firstcontinuetramp(SB),NOSPLIT|NOFRAME,$0-0
	// PExceptionPointers already on CX
	MOVQ	$const_callbackFirstVCH, DX
	JMP	sigtramp<>(SB)

TEXT runtime·lastcontinuetramp(SB),NOSPLIT|NOFRAME,$0-0
	// PExceptionPointers already on CX
	MOVQ	$const_callbackLastVCH, DX
	JMP	sigtramp<>(SB)

TEXT runtime·sehtramp(SB),NOSPLIT,$40-0
	// CX: PEXCEPTION_RECORD ExceptionRecord
	// DX: ULONG64 EstablisherFrame
	// R8: PCONTEXT ContextRecord
	// R9: PDISPATCHER_CONTEXT DispatcherContext
	// Switch from the host ABI to the Go ABI.
	PUSH_REGS_HOST_TO_ABI0()

	get_tls(AX)
	CMPQ	AX, $0
	JNE	2(PC)
	// This shouldn't happen, sehtramp is only attached to functions
	// called from Go, and exception handlers are only called from
	// the thread that threw the exception.
	INT	$3

	// Exception from Go thread, set R14.
	MOVQ	g(AX), R14

	ADJSP	$40
	MOVQ	CX, 0(SP)
	MOVQ	DX, 8(SP)
	MOVQ	R8, 16(SP)
	MOVQ	R9, 24(SP)
	CALL	runtime·sehhandler(SB)
	MOVL	32(SP), AX

	ADJSP	$-40

	POP_REGS_HOST_TO_ABI0()
	RET

TEXT runtime·callbackasm1(SB),NOSPLIT|NOFRAME,$0
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
TEXT runtime·tstart_stdcall(SB),NOSPLIT|NOFRAME,$0
	// Switch from the host ABI to the Go ABI.
	PUSH_REGS_HOST_TO_ABI0()

	// CX contains first arg newm
	MOVQ	m_g0(CX), DX		// g

	// Layout new m scheduler stack on os stack.
	MOVQ	SP, AX
	MOVQ	AX, (g_stack+stack_hi)(DX)
	SUBQ	$(64*1024), AX		// initial stack size (adjusted later)
	MOVQ	AX, (g_stack+stack_lo)(DX)
	ADDQ	$const_stackGuard, AX
	MOVQ	AX, g_stackguard0(DX)
	MOVQ	AX, g_stackguard1(DX)

	// Set up tls.
	LEAQ	m_tls(CX), DI
	MOVQ	CX, g_m(DX)
	MOVQ	DX, g(DI)
	CALL	runtime·settls(SB) // clobbers CX

	CALL	runtime·stackcheck(SB)	// clobbers AX,CX
	CALL	runtime·mstart(SB)

	POP_REGS_HOST_TO_ABI0()

	XORL	AX, AX			// return 0 == success
	RET

// set tls base to DI
TEXT runtime·settls(SB),NOSPLIT,$0
	MOVQ	runtime·tls_g(SB), CX
	MOVQ	DI, 0(CX)(GS)
	RET

TEXT runtime·nanotime1(SB),NOSPLIT,$0-8
	MOVQ	$_INTERRUPT_TIME, DI
	MOVQ	time_lo(DI), AX
	IMULQ	$100, AX
	MOVQ	AX, ret+0(FP)
	RET

// func osSetupTLS(mp *m)
// Setup TLS. for use by needm on Windows.
TEXT runtime·osSetupTLS(SB),NOSPLIT,$0-8
	MOVQ	mp+0(FP), AX
	LEAQ	m_tls(AX), DI
	CALL	runtime·settls(SB)
	RET

// This is called from rt0_go, which runs on the system stack
// using the initial stack allocated by the OS.
TEXT runtime·wintls(SB),NOSPLIT,$0
	// Allocate a TLS slot to hold g across calls to external code
	MOVQ	SP, AX
	ANDQ	$~15, SP	// alignment as per Windows requirement
	SUBQ	$48, SP	// room for SP and 4 args as per Windows requirement
			// plus one extra word to keep stack 16 bytes aligned
	MOVQ	AX, 32(SP)
	MOVQ	runtime·_TlsAlloc(SB), AX
	CALL	AX
	MOVQ	32(SP), SP

	MOVQ	AX, CX	// TLS index

	// Assert that slot is less than 64 so we can use _TEB->TlsSlots
	CMPQ	CX, $64
	JB	ok

	// Fallback to the TEB arbitrary pointer.
	// TODO: don't use the arbitrary pointer (see go.dev/issue/59824)
	MOVQ	$TEB_ArbitraryPtr, CX
	JMP	settls
ok:
	// Convert the TLS index at CX into
	// an offset from TEB_TlsSlots.
	SHLQ	$3, CX

	// Save offset from TLS into tls_g.
	ADDQ	$TEB_TlsSlots, CX
settls:
	MOVQ	CX, runtime·tls_g(SB)
	RET
