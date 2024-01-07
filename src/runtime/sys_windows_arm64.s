// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"
#include "funcdata.h"
#include "time_windows.h"
#include "cgo/abi_arm64.h"

// Offsets into Thread Environment Block (pointer in R18)
#define TEB_error 0x68
#define TEB_TlsSlots 0x1480
#define TEB_ArbitraryPtr 0x28

// Note: R0-R7 are args, R8 is indirect return value address,
// R9-R15 are caller-save, R19-R29 are callee-save.
//
// load_g and save_g (in tls_arm64.s) clobber R27 (REGTMP) and R0.

TEXT runtime·asmstdcall_trampoline<ABIInternal>(SB),NOSPLIT,$0
	B	runtime·asmstdcall(SB)

// void runtime·asmstdcall(void *c);
TEXT runtime·asmstdcall(SB),NOSPLIT,$16
	STP	(R19, R20), 16(RSP) // save old R19, R20
	MOVD	R0, R19	// save libcall pointer
	MOVD	RSP, R20	// save stack pointer

	// SetLastError(0)
	MOVD	$0,	TEB_error(R18_PLATFORM)
	MOVD	libcall_args(R19), R12	// libcall->args

	// Do we have more than 8 arguments?
	MOVD	libcall_n(R19), R0
	CMP	$0,	R0; BEQ	_0args
	CMP	$1,	R0; BEQ	_1args
	CMP	$2,	R0; BEQ	_2args
	CMP	$3,	R0; BEQ	_3args
	CMP	$4,	R0; BEQ	_4args
	CMP	$5,	R0; BEQ	_5args
	CMP	$6,	R0; BEQ	_6args
	CMP	$7,	R0; BEQ	_7args
	CMP	$8,	R0; BEQ	_8args

	// Reserve stack space for remaining args
	SUB	$8, R0, R2
	ADD	$1, R2, R3 // make even number of words for stack alignment
	AND	$~1, R3
	LSL	$3, R3
	SUB	R3, RSP

	// R4: size of stack arguments (n-8)*8
	// R5: &args[8]
	// R6: loop counter, from 0 to (n-8)*8
	// R7: scratch
	// R8: copy of RSP - (R2)(RSP) assembles as (R2)(ZR)
	SUB	$8, R0, R4
	LSL	$3, R4
	ADD	$(8*8), R12, R5
	MOVD	$0, R6
	MOVD	RSP, R8
stackargs:
	MOVD	(R6)(R5), R7
	MOVD	R7, (R6)(R8)
	ADD	$8, R6
	CMP	R6, R4
	BNE	stackargs

_8args:
	MOVD	(7*8)(R12), R7
_7args:
	MOVD	(6*8)(R12), R6
_6args:
	MOVD	(5*8)(R12), R5
_5args:
	MOVD	(4*8)(R12), R4
_4args:
	MOVD	(3*8)(R12), R3
_3args:
	MOVD	(2*8)(R12), R2
_2args:
	MOVD	(1*8)(R12), R1
_1args:
	MOVD	(0*8)(R12), R0
_0args:

	MOVD	libcall_fn(R19), R12	// branch to libcall->fn
	BL	(R12)

	MOVD	R20, RSP			// free stack space
	MOVD	R0, libcall_r1(R19)		// save return value to libcall->r1
	// TODO(rsc) floating point like amd64 in libcall->r2?

	// GetLastError
	MOVD	TEB_error(R18_PLATFORM), R0
	MOVD	R0, libcall_err(R19)

	// Restore callee-saved registers.
	LDP	16(RSP), (R19, R20)
	RET

TEXT runtime·getlasterror(SB),NOSPLIT,$0
	MOVD	TEB_error(R18_PLATFORM), R0
	MOVD	R0, ret+0(FP)
	RET

// Called by Windows as a Vectored Exception Handler (VEH).
// R0 is pointer to struct containing
// exception record and context pointers.
// R1 is the kind of sigtramp function.
// Return value of sigtrampgo is stored in R0.
TEXT sigtramp<>(SB),NOSPLIT,$176
	// Switch from the host ABI to the Go ABI, safe args and lr.
	MOVD	R0, R5
	MOVD	R1, R6
	MOVD	LR, R7
	SAVE_R19_TO_R28(8*4)
	SAVE_F8_TO_F15(8*14)

	BL	runtime·load_g(SB)	// Clobers R0, R27, R28 (g)

	MOVD	R5, R0
	MOVD	R6, R1
	// Calling ABIInternal because TLS might be nil.
	BL	runtime·sigtrampgo<ABIInternal>(SB)
	// Return value is already stored in R0.

	// Restore callee-save registers.
	RESTORE_R19_TO_R28(8*4)
	RESTORE_F8_TO_F15(8*14)
	MOVD	R7, LR
	RET

// Trampoline to resume execution from exception handler.
// This is part of the control flow guard workaround.
// It switches stacks and jumps to the continuation address.
// R0 and R1 are set above at the end of sigtrampgo
// in the context that starts executing at sigresume.
TEXT runtime·sigresume(SB),NOSPLIT|NOFRAME,$0
	// Important: do not smash LR,
	// which is set to a live value when handling
	// a signal by pushing a call to sigpanic onto the stack.
	MOVD	R0, RSP
	B	(R1)

TEXT runtime·exceptiontramp(SB),NOSPLIT|NOFRAME,$0
	MOVD	$const_callbackVEH, R1
	B	sigtramp<>(SB)

TEXT runtime·firstcontinuetramp(SB),NOSPLIT|NOFRAME,$0
	MOVD	$const_callbackFirstVCH, R1
	B	sigtramp<>(SB)

TEXT runtime·lastcontinuetramp(SB),NOSPLIT|NOFRAME,$0
	MOVD	$const_callbackLastVCH, R1
	B	sigtramp<>(SB)

TEXT runtime·callbackasm1(SB),NOSPLIT,$208-0
	NO_LOCAL_POINTERS

	// On entry, the trampoline in zcallback_windows_arm64.s left
	// the callback index in R12 (which is volatile in the C ABI).

	// Save callback register arguments R0-R7.
	// We do this at the top of the frame so they're contiguous with stack arguments.
	// The 7*8 setting up R14 looks like a bug but is not: the eighth word
	// is the space the assembler reserved for our caller's frame pointer,
	// but we are not called from Go so that space is ours to use,
	// and we must to be contiguous with the stack arguments.
	MOVD	$arg0-(7*8)(SP), R14
	STP	(R0, R1), (0*8)(R14)
	STP	(R2, R3), (2*8)(R14)
	STP	(R4, R5), (4*8)(R14)
	STP	(R6, R7), (6*8)(R14)

	// Push C callee-save registers R19-R28.
	// LR, FP already saved.
	SAVE_R19_TO_R28(8*9)

	// Create a struct callbackArgs on our stack.
	MOVD	$cbargs-(18*8+callbackArgs__size)(SP), R13
	MOVD	R12, callbackArgs_index(R13)	// callback index
	MOVD	R14, R0
	MOVD	R0, callbackArgs_args(R13)		// address of args vector
	MOVD	$0, R0
	MOVD	R0, callbackArgs_result(R13)	// result

	// Call cgocallback, which will call callbackWrap(frame).
	MOVD	$·callbackWrap<ABIInternal>(SB), R0	// PC of function to call, cgocallback takes an ABIInternal entry-point
	MOVD	R13, R1	// frame (&callbackArgs{...})
	MOVD	$0, R2	// context
	STP	(R0, R1), (1*8)(RSP)
	MOVD	R2, (3*8)(RSP)
	BL	runtime·cgocallback(SB)

	// Get callback result.
	MOVD	$cbargs-(18*8+callbackArgs__size)(SP), R13
	MOVD	callbackArgs_result(R13), R0

	RESTORE_R19_TO_R28(8*9)

	RET

// uint32 tstart_stdcall(M *newm);
TEXT runtime·tstart_stdcall(SB),NOSPLIT,$96-0
	SAVE_R19_TO_R28(8*3)

	MOVD	m_g0(R0), g
	MOVD	R0, g_m(g)
	BL	runtime·save_g(SB)

	// Set up stack guards for OS stack.
	MOVD	RSP, R0
	MOVD	R0, g_stack+stack_hi(g)
	SUB	$(64*1024), R0
	MOVD	R0, (g_stack+stack_lo)(g)
	MOVD	R0, g_stackguard0(g)
	MOVD	R0, g_stackguard1(g)

	BL	runtime·emptyfunc(SB)	// fault if stack check is wrong
	BL	runtime·mstart(SB)

	RESTORE_R19_TO_R28(8*3)

	// Exit the thread.
	MOVD	$0, R0
	RET

TEXT runtime·nanotime1(SB),NOSPLIT,$0-8
	MOVD	$_INTERRUPT_TIME, R3
	MOVD	time_lo(R3), R0
	MOVD	$100, R1
	MUL	R1, R0
	MOVD	R0, ret+0(FP)
	RET

// This is called from rt0_go, which runs on the system stack
// using the initial stack allocated by the OS.
// It calls back into standard C using the BL below.
TEXT runtime·wintls(SB),NOSPLIT,$0
	// Allocate a TLS slot to hold g across calls to external code
	MOVD	runtime·_TlsAlloc(SB), R0
	SUB	$16, RSP	// skip over saved frame pointer below RSP
	BL	(R0)
	ADD	$16, RSP

	// Assert that slot is less than 64 so we can use _TEB->TlsSlots
	CMP	$64, R0
	BLT	ok
	// Fallback to the TEB arbitrary pointer.
	// TODO: don't use the arbitrary pointer (see go.dev/issue/59824)
	MOVD	$TEB_ArbitraryPtr, R0
	B	settls
ok:

	// Save offset from R18 into tls_g.
	LSL	$3, R0
	ADD	$TEB_TlsSlots, R0
settls:
	MOVD	R0, runtime·tls_g(SB)
	RET
