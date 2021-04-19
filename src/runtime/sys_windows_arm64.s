// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"
#include "funcdata.h"

// Offsets into Thread Environment Block (pointer in R18)
#define TEB_error 0x68
#define TEB_TlsSlots 0x1480

// Note: R0-R7 are args, R8 is indirect return value address,
// R9-R15 are caller-save, R19-R29 are callee-save.
//
// load_g and save_g (in tls_arm64.s) clobber R27 (REGTMP) and R0.

// void runtime·asmstdcall(void *c);
TEXT runtime·asmstdcall<ABIInternal>(SB),NOSPLIT|NOFRAME,$0
	STP.W	(R29, R30), -32(RSP)	// allocate C ABI stack frame
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
	LDP.P	32(RSP), (R29, R30)
	RET

TEXT runtime·badsignal2(SB),NOSPLIT,$16-0
	NO_LOCAL_POINTERS

	// stderr
	MOVD	runtime·_GetStdHandle(SB), R1
	MOVD	$-12, R0
	SUB	$16, RSP	// skip over saved frame pointer below RSP
	BL	(R1)
	ADD	$16, RSP

	// handle in R0 already
	MOVD	$runtime·badsignalmsg(SB), R1	// lpBuffer
	MOVD	$runtime·badsignallen(SB), R2	// lpNumberOfBytesToWrite
	MOVD	(R2), R2
	MOVD	R13, R3		// lpNumberOfBytesWritten
	MOVD	$0, R4			// lpOverlapped
	MOVD	runtime·_WriteFile(SB), R12
	SUB	$16, RSP	// skip over saved frame pointer below RSP
	BL	(R12)
	ADD	$16, RSP

	RET

TEXT runtime·getlasterror(SB),NOSPLIT|NOFRAME,$0
	MOVD	TEB_error(R18_PLATFORM), R0
	MOVD	R0, ret+0(FP)
	RET

#define SAVE_R19_TO_R28(offset) \
	MOVD	R19, savedR19+((offset)+0*8)(SP); \
	MOVD	R20, savedR20+((offset)+1*8)(SP); \
	MOVD	R21, savedR21+((offset)+2*8)(SP); \
	MOVD	R22, savedR22+((offset)+3*8)(SP); \
	MOVD	R23, savedR23+((offset)+4*8)(SP); \
	MOVD	R24, savedR24+((offset)+5*8)(SP); \
	MOVD	R25, savedR25+((offset)+6*8)(SP); \
	MOVD	R26, savedR26+((offset)+7*8)(SP); \
	MOVD	R27, savedR27+((offset)+8*8)(SP); \
	MOVD	g, savedR28+((offset)+9*8)(SP);

#define RESTORE_R19_TO_R28(offset) \
	MOVD	savedR19+((offset)+0*8)(SP), R19; \
	MOVD	savedR20+((offset)+1*8)(SP), R20; \
	MOVD	savedR21+((offset)+2*8)(SP), R21; \
	MOVD	savedR22+((offset)+3*8)(SP), R22; \
	MOVD	savedR23+((offset)+4*8)(SP), R23; \
	MOVD	savedR24+((offset)+5*8)(SP), R24; \
	MOVD	savedR25+((offset)+6*8)(SP), R25; \
	MOVD	savedR26+((offset)+7*8)(SP), R26; \
	MOVD	savedR27+((offset)+8*8)(SP), R27; \
	MOVD	savedR28+((offset)+9*8)(SP), g; /* R28 */

// Called by Windows as a Vectored Exception Handler (VEH).
// First argument is pointer to struct containing
// exception record and context pointers.
// Handler function is stored in R1
// Return 0 for 'not handled', -1 for handled.
// int32_t sigtramp(
//     PEXCEPTION_POINTERS ExceptionInfo,
//     func *GoExceptionHandler);
TEXT sigtramp<>(SB),NOSPLIT|NOFRAME,$0
	// Save R0, R1 (args) as well as LR, R27, R28 (callee-save).
	MOVD	R0, R5
	MOVD	R1, R6
	MOVD	LR, R7
	MOVD	R27, R16		// saved R27 (callee-save)
	MOVD	g, R17 			// saved R28 (callee-save from Windows, not really g)

	BL      runtime·load_g(SB)	// smashes R0, R27, R28 (g)
	CMP	$0, g			// is there a current g?
	BNE	2(PC)
	BL	runtime·badsignal2(SB)

	// Do we need to switch to the g0 stack?
	MOVD	g, R3			// R3 = oldg (for sigtramp_g0)
	MOVD	g_m(g), R2		// R2 = m
	MOVD	m_g0(R2), R2		// R2 = g0
	CMP	g, R2			// if curg == g0
	BNE	switch

	// No: on g0 stack already, tail call to sigtramp_g0.
	// Restore all the callee-saves so sigtramp_g0 can return to our caller.
	// We also pass R2 = g0, R3 = oldg, both set above.
	MOVD	R5, R0
	MOVD	R6, R1
	MOVD	R7, LR
	MOVD	R16, R27		// restore R27
	MOVD	R17, g 			// restore R28
	B	sigtramp_g0<>(SB)

switch:
	// switch to g0 stack (but do not update g - that's sigtramp_g0's job)
	MOVD	RSP, R8
	MOVD	(g_sched+gobuf_sp)(R2), R4	// R4 = g->gobuf.sp
	SUB	$(6*8), R4			// alloc space for saves - 2 words below SP for frame pointer, 3 for us to use, 1 for alignment
	MOVD	R4, RSP				// switch to g0 stack

	MOVD	$0, (0*8)(RSP)	// fake saved LR
	MOVD	R7, (1*8)(RSP)	// saved LR
	MOVD	R8, (2*8)(RSP)	// saved SP

	MOVD	R5, R0		// original args
	MOVD	R6, R1		// original args
	MOVD	R16, R27
	MOVD	R17, g 		// R28
	BL	sigtramp_g0<>(SB)

	// switch back to original stack; g already updated
	MOVD	(1*8)(RSP), R7	// saved LR
	MOVD	(2*8)(RSP), R8	// saved SP
	MOVD	R7, LR
	MOVD	R8, RSP
	RET

// sigtramp_g0 is running on the g0 stack, with R2 = g0, R3 = oldg.
// But g itself is not set - that's R28, a callee-save register,
// and it still holds the value from the Windows DLL caller.
TEXT sigtramp_g0<>(SB),NOSPLIT,$128
	NO_LOCAL_POINTERS

	// Push C callee-save registers R19-R28. LR, FP already saved.
	SAVE_R19_TO_R28(-10*8)

	MOVD	0(R0), R5	// R5 = ExceptionPointers->ExceptionRecord
	MOVD	8(R0), R6	// R6 = ExceptionPointers->ContextRecord
	MOVD	R6, context-(11*8)(SP)

	MOVD	R2, g 			// g0
	BL      runtime·save_g(SB)	// smashes R0

	MOVD	R5, (1*8)(RSP)	// arg0 (ExceptionRecord)
	MOVD	R6, (2*8)(RSP)	// arg1 (ContextRecord)
	MOVD	R3, (3*8)(RSP)	// arg2 (original g)
	MOVD	R3, oldg-(12*8)(SP)
	BL	(R1)
	MOVD	oldg-(12*8)(SP), g
	BL      runtime·save_g(SB)	// smashes R0
	MOVW	(4*8)(RSP), R0	// return value (0 or -1)

	// if return value is CONTINUE_SEARCH, do not set up control
	// flow guard workaround
	CMP	$0, R0
	BEQ	return

	// Check if we need to set up the control flow guard workaround.
	// On Windows, the stack pointer in the context must lie within
	// system stack limits when we resume from exception.
	// Store the resume SP and PC in alternate registers
	// and return to sigresume on the g0 stack.
	// sigresume makes no use of the stack at all,
	// loading SP from R0 and jumping to R1.
	// Note that smashing R0 and R1 is only safe because we know sigpanic
	// will not actually return to the original frame, so the registers
	// are effectively dead. But this does mean we can't use the
	// same mechanism for async preemption.
	MOVD	context-(11*8)(SP), R6
	MOVD	context_pc(R6), R2		// load PC from context record
	MOVD	$sigresume<>(SB), R1

	CMP	R1, R2
	BEQ	return				// do not clobber saved SP/PC

	// Save resume SP and PC into R0, R1.
	MOVD	context_xsp(R6), R2
	MOVD	R2, (context_x+0*8)(R6)
	MOVD	context_pc(R6), R2
	MOVD	R2, (context_x+1*8)(R6)

	// Set up context record to return to sigresume on g0 stack
	MOVD	RSP, R2
	MOVD	R2, context_xsp(R6)
	MOVD	$sigresume<>(SB), R2
	MOVD	R2, context_pc(R6)

return:
	RESTORE_R19_TO_R28(-10*8)		// smashes g
	RET

// Trampoline to resume execution from exception handler.
// This is part of the control flow guard workaround.
// It switches stacks and jumps to the continuation address.
// R0 and R1 are set above at the end of sigtramp<>
// in the context that starts executing at sigresume<>.
TEXT sigresume<>(SB),NOSPLIT|NOFRAME,$0
	// Important: do not smash LR,
	// which is set to a live value when handling
	// a signal by pushing a call to sigpanic onto the stack.
	MOVD	R0, RSP
	B	(R1)

TEXT runtime·exceptiontramp<ABIInternal>(SB),NOSPLIT|NOFRAME,$0
	MOVD	$runtime·exceptionhandler<ABIInternal>(SB), R1
	B	sigtramp<>(SB)

TEXT runtime·firstcontinuetramp<ABIInternal>(SB),NOSPLIT|NOFRAME,$0
	MOVD	$runtime·firstcontinuehandler<ABIInternal>(SB), R1
	B	sigtramp<>(SB)

TEXT runtime·lastcontinuetramp(SB),NOSPLIT|NOFRAME,$0
	MOVD	$runtime·lastcontinuehandler<ABIInternal>(SB), R1
	B	sigtramp<>(SB)

TEXT runtime·ctrlhandler<ABIInternal>(SB),NOSPLIT|NOFRAME,$0
	MOVD	$runtime·ctrlhandler1(SB), R1
	B	runtime·externalthreadhandler<ABIInternal>(SB)

TEXT runtime·profileloop<ABIInternal>(SB),NOSPLIT|NOFRAME,$0
	MOVD	$runtime·profileloop1(SB), R1
	B	runtime·externalthreadhandler<ABIInternal>(SB)

// externalthreadhander called with R0 = uint32 arg, R1 = Go function f.
// Need to call f(arg), which returns a uint32, and return it in R0.
TEXT runtime·externalthreadhandler<ABIInternal>(SB),NOSPLIT|TOPFRAME,$96-0
	NO_LOCAL_POINTERS

	// Push C callee-save registers R19-R28. LR, FP already saved.
	SAVE_R19_TO_R28(-10*8)

	// Allocate space for args, saved R0+R1, g, and m structures.
	// Hide from nosplit check.
	#define extra ((64+g__size+m__size+15)&~15)
	SUB	$extra, RSP, R2	// hide from nosplit overflow check
	MOVD	R2, RSP

	// Save R0 and R1 (our args).
	MOVD	R0, 32(RSP)
	MOVD	R1, 40(RSP)

	// Zero out m and g structures.
	MOVD	$64(RSP), R0
	MOVD	R0, 8(RSP)
	MOVD	$(m__size + g__size), R0
	MOVD	R0, 16(RSP)
	MOVD	$0, 0(RSP)	// not-saved LR
	BL	runtime·memclrNoHeapPointers(SB)

	// Initialize m and g structures.
	MOVD	$64(RSP), g
	MOVD	$g__size(g), R3		// m
	MOVD	R3, g_m(g)		// g->m = m
	MOVD	g, m_g0(R3)		// m->g0 = g
	MOVD	g, m_curg(R3)		// m->curg = g
	MOVD	RSP, R0
	MOVD	R0, g_stack+stack_hi(g)
	SUB	$(32*1024), R0
	MOVD	R0, (g_stack+stack_lo)(g)
	MOVD	R0, g_stackguard0(g)
	MOVD	R0, g_stackguard1(g)
	BL	runtime·save_g(SB)

	// Call function.
	MOVD	32(RSP), R0
	MOVD	40(RSP), R1
	MOVW	R0, 8(RSP)
	BL	(R1)

	// Clear g.
	MOVD	$0, g
	BL	runtime·save_g(SB)

	// Load return value (save_g would have smashed)
	MOVW	(2*8)(RSP), R0

	ADD	$extra, RSP, R2
	MOVD	R2, RSP
	#undef extra

	RESTORE_R19_TO_R28(-10*8)
	RET

GLOBL runtime·cbctxts(SB), NOPTR, $4

TEXT runtime·callbackasm1<ABIInternal>(SB),NOSPLIT,$208-0
	NO_LOCAL_POINTERS

	// On entry, the trampoline in zcallback_windows_arm64.s left
	// the callback index in R12 (which is volatile in the C ABI).

	// Save callback register arguments R0-R7.
	// We do this at the top of the frame so they're contiguous with stack arguments.
	MOVD	R0, arg0-(8*8)(SP)
	MOVD	R1, arg1-(7*8)(SP)
	MOVD	R2, arg2-(6*8)(SP)
	MOVD	R3, arg3-(5*8)(SP)
	MOVD	R4, arg4-(4*8)(SP)
	MOVD	R5, arg5-(3*8)(SP)
	MOVD	R6, arg6-(2*8)(SP)
	MOVD	R7, arg7-(1*8)(SP)

	// Push C callee-save registers R19-R28.
	// LR, FP already saved.
	SAVE_R19_TO_R28(-18*8)

	// Create a struct callbackArgs on our stack.
	MOVD	$cbargs-(18*8+callbackArgs__size)(SP), R13
	MOVD	R12, callbackArgs_index(R13)	// callback index
	MOVD	$arg0-(8*8)(SP), R0
	MOVD	R0, callbackArgs_args(R13)		// address of args vector
	MOVD	$0, R0
	MOVD	R0, callbackArgs_result(R13)	// result

	// Call cgocallback, which will call callbackWrap(frame).
	MOVD	$·callbackWrap(SB), R0	// PC of function to call
	MOVD	R13, R1	// frame (&callbackArgs{...})
	MOVD	$0, R2	// context
	MOVD	R0, (1*8)(RSP)
	MOVD	R1, (2*8)(RSP)
	MOVD	R2, (3*8)(RSP)
	BL	runtime·cgocallback(SB)

	// Get callback result.
	MOVD	$cbargs-(18*8+callbackArgs__size)(SP), R13
	MOVD	callbackArgs_result(R13), R0

	RESTORE_R19_TO_R28(-18*8)

	RET

// uint32 tstart_stdcall(M *newm);
TEXT runtime·tstart_stdcall<ABIInternal>(SB),NOSPLIT,$96-0
	SAVE_R19_TO_R28(-10*8)

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

	RESTORE_R19_TO_R28(-10*8)

	// Exit the thread.
	MOVD	$0, R0
	RET

// Runs on OS stack.
// duration (in -100ns units) is in dt+0(FP).
// g may be nil.
TEXT runtime·usleep2(SB),NOSPLIT,$32-4
	MOVW	dt+0(FP), R0
	MOVD	$16(RSP), R2		// R2 = pTime
	MOVD	R0, 0(R2)		// *pTime = -dt
	MOVD	$-1, R0			// R0 = handle
	MOVD	$0, R1			// R1 = FALSE (alertable)
	MOVD	runtime·_NtWaitForSingleObject(SB), R3
	SUB	$16, RSP	// skip over saved frame pointer below RSP
	BL	(R3)
	ADD	$16, RSP
	RET

// Runs on OS stack.
// duration (in -100ns units) is in dt+0(FP).
// g is valid.
// TODO: neeeds to be implemented properly.
TEXT runtime·usleep2HighRes(SB),NOSPLIT,$0-4
	B	runtime·abort(SB)

// Runs on OS stack.
TEXT runtime·switchtothread(SB),NOSPLIT,$16-0
	MOVD	runtime·_SwitchToThread(SB), R0
	SUB	$16, RSP	// skip over saved frame pointer below RSP
	BL	(R0)
	ADD	$16, RSP
	RET

// See http://www.dcl.hpi.uni-potsdam.de/research/WRK/2007/08/getting-os-information-the-kuser_shared_data-structure/
// Must read hi1, then lo, then hi2. The snapshot is valid if hi1 == hi2.
#define _INTERRUPT_TIME 0x7ffe0008
#define _SYSTEM_TIME 0x7ffe0014
#define time_lo 0
#define time_hi1 4
#define time_hi2 8

TEXT runtime·nanotime1(SB),NOSPLIT|NOFRAME,$0-8
	MOVB	runtime·useQPCTime(SB), R0
	CMP	$0, R0
	BNE	useQPC
	MOVD	$_INTERRUPT_TIME, R3
loop:
	MOVWU	time_hi1(R3), R1
	MOVWU	time_lo(R3), R0
	MOVWU	time_hi2(R3), R2
	CMP	R1, R2
	BNE	loop

	// wintime = R1:R0, multiply by 100
	ORR	R1<<32, R0
	MOVD	$100, R1
	MUL	R1, R0
	MOVD	R0, ret+0(FP)
	RET
useQPC:
	B	runtime·nanotimeQPC(SB)		// tail call

TEXT time·now(SB),NOSPLIT|NOFRAME,$0-24
	MOVB    runtime·useQPCTime(SB), R0
	CMP	$0, R0
	BNE	useQPC
	MOVD	$_INTERRUPT_TIME, R3
loop:
	MOVWU	time_hi1(R3), R1
	MOVWU	time_lo(R3), R0
	MOVWU	time_hi2(R3), R2
	CMP	R1, R2
	BNE	loop

	// wintime = R1:R0, multiply by 100
	ORR	R1<<32, R0
	MOVD	$100, R1
	MUL	R1, R0
	MOVD	R0, mono+16(FP)

	MOVD	$_SYSTEM_TIME, R3
wall:
	MOVWU	time_hi1(R3), R1
	MOVWU	time_lo(R3), R0
	MOVWU	time_hi2(R3), R2
	CMP	R1, R2
	BNE	wall

	// w = R1:R0 in 100ns units
	// convert to Unix epoch (but still 100ns units)
	#define delta 116444736000000000
	ORR	R1<<32, R0
	SUB	$delta, R0

	// Convert to nSec
	MOVD	$100, R1
	MUL	R1, R0

	// Code stolen from compiler output for:
	//
	//	var x uint64
	//	func f() (sec uint64, nsec uint32) { return x / 1000000000, uint32(x % 100000000) }
	//
	LSR	$1, R0, R1
	MOVD	$-8543223759426509416, R2
	UMULH	R2, R1, R1
	LSR	$28, R1, R1
	MOVD	R1, sec+0(FP)
	MOVD	$-6067343680855748867, R1
	UMULH	R0, R1, R1
	LSR	$26, R1, R1
	MOVD	$100000000, R2
	MSUB	R1, R0, R2, R0
	MOVW	R0, nsec+8(FP)
	RET
useQPC:
	B	runtime·nowQPC(SB)		// tail call

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
	MOVD	$runtime·abort(SB), R1
	BL	(R1)
ok:

	// Save offset from R18 into tls_g.
	LSL	$3, R0
	ADD	$TEB_TlsSlots, R0
	MOVD	R0, runtime·tls_g(SB)
	RET
