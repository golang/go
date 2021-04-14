// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

// void runtime·asmstdcall(void *c);
TEXT runtime·asmstdcall(SB),NOSPLIT|NOFRAME,$0
	MOVM.DB.W [R4, R5, R14], (R13)	// push {r4, r5, lr}
	MOVW	R0, R4			// put libcall * in r4
	MOVW	R13, R5			// save stack pointer in r5

	// SetLastError(0)
	MOVW	$0, R0
	MRC	15, 0, R1, C13, C0, 2
	MOVW	R0, 0x34(R1)

	MOVW	8(R4), R12	// libcall->args

	// Do we have more than 4 arguments?
	MOVW	4(R4), R0	// libcall->n
	SUB.S	$4, R0, R2
	BLE	loadregs

	// Reserve stack space for remaining args
	SUB	R2<<2, R13
	BIC	$0x7, R13	// alignment for ABI

	// R0: count of arguments
	// R1:
	// R2: loop counter, from 0 to (n-4)
	// R3: scratch
	// R4: pointer to libcall struct
	// R12: libcall->args
	MOVW	$0, R2
stackargs:
	ADD	$4, R2, R3		// r3 = args[4 + i]
	MOVW	R3<<2(R12), R3
	MOVW	R3, R2<<2(R13)		// stack[i] = r3

	ADD	$1, R2			// i++
	SUB	$4, R0, R3		// while (i < (n - 4))
	CMP	R3, R2
	BLT	stackargs

loadregs:
	CMP	$3, R0
	MOVW.GT 12(R12), R3

	CMP	$2, R0
	MOVW.GT 8(R12), R2

	CMP	$1, R0
	MOVW.GT 4(R12), R1

	CMP	$0, R0
	MOVW.GT 0(R12), R0

	BIC	$0x7, R13		// alignment for ABI
	MOVW	0(R4), R12		// branch to libcall->fn
	BL	(R12)

	MOVW	R5, R13			// free stack space
	MOVW	R0, 12(R4)		// save return value to libcall->r1
	MOVW	R1, 16(R4)

	// GetLastError
	MRC	15, 0, R1, C13, C0, 2
	MOVW	0x34(R1), R0
	MOVW	R0, 20(R4)		// store in libcall->err

	MOVM.IA.W (R13), [R4, R5, R15]

TEXT runtime·badsignal2(SB),NOSPLIT|NOFRAME,$0
	MOVM.DB.W [R4, R14], (R13)	// push {r4, lr}
	MOVW	R13, R4			// save original stack pointer
	SUB	$8, R13			// space for 2 variables
	BIC	$0x7, R13		// alignment for ABI

	// stderr
	MOVW	runtime·_GetStdHandle(SB), R1
	MOVW	$-12, R0
	BL	(R1)

	MOVW	$runtime·badsignalmsg(SB), R1	// lpBuffer
	MOVW	$runtime·badsignallen(SB), R2	// lpNumberOfBytesToWrite
	MOVW	(R2), R2
	ADD	$0x4, R13, R3		// lpNumberOfBytesWritten
	MOVW	$0, R12			// lpOverlapped
	MOVW	R12, (R13)

	MOVW	runtime·_WriteFile(SB), R12
	BL	(R12)

	MOVW	R4, R13			// restore SP
	MOVM.IA.W (R13), [R4, R15]	// pop {r4, pc}

TEXT runtime·getlasterror(SB),NOSPLIT,$0
	MRC	15, 0, R0, C13, C0, 2
	MOVW	0x34(R0), R0
	MOVW	R0, ret+0(FP)
	RET

TEXT runtime·setlasterror(SB),NOSPLIT|NOFRAME,$0
	MRC	15, 0, R1, C13, C0, 2
	MOVW	R0, 0x34(R1)
	RET

// Called by Windows as a Vectored Exception Handler (VEH).
// First argument is pointer to struct containing
// exception record and context pointers.
// Handler function is stored in R1
// Return 0 for 'not handled', -1 for handled.
// int32_t sigtramp(
//     PEXCEPTION_POINTERS ExceptionInfo,
//     func *GoExceptionHandler);
TEXT sigtramp<>(SB),NOSPLIT|NOFRAME,$0
	MOVM.DB.W [R0, R4-R11, R14], (R13)	// push {r0, r4-r11, lr} (SP-=40)
	SUB	$(8+20), R13		// reserve space for g, sp, and
					// parameters/retval to go call

	MOVW	R0, R6			// Save param0
	MOVW	R1, R7			// Save param1

	BL      runtime·load_g(SB)
	CMP	$0, g			// is there a current g?
	BL.EQ	runtime·badsignal2(SB)

	// save g and SP in case of stack switch
	MOVW	R13, 24(R13)
	MOVW	g, 20(R13)

	// do we need to switch to the g0 stack?
	MOVW	g, R5			// R5 = g
	MOVW	g_m(R5), R2		// R2 = m
	MOVW	m_g0(R2), R4		// R4 = g0
	CMP	R5, R4			// if curg == g0
	BEQ	g0

	// switch to g0 stack
	MOVW	R4, g				// g = g0
	MOVW	(g_sched+gobuf_sp)(g), R3	// R3 = g->gobuf.sp
	BL      runtime·save_g(SB)

	// traceback will think that we've done PUSH and SUB
	// on this stack, so subtract them here to match.
	// (we need room for sighandler arguments anyway).
	// and re-save old SP for restoring later.
	SUB	$(40+8+20), R3
	MOVW	R13, 24(R3)		// save old stack pointer
	MOVW	R3, R13			// switch stack

g0:
	MOVW	0(R6), R2	// R2 = ExceptionPointers->ExceptionRecord
	MOVW	4(R6), R3	// R3 = ExceptionPointers->ContextRecord

	// make it look like mstart called us on g0, to stop traceback
	MOVW    $runtime·mstart(SB), R4

	MOVW	R4, 0(R13)	// Save link register for traceback
	MOVW	R2, 4(R13)	// Move arg0 (ExceptionRecord) into position
	MOVW	R3, 8(R13)	// Move arg1 (ContextRecord) into position
	MOVW	R5, 12(R13)	// Move arg2 (original g) into position
	BL	(R7)		// Call the go routine
	MOVW	16(R13), R4	// Fetch return value from stack

	// Compute the value of the g0 stack pointer after deallocating
	// this frame, then allocating 8 bytes. We may need to store
	// the resume SP and PC on the g0 stack to work around
	// control flow guard when we resume from the exception.
	ADD	$(40+20), R13, R12

	// switch back to original stack and g
	MOVW	24(R13), R13
	MOVW	20(R13), g
	BL      runtime·save_g(SB)

done:
	MOVW	R4, R0				// move retval into position
	ADD	$(8 + 20), R13			// free locals
	MOVM.IA.W (R13), [R3, R4-R11, R14]	// pop {r3, r4-r11, lr}

	// if return value is CONTINUE_SEARCH, do not set up control
	// flow guard workaround
	CMP	$0, R0
	BEQ	return

	// Check if we need to set up the control flow guard workaround.
	// On Windows/ARM, the stack pointer must lie within system
	// stack limits when we resume from exception.
	// Store the resume SP and PC on the g0 stack,
	// and return to returntramp on the g0 stack. returntramp
	// pops the saved PC and SP from the g0 stack, resuming execution
	// at the desired location.
	// If returntramp has already been set up by a previous exception
	// handler, don't clobber the stored SP and PC on the stack.
	MOVW	4(R3), R3			// PEXCEPTION_POINTERS->Context
	MOVW	0x40(R3), R2			// load PC from context record
	MOVW	$returntramp<>(SB), R1
	CMP	R1, R2
	B.EQ	return				// do not clobber saved SP/PC

	// Save resume SP and PC on g0 stack
	MOVW	0x38(R3), R2			// load SP from context record
	MOVW	R2, 0(R12)			// Store resume SP on g0 stack
	MOVW	0x40(R3), R2			// load PC from context record
	MOVW	R2, 4(R12)			// Store resume PC on g0 stack

	// Set up context record to return to returntramp on g0 stack
	MOVW	R12, 0x38(R3)			// save g0 stack pointer
						// in context record
	MOVW	$returntramp<>(SB), R2	// save resume address
	MOVW	R2, 0x40(R3)			// in context record

return:
	B	(R14)				// return

//
// Trampoline to resume execution from exception handler.
// This is part of the control flow guard workaround.
// It switches stacks and jumps to the continuation address.
//
TEXT returntramp<>(SB),NOSPLIT|NOFRAME,$0
	MOVM.IA	(R13), [R13, R15]		// ldm sp, [sp, pc]

TEXT runtime·exceptiontramp(SB),NOSPLIT|NOFRAME,$0
	MOVW	$runtime·exceptionhandler(SB), R1
	B	sigtramp<>(SB)

TEXT runtime·firstcontinuetramp(SB),NOSPLIT|NOFRAME,$0
	MOVW	$runtime·firstcontinuehandler(SB), R1
	B	sigtramp<>(SB)

TEXT runtime·lastcontinuetramp(SB),NOSPLIT|NOFRAME,$0
	MOVW	$runtime·lastcontinuehandler(SB), R1
	B	sigtramp<>(SB)

TEXT runtime·ctrlhandler(SB),NOSPLIT|NOFRAME,$0
	MOVW	$runtime·ctrlhandler1(SB), R1
	B	runtime·externalthreadhandler(SB)

TEXT runtime·profileloop(SB),NOSPLIT|NOFRAME,$0
	MOVW	$runtime·profileloop1(SB), R1
	B	runtime·externalthreadhandler(SB)

// int32 externalthreadhandler(uint32 arg, int (*func)(uint32))
// stack layout:
//   +----------------+
//   | callee-save    |
//   | registers      |
//   +----------------+
//   | m              |
//   +----------------+
// 20| g              |
//   +----------------+
// 16| func ptr (r1)  |
//   +----------------+
// 12| argument (r0)  |
//---+----------------+
// 8 | param1         |
//   +----------------+
// 4 | param0         |
//   +----------------+
// 0 | retval         |
//   +----------------+
//
TEXT runtime·externalthreadhandler(SB),NOSPLIT|NOFRAME,$0
	MOVM.DB.W [R4-R11, R14], (R13)		// push {r4-r11, lr}
	SUB	$(m__size + g__size + 20), R13	// space for locals
	MOVW	R0, 12(R13)
	MOVW	R1, 16(R13)

	// zero out m and g structures
	ADD	$20, R13, R0			// compute pointer to g
	MOVW	R0, 4(R13)
	MOVW	$(m__size + g__size), R0
	MOVW	R0, 8(R13)
	BL	runtime·memclrNoHeapPointers(SB)

	// initialize m and g structures
	ADD	$20, R13, R2			// R2 = g
	ADD	$(20 + g__size), R13, R3	// R3 = m
	MOVW	R2, m_g0(R3)			// m->g0 = g
	MOVW	R3, g_m(R2)			// g->m = m
	MOVW	R2, m_curg(R3)			// m->curg = g

	MOVW	R2, g
	BL	runtime·save_g(SB)

	// set up stackguard stuff
	MOVW	R13, R0
	MOVW	R0, g_stack+stack_hi(g)
	SUB	$(32*1024), R0
	MOVW	R0, (g_stack+stack_lo)(g)
	MOVW	R0, g_stackguard0(g)
	MOVW	R0, g_stackguard1(g)

	// move argument into position and call function
	MOVW	12(R13), R0
	MOVW	R0, 4(R13)
	MOVW	16(R13), R1
	BL	(R1)

	// clear g
	MOVW	$0, g
	BL	runtime·save_g(SB)

	MOVW	0(R13), R0			// load return value
	ADD	$(m__size + g__size + 20), R13	// free locals
	MOVM.IA.W (R13), [R4-R11, R15]		// pop {r4-r11, pc}

GLOBL runtime·cbctxts(SB), NOPTR, $4

TEXT runtime·callbackasm1(SB),NOSPLIT|NOFRAME,$0
	// On entry, the trampoline in zcallback_windows_arm.s left
	// the callback index in R12 (which is volatile in the C ABI).

	// Push callback register arguments r0-r3. We do this first so
	// they're contiguous with stack arguments.
	MOVM.DB.W [R0-R3], (R13)
	// Push C callee-save registers r4-r11 and lr.
	MOVM.DB.W [R4-R11, R14], (R13)
	SUB	$(16 + callbackArgs__size), R13	// space for locals

	// Create a struct callbackArgs on our stack.
	MOVW	R12, (16+callbackArgs_index)(R13)	// callback index
	MOVW	$(16+callbackArgs__size+4*9)(R13), R0
	MOVW	R0, (16+callbackArgs_args)(R13)		// address of args vector
	MOVW	$0, R0
	MOVW	R0, (16+callbackArgs_result)(R13)	// result

	// Prepare for entry to Go.
	BL	runtime·load_g(SB)

	// Call cgocallback, which will call callbackWrap(frame).
	MOVW	$0, R0
	MOVW	R0, 12(R13)	// context
	MOVW	$16(R13), R1	// R1 = &callbackArgs{...}
	MOVW	R1, 8(R13)	// frame (address of callbackArgs)
	MOVW	$·callbackWrap(SB), R1
	MOVW	R1, 4(R13)	// PC of function to call
	BL	runtime·cgocallback(SB)

	// Get callback result.
	MOVW	(16+callbackArgs_result)(R13), R0

	ADD	$(16 + callbackArgs__size), R13	// free locals
	MOVM.IA.W (R13), [R4-R11, R12]	// pop {r4-r11, lr=>r12}
	ADD	$(4*4), R13	// skip r0-r3
	B	(R12)	// return

// uint32 tstart_stdcall(M *newm);
TEXT runtime·tstart_stdcall(SB),NOSPLIT|NOFRAME,$0
	MOVM.DB.W [R4-R11, R14], (R13)		// push {r4-r11, lr}

	MOVW	m_g0(R0), g
	MOVW	R0, g_m(g)
	BL	runtime·save_g(SB)

	// do per-thread TLS initialization
	BL	init_thread_tls<>(SB)

	// Layout new m scheduler stack on os stack.
	MOVW	R13, R0
	MOVW	R0, g_stack+stack_hi(g)
	SUB	$(64*1024), R0
	MOVW	R0, (g_stack+stack_lo)(g)
	MOVW	R0, g_stackguard0(g)
	MOVW	R0, g_stackguard1(g)

	BL	runtime·emptyfunc(SB)	// fault if stack check is wrong
	BL	runtime·mstart(SB)

	// Exit the thread.
	MOVW	$0, R0
	MOVM.IA.W (R13), [R4-R11, R15]		// pop {r4-r11, pc}

// onosstack calls fn on OS stack.
// adapted from asm_arm.s : systemstack
// func onosstack(fn unsafe.Pointer, arg uint32)
TEXT runtime·onosstack(SB),NOSPLIT,$0
	MOVW	fn+0(FP), R5		// R5 = fn
	MOVW	arg+4(FP), R6		// R6 = arg

	// This function can be called when there is no g,
	// for example, when we are handling a callback on a non-go thread.
	// In this case we're already on the system stack.
	CMP	$0, g
	BEQ	noswitch

	MOVW	g_m(g), R1		// R1 = m

	MOVW	m_gsignal(R1), R2	// R2 = gsignal
	CMP	g, R2
	B.EQ	noswitch

	MOVW	m_g0(R1), R2		// R2 = g0
	CMP	g, R2
	B.EQ	noswitch

	MOVW	m_curg(R1), R3
	CMP	g, R3
	B.EQ	switch

	// Bad: g is not gsignal, not g0, not curg. What is it?
	// Hide call from linker nosplit analysis.
	MOVW	$runtime·badsystemstack(SB), R0
	BL	(R0)
	B	runtime·abort(SB)

switch:
	// save our state in g->sched. Pretend to
	// be systemstack_switch if the G stack is scanned.
	MOVW	$runtime·systemstack_switch(SB), R3
	ADD	$4, R3, R3 // get past push {lr}
	MOVW	R3, (g_sched+gobuf_pc)(g)
	MOVW	R13, (g_sched+gobuf_sp)(g)
	MOVW	LR, (g_sched+gobuf_lr)(g)
	MOVW	g, (g_sched+gobuf_g)(g)

	// switch to g0
	MOVW	R2, g
	MOVW	(g_sched+gobuf_sp)(R2), R3
	// make it look like mstart called systemstack on g0, to stop traceback
	SUB	$4, R3, R3
	MOVW	$runtime·mstart(SB), R4
	MOVW	R4, 0(R3)
	MOVW	R3, R13

	// call target function
	MOVW	R6, R0		// arg
	BL	(R5)

	// switch back to g
	MOVW	g_m(g), R1
	MOVW	m_curg(R1), g
	MOVW	(g_sched+gobuf_sp)(g), R13
	MOVW	$0, R3
	MOVW	R3, (g_sched+gobuf_sp)(g)
	RET

noswitch:
	// Using a tail call here cleans up tracebacks since we won't stop
	// at an intermediate systemstack.
	MOVW.P	4(R13), R14	// restore LR
	MOVW	R6, R0		// arg
	B	(R5)

// Runs on OS stack. Duration (in 100ns units) is in R0.
TEXT runtime·usleep2(SB),NOSPLIT|NOFRAME,$0
	MOVM.DB.W [R4, R14], (R13)	// push {r4, lr}
	MOVW	R13, R4			// Save SP
	SUB	$8, R13			// R13 = R13 - 8
	BIC	$0x7, R13		// Align SP for ABI
	RSB	$0, R0, R3		// R3 = -R0
	MOVW	$0, R1			// R1 = FALSE (alertable)
	MOVW	$-1, R0			// R0 = handle
	MOVW	R13, R2			// R2 = pTime
	MOVW	R3, 0(R2)		// time_lo
	MOVW	R0, 4(R2)		// time_hi
	MOVW	runtime·_NtWaitForSingleObject(SB), R3
	BL	(R3)
	MOVW	R4, R13			// Restore SP
	MOVM.IA.W (R13), [R4, R15]	// pop {R4, pc}

// Runs on OS stack. Duration (in 100ns units) is in R0.
// TODO: neeeds to be implemented properly.
TEXT runtime·usleep2HighRes(SB),NOSPLIT|NOFRAME,$0
	B	runtime·abort(SB)

// Runs on OS stack.
TEXT runtime·switchtothread(SB),NOSPLIT|NOFRAME,$0
	MOVM.DB.W [R4, R14], (R13)  	// push {R4, lr}
	MOVW    R13, R4
	BIC	$0x7, R13		// alignment for ABI
	MOVW	runtime·_SwitchToThread(SB), R0
	BL	(R0)
	MOVW 	R4, R13			// restore stack pointer
	MOVM.IA.W (R13), [R4, R15]	// pop {R4, pc}

TEXT ·publicationBarrier(SB),NOSPLIT|NOFRAME,$0-0
	B	runtime·armPublicationBarrier(SB)

// never called (cgo not supported)
TEXT runtime·read_tls_fallback(SB),NOSPLIT|NOFRAME,$0
	MOVW	$0xabcd, R0
	MOVW	R0, (R0)
	RET

// See http://www.dcl.hpi.uni-potsdam.de/research/WRK/2007/08/getting-os-information-the-kuser_shared_data-structure/
// Must read hi1, then lo, then hi2. The snapshot is valid if hi1 == hi2.
#define _INTERRUPT_TIME 0x7ffe0008
#define _SYSTEM_TIME 0x7ffe0014
#define time_lo 0
#define time_hi1 4
#define time_hi2 8

TEXT runtime·nanotime1(SB),NOSPLIT,$0-8
	MOVW	$0, R0
	MOVB	runtime·useQPCTime(SB), R0
	CMP	$0, R0
	BNE	useQPC
	MOVW	$_INTERRUPT_TIME, R3
loop:
	MOVW	time_hi1(R3), R1
	MOVW	time_lo(R3), R0
	MOVW	time_hi2(R3), R2
	CMP	R1, R2
	BNE	loop

	// wintime = R1:R0, multiply by 100
	MOVW	$100, R2
	MULLU	R0, R2, (R4, R3)    // R4:R3 = R1:R0 * R2
	MULA	R1, R2, R4, R4

	// wintime*100 = R4:R3
	MOVW	R3, ret_lo+0(FP)
	MOVW	R4, ret_hi+4(FP)
	RET
useQPC:
	B	runtime·nanotimeQPC(SB)		// tail call
	RET

TEXT time·now(SB),NOSPLIT,$0-20
	MOVW    $0, R0
	MOVB    runtime·useQPCTime(SB), R0
	CMP	$0, R0
	BNE	useQPC
	MOVW	$_INTERRUPT_TIME, R3
loop:
	MOVW	time_hi1(R3), R1
	MOVW	time_lo(R3), R0
	MOVW	time_hi2(R3), R2
	CMP	R1, R2
	BNE	loop

	// wintime = R1:R0, multiply by 100
	MOVW	$100, R2
	MULLU	R0, R2, (R4, R3)    // R4:R3 = R1:R0 * R2
	MULA	R1, R2, R4, R4

	// wintime*100 = R4:R3
	MOVW	R3, mono+12(FP)
	MOVW	R4, mono+16(FP)

	MOVW	$_SYSTEM_TIME, R3
wall:
	MOVW	time_hi1(R3), R1
	MOVW	time_lo(R3), R0
	MOVW	time_hi2(R3), R2
	CMP	R1, R2
	BNE	wall

	// w = R1:R0 in 100ns untis
	// convert to Unix epoch (but still 100ns units)
	#define delta 116444736000000000
	SUB.S   $(delta & 0xFFFFFFFF), R0
	SBC     $(delta >> 32), R1

	// Convert to nSec
	MOVW    $100, R2
	MULLU   R0, R2, (R4, R3)    // R4:R3 = R1:R0 * R2
	MULA    R1, R2, R4, R4
	// w = R2:R1 in nSec
	MOVW    R3, R1	      // R4:R3 -> R2:R1
	MOVW    R4, R2

	// multiply nanoseconds by reciprocal of 10**9 (scaled by 2**61)
	// to get seconds (96 bit scaled result)
	MOVW	$0x89705f41, R3		// 2**61 * 10**-9
	MULLU	R1,R3,(R6,R5)		// R7:R6:R5 = R2:R1 * R3
	MOVW	$0,R7
	MULALU	R2,R3,(R7,R6)

	// unscale by discarding low 32 bits, shifting the rest by 29
	MOVW	R6>>29,R6		// R7:R6 = (R7:R6:R5 >> 61)
	ORR	R7<<3,R6
	MOVW	R7>>29,R7

	// subtract (10**9 * sec) from nsec to get nanosecond remainder
	MOVW	$1000000000, R5	// 10**9
	MULLU	R6,R5,(R9,R8)   // R9:R8 = R7:R6 * R5
	MULA	R7,R5,R9,R9
	SUB.S	R8,R1		// R2:R1 -= R9:R8
	SBC	R9,R2

	// because reciprocal was a truncated repeating fraction, quotient
	// may be slightly too small -- adjust to make remainder < 10**9
	CMP	R5,R1	// if remainder > 10**9
	SUB.HS	R5,R1   //    remainder -= 10**9
	ADD.HS	$1,R6	//    sec += 1

	MOVW	R6,sec_lo+0(FP)
	MOVW	R7,sec_hi+4(FP)
	MOVW	R1,nsec+8(FP)
	RET
useQPC:
	B	runtime·nanotimeQPC(SB)		// tail call
	RET

// save_g saves the g register (R10) into thread local memory
// so that we can call externally compiled
// ARM code that will overwrite those registers.
// NOTE: runtime.gogo assumes that R1 is preserved by this function.
//       runtime.mcall assumes this function only clobbers R0 and R11.
// Returns with g in R0.
// Save the value in the _TEB->TlsSlots array.
// Effectively implements TlsSetValue().
// tls_g stores the TLS slot allocated TlsAlloc().
TEXT runtime·save_g(SB),NOSPLIT|NOFRAME,$0
	MRC	15, 0, R0, C13, C0, 2
	ADD	$0xe10, R0
	MOVW 	$runtime·tls_g(SB), R11
	MOVW	(R11), R11
	MOVW	g, R11<<2(R0)
	MOVW	g, R0	// preserve R0 across call to setg<>
	RET

// load_g loads the g register from thread-local memory,
// for use after calling externally compiled
// ARM code that overwrote those registers.
// Get the value from the _TEB->TlsSlots array.
// Effectively implements TlsGetValue().
TEXT runtime·load_g(SB),NOSPLIT|NOFRAME,$0
	MRC	15, 0, R0, C13, C0, 2
	ADD	$0xe10, R0
	MOVW 	$runtime·tls_g(SB), g
	MOVW	(g), g
	MOVW	g<<2(R0), g
	RET

// This is called from rt0_go, which runs on the system stack
// using the initial stack allocated by the OS.
// It calls back into standard C using the BL below.
// To do that, the stack pointer must be 8-byte-aligned.
TEXT runtime·_initcgo(SB),NOSPLIT|NOFRAME,$0
	MOVM.DB.W [R4, R14], (R13)	// push {r4, lr}

	// Ensure stack is 8-byte aligned before calling C code
	MOVW	R13, R4
	BIC	$0x7, R13

	// Allocate a TLS slot to hold g across calls to external code
	MOVW 	$runtime·_TlsAlloc(SB), R0
	MOVW	(R0), R0
	BL	(R0)

	// Assert that slot is less than 64 so we can use _TEB->TlsSlots
	CMP	$64, R0
	MOVW	$runtime·abort(SB), R1
	BL.GE	(R1)

	// Save Slot into tls_g
	MOVW 	$runtime·tls_g(SB), R1
	MOVW	R0, (R1)

	BL	init_thread_tls<>(SB)

	MOVW	R4, R13
	MOVM.IA.W (R13), [R4, R15]	// pop {r4, pc}

// void init_thread_tls()
//
// Does per-thread TLS initialization. Saves a pointer to the TLS slot
// holding G, in the current m.
//
//     g->m->tls[0] = &_TEB->TlsSlots[tls_g]
//
// The purpose of this is to enable the profiling handler to get the
// current g associated with the thread. We cannot use m->curg because curg
// only holds the current user g. If the thread is executing system code or
// external code, m->curg will be NULL. The thread's TLS slot always holds
// the current g, so save a reference to this location so the profiling
// handler can get the real g from the thread's m.
//
// Clobbers R0-R3
TEXT init_thread_tls<>(SB),NOSPLIT|NOFRAME,$0
	// compute &_TEB->TlsSlots[tls_g]
	MRC	15, 0, R0, C13, C0, 2
	ADD	$0xe10, R0
	MOVW 	$runtime·tls_g(SB), R1
	MOVW	(R1), R1
	MOVW	R1<<2, R1
	ADD	R1, R0

	// save in g->m->tls[0]
	MOVW	g_m(g), R1
	MOVW	R0, m_tls(R1)
	RET

// Holds the TLS Slot, which was allocated by TlsAlloc()
GLOBL runtime·tls_g+0(SB), NOPTR, $4
