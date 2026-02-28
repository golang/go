// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"
#include "time_windows.h"

// Note: For system ABI, R0-R3 are args, R4-R11 are callee-save.

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

	// Does not return.
	B	runtime·abort(SB)

TEXT runtime·getlasterror(SB),NOSPLIT,$0
	MRC	15, 0, R0, C13, C0, 2
	MOVW	0x34(R0), R0
	MOVW	R0, ret+0(FP)
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

	// make room for sighandler arguments
	// and re-save old SP for restoring later.
	// (note that the 24(R3) here must match the 24(R13) above.)
	SUB	$40, R3
	MOVW	R13, 24(R3)		// save old stack pointer
	MOVW	R3, R13			// switch stack

g0:
	MOVW	0(R6), R2	// R2 = ExceptionPointers->ExceptionRecord
	MOVW	4(R6), R3	// R3 = ExceptionPointers->ContextRecord

	MOVW	$0, R4
	MOVW	R4, 0(R13)	// No saved link register.
	MOVW	R2, 4(R13)	// Move arg0 (ExceptionRecord) into position
	MOVW	R3, 8(R13)	// Move arg1 (ContextRecord) into position
	MOVW	R5, 12(R13)	// Move arg2 (original g) into position
	BL	(R7)		// Call the goroutine
	MOVW	16(R13), R4	// Fetch return value from stack

	// Save system stack pointer for sigresume setup below.
	// The exact value does not matter - nothing is read or written
	// from this address. It just needs to be on the system stack.
	MOVW	R13, R12

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
	// On Windows, the stack pointer in the context must lie within
	// system stack limits when we resume from exception.
	// Store the resume SP and PC on the g0 stack,
	// and return to sigresume on the g0 stack. sigresume
	// pops the saved PC and SP from the g0 stack, resuming execution
	// at the desired location.
	// If sigresume has already been set up by a previous exception
	// handler, don't clobber the stored SP and PC on the stack.
	MOVW	4(R3), R3			// PEXCEPTION_POINTERS->Context
	MOVW	context_pc(R3), R2		// load PC from context record
	MOVW	$sigresume<>(SB), R1
	CMP	R1, R2
	B.EQ	return				// do not clobber saved SP/PC

	// Save resume SP and PC into R0, R1.
	MOVW	context_spr(R3), R2
	MOVW	R2, context_r0(R3)
	MOVW	context_pc(R3), R2
	MOVW	R2, context_r1(R3)

	// Set up context record to return to sigresume on g0 stack
	MOVW	R12, context_spr(R3)
	MOVW	$sigresume<>(SB), R2
	MOVW	R2, context_pc(R3)

return:
	B	(R14)				// return

// Trampoline to resume execution from exception handler.
// This is part of the control flow guard workaround.
// It switches stacks and jumps to the continuation address.
// R0 and R1 are set above at the end of sigtramp<>
// in the context that starts executing at sigresume<>.
TEXT sigresume<>(SB),NOSPLIT|NOFRAME,$0
	// Important: do not smash LR,
	// which is set to a live value when handling
	// a signal by pushing a call to sigpanic onto the stack.
	MOVW	R0, R13
	B	(R1)

TEXT runtime·exceptiontramp(SB),NOSPLIT|NOFRAME,$0
	MOVW	$runtime·exceptionhandler(SB), R1
	B	sigtramp<>(SB)

TEXT runtime·firstcontinuetramp(SB),NOSPLIT|NOFRAME,$0
	MOVW	$runtime·firstcontinuehandler(SB), R1
	B	sigtramp<>(SB)

TEXT runtime·lastcontinuetramp(SB),NOSPLIT|NOFRAME,$0
	MOVW	$runtime·lastcontinuehandler(SB), R1
	B	sigtramp<>(SB)

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

// Runs on OS stack.
// duration (in -100ns units) is in dt+0(FP).
// g may be nil.
TEXT runtime·usleep2(SB),NOSPLIT|NOFRAME,$0-4
	MOVW	dt+0(FP), R3
	MOVM.DB.W [R4, R14], (R13)	// push {r4, lr}
	MOVW	R13, R4			// Save SP
	SUB	$8, R13			// R13 = R13 - 8
	BIC	$0x7, R13		// Align SP for ABI
	MOVW	$0, R1			// R1 = FALSE (alertable)
	MOVW	$-1, R0			// R0 = handle
	MOVW	R13, R2			// R2 = pTime
	MOVW	R3, 0(R2)		// time_lo
	MOVW	R0, 4(R2)		// time_hi
	MOVW	runtime·_NtWaitForSingleObject(SB), R3
	BL	(R3)
	MOVW	R4, R13			// Restore SP
	MOVM.IA.W (R13), [R4, R15]	// pop {R4, pc}

// Runs on OS stack.
// duration (in -100ns units) is in dt+0(FP).
// g is valid.
// TODO: neeeds to be implemented properly.
TEXT runtime·usleep2HighRes(SB),NOSPLIT|NOFRAME,$0-4
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

// never called (this is a GOARM=7 platform)
TEXT runtime·read_tls_fallback(SB),NOSPLIT|NOFRAME,$0
	MOVW	$0xabcd, R0
	MOVW	R0, (R0)
	RET

TEXT runtime·nanotime1(SB),NOSPLIT|NOFRAME,$0-8
	MOVW	$0, R0
	MOVB	runtime·useQPCTime(SB), R0
	CMP	$0, R0
	BNE	useQPC
	MOVW	$_INTERRUPT_TIME, R3
loop:
	MOVW	time_hi1(R3), R1
	DMB	MB_ISH
	MOVW	time_lo(R3), R0
	DMB	MB_ISH
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

	MOVW	R4, R13
	MOVM.IA.W (R13), [R4, R15]	// pop {r4, pc}

// Holds the TLS Slot, which was allocated by TlsAlloc()
GLOBL runtime·tls_g+0(SB), NOPTR, $4
