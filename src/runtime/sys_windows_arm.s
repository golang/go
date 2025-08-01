// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"
#include "time_windows.h"

// Note: For system ABI, R0-R3 are args, R4-R11 are callee-save.

TEXT runtime·getlasterror(SB),NOSPLIT,$0
	MRC	15, 0, R0, C13, C0, 2
	MOVW	0x34(R0), R0
	MOVW	R0, ret+0(FP)
	RET

// Called by Windows as a Vectored Exception Handler (VEH).
// R0 is pointer to struct containing
// exception record and context pointers.
// R1 is the kind of sigtramp function.
// Return value of sigtrampgo is stored in R0.
TEXT sigtramp<>(SB),NOSPLIT|NOFRAME,$0
	MOVM.DB.W [R4-R11, R14], (R13)	// push {r4-r11, lr} (SP-=40)
	SUB	$(16), R13		// reserve space for parameters/retval to go call

	MOVW	R0, R6			// Save param0
	MOVW	R1, R7			// Save param1
	BL	runtime·load_g(SB)	// Clobbers R0

	MOVW	$0, R4
	MOVW	R4, 0(R13)	// No saved link register.
	MOVW	R6, 4(R13)	// Move arg0 into position
	MOVW	R7, 8(R13)	// Move arg1 into position
	BL	runtime·sigtrampgo(SB)
	MOVW	12(R13), R0	// Fetch return value from stack

	ADD	$(16), R13			// free locals
	MOVM.IA.W (R13), [R4-R11, R14]	// pop {r4-r11, lr}

	B	(R14)				// return

// Trampoline to resume execution from exception handler.
// This is part of the control flow guard workaround.
// It switches stacks and jumps to the continuation address.
// R0 and R1 are set above at the end of sigtrampgo
// in the context that starts executing at sigresume.
TEXT runtime·sigresume(SB),NOSPLIT|NOFRAME,$0
	// Important: do not smash LR,
	// which is set to a live value when handling
	// a signal by pushing a call to sigpanic onto the stack.
	MOVW	R0, R13
	B	(R1)

TEXT runtime·exceptiontramp(SB),NOSPLIT|NOFRAME,$0
	MOVW	$const_callbackVEH, R1
	B	sigtramp<>(SB)

TEXT runtime·firstcontinuetramp(SB),NOSPLIT|NOFRAME,$0
	MOVW	$const_callbackFirstVCH, R1
	B	sigtramp<>(SB)

TEXT runtime·lastcontinuetramp(SB),NOSPLIT|NOFRAME,$0
	MOVW	$const_callbackLastVCH, R1
	B	sigtramp<>(SB)

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

TEXT ·publicationBarrier(SB),NOSPLIT|NOFRAME,$0-0
	B	runtime·armPublicationBarrier(SB)

// never called (this is a GOARM=7 platform)
TEXT runtime·read_tls_fallback(SB),NOSPLIT,$0
	MOVW	$0xabcd, R0
	MOVW	R0, (R0)
	RET

TEXT runtime·nanotime1(SB),NOSPLIT,$0-8
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

// save_g saves the g register (R10) into thread local memory
// so that we can call externally compiled
// ARM code that will overwrite those registers.
// NOTE: runtime.gogo assumes that R1 is preserved by this function.
//       runtime.mcall assumes this function only clobbers R0 and R11.
// Returns with g in R0.
// Save the value in the _TEB->TlsSlots array.
// Effectively implements TlsSetValue().
// tls_g stores the TLS slot allocated TlsAlloc().
TEXT runtime·save_g(SB),NOSPLIT,$0
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
TEXT runtime·load_g(SB),NOSPLIT,$0
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
