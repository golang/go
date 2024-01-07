// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build race

#include "go_asm.h"
#include "funcdata.h"
#include "textflag.h"

// The following thunks allow calling the gcc-compiled race runtime directly
// from Go code without going all the way through cgo.
// First, it's much faster (up to 50% speedup for real Go programs).
// Second, it eliminates race-related special cases from cgocall and scheduler.
// Third, in long-term it will allow to remove cyclic runtime/race dependency on cmd/go.

// A brief recap of the s390x C calling convention.
// Arguments are passed in R2...R6, the rest is on stack.
// Callee-saved registers are: R6...R13, R15.
// Temporary registers are: R0...R5, R14.

// When calling racecalladdr, R1 is the call target address.

// The race ctx, ThreadState *thr below, is passed in R2 and loaded in racecalladdr.

// func runtime·raceread(addr uintptr)
// Called from instrumented code.
TEXT	runtime·raceread(SB), NOSPLIT, $0-8
	// void __tsan_read(ThreadState *thr, void *addr, void *pc);
	MOVD	$__tsan_read(SB), R1
	MOVD	addr+0(FP), R3
	MOVD	R14, R4
	JMP	racecalladdr<>(SB)

// func runtime·RaceRead(addr uintptr)
TEXT	runtime·RaceRead(SB), NOSPLIT, $0-8
	// This needs to be a tail call, because raceread reads caller pc.
	JMP	runtime·raceread(SB)

// func runtime·racereadpc(void *addr, void *callpc, void *pc)
TEXT	runtime·racereadpc(SB), NOSPLIT, $0-24
	// void __tsan_read_pc(ThreadState *thr, void *addr, void *callpc, void *pc);
	MOVD	$__tsan_read_pc(SB), R1
	LMG	addr+0(FP), R3, R5
	JMP	racecalladdr<>(SB)

// func runtime·racewrite(addr uintptr)
// Called from instrumented code.
TEXT	runtime·racewrite(SB), NOSPLIT, $0-8
	// void __tsan_write(ThreadState *thr, void *addr, void *pc);
	MOVD	$__tsan_write(SB), R1
	MOVD	addr+0(FP), R3
	MOVD	R14, R4
	JMP	racecalladdr<>(SB)

// func runtime·RaceWrite(addr uintptr)
TEXT	runtime·RaceWrite(SB), NOSPLIT, $0-8
	// This needs to be a tail call, because racewrite reads caller pc.
	JMP	runtime·racewrite(SB)

// func runtime·racewritepc(void *addr, void *callpc, void *pc)
TEXT	runtime·racewritepc(SB), NOSPLIT, $0-24
	// void __tsan_write_pc(ThreadState *thr, void *addr, void *callpc, void *pc);
	MOVD	$__tsan_write_pc(SB), R1
	LMG	addr+0(FP), R3, R5
	JMP	racecalladdr<>(SB)

// func runtime·racereadrange(addr, size uintptr)
// Called from instrumented code.
TEXT	runtime·racereadrange(SB), NOSPLIT, $0-16
	// void __tsan_read_range(ThreadState *thr, void *addr, uintptr size, void *pc);
	MOVD	$__tsan_read_range(SB), R1
	LMG	addr+0(FP), R3, R4
	MOVD	R14, R5
	JMP	racecalladdr<>(SB)

// func runtime·RaceReadRange(addr, size uintptr)
TEXT	runtime·RaceReadRange(SB), NOSPLIT, $0-16
	// This needs to be a tail call, because racereadrange reads caller pc.
	JMP	runtime·racereadrange(SB)

// func runtime·racereadrangepc1(void *addr, uintptr sz, void *pc)
TEXT	runtime·racereadrangepc1(SB), NOSPLIT, $0-24
	// void __tsan_read_range(ThreadState *thr, void *addr, uintptr size, void *pc);
	MOVD	$__tsan_read_range(SB), R1
	LMG	addr+0(FP), R3, R5
	// pc is an interceptor address, but TSan expects it to point to the
	// middle of an interceptor (see LLVM's SCOPED_INTERCEPTOR_RAW).
	ADD	$2, R5
	JMP	racecalladdr<>(SB)

// func runtime·racewriterange(addr, size uintptr)
// Called from instrumented code.
TEXT	runtime·racewriterange(SB), NOSPLIT, $0-16
	// void __tsan_write_range(ThreadState *thr, void *addr, uintptr size, void *pc);
	MOVD	$__tsan_write_range(SB), R1
	LMG	addr+0(FP), R3, R4
	MOVD	R14, R5
	JMP	racecalladdr<>(SB)

// func runtime·RaceWriteRange(addr, size uintptr)
TEXT	runtime·RaceWriteRange(SB), NOSPLIT, $0-16
	// This needs to be a tail call, because racewriterange reads caller pc.
	JMP	runtime·racewriterange(SB)

// func runtime·racewriterangepc1(void *addr, uintptr sz, void *pc)
TEXT	runtime·racewriterangepc1(SB), NOSPLIT, $0-24
	// void __tsan_write_range(ThreadState *thr, void *addr, uintptr size, void *pc);
	MOVD	$__tsan_write_range(SB), R1
	LMG	addr+0(FP), R3, R5
	// pc is an interceptor address, but TSan expects it to point to the
	// middle of an interceptor (see LLVM's SCOPED_INTERCEPTOR_RAW).
	ADD	$2, R5
	JMP	racecalladdr<>(SB)

// If R3 is out of range, do nothing. Otherwise, setup goroutine context and
// invoke racecall. Other arguments are already set.
TEXT	racecalladdr<>(SB), NOSPLIT, $0-0
	MOVD	runtime·racearenastart(SB), R0
	CMPUBLT	R3, R0, data			// Before racearena start?
	MOVD	runtime·racearenaend(SB), R0
	CMPUBLT	R3, R0, call			// Before racearena end?
data:
	MOVD	runtime·racedatastart(SB), R0
	CMPUBLT	R3, R0, ret			// Before racedata start?
	MOVD	runtime·racedataend(SB), R0
	CMPUBGE	R3, R0, ret			// At or after racedata end?
call:
	MOVD	g_racectx(g), R2
	JMP	racecall<>(SB)
ret:
	RET

// func runtime·racefuncenter(pc uintptr)
// Called from instrumented code.
TEXT	runtime·racefuncenter(SB), NOSPLIT, $0-8
	MOVD	callpc+0(FP), R3
	JMP	racefuncenter<>(SB)

// Common code for racefuncenter
// R3 = caller's return address
TEXT	racefuncenter<>(SB), NOSPLIT, $0-0
	// void __tsan_func_enter(ThreadState *thr, void *pc);
	MOVD	$__tsan_func_enter(SB), R1
	MOVD	g_racectx(g), R2
	BL	racecall<>(SB)
	RET

// func runtime·racefuncexit()
// Called from instrumented code.
TEXT	runtime·racefuncexit(SB), NOSPLIT, $0-0
	// void __tsan_func_exit(ThreadState *thr);
	MOVD	$__tsan_func_exit(SB), R1
	MOVD	g_racectx(g), R2
	JMP	racecall<>(SB)

// Atomic operations for sync/atomic package.

// Load

TEXT	sync∕atomic·LoadInt32(SB), NOSPLIT, $0-12
	GO_ARGS
	MOVD	$__tsan_go_atomic32_load(SB), R1
	BL	racecallatomic<>(SB)
	RET

TEXT	sync∕atomic·LoadInt64(SB), NOSPLIT, $0-16
	GO_ARGS
	MOVD	$__tsan_go_atomic64_load(SB), R1
	BL	racecallatomic<>(SB)
	RET

TEXT	sync∕atomic·LoadUint32(SB), NOSPLIT, $0-12
	GO_ARGS
	JMP	sync∕atomic·LoadInt32(SB)

TEXT	sync∕atomic·LoadUint64(SB), NOSPLIT, $0-16
	GO_ARGS
	JMP	sync∕atomic·LoadInt64(SB)

TEXT	sync∕atomic·LoadUintptr(SB), NOSPLIT, $0-16
	GO_ARGS
	JMP	sync∕atomic·LoadInt64(SB)

TEXT	sync∕atomic·LoadPointer(SB), NOSPLIT, $0-16
	GO_ARGS
	JMP	sync∕atomic·LoadInt64(SB)

// Store

TEXT	sync∕atomic·StoreInt32(SB), NOSPLIT, $0-12
	GO_ARGS
	MOVD	$__tsan_go_atomic32_store(SB), R1
	BL	racecallatomic<>(SB)
	RET

TEXT	sync∕atomic·StoreInt64(SB), NOSPLIT, $0-16
	GO_ARGS
	MOVD	$__tsan_go_atomic64_store(SB), R1
	BL	racecallatomic<>(SB)
	RET

TEXT	sync∕atomic·StoreUint32(SB), NOSPLIT, $0-12
	GO_ARGS
	JMP	sync∕atomic·StoreInt32(SB)

TEXT	sync∕atomic·StoreUint64(SB), NOSPLIT, $0-16
	GO_ARGS
	JMP	sync∕atomic·StoreInt64(SB)

TEXT	sync∕atomic·StoreUintptr(SB), NOSPLIT, $0-16
	GO_ARGS
	JMP	sync∕atomic·StoreInt64(SB)

// Swap

TEXT	sync∕atomic·SwapInt32(SB), NOSPLIT, $0-20
	GO_ARGS
	MOVD	$__tsan_go_atomic32_exchange(SB), R1
	BL	racecallatomic<>(SB)
	RET

TEXT	sync∕atomic·SwapInt64(SB), NOSPLIT, $0-24
	GO_ARGS
	MOVD	$__tsan_go_atomic64_exchange(SB), R1
	BL	racecallatomic<>(SB)
	RET

TEXT	sync∕atomic·SwapUint32(SB), NOSPLIT, $0-20
	GO_ARGS
	JMP	sync∕atomic·SwapInt32(SB)

TEXT	sync∕atomic·SwapUint64(SB), NOSPLIT, $0-24
	GO_ARGS
	JMP	sync∕atomic·SwapInt64(SB)

TEXT	sync∕atomic·SwapUintptr(SB), NOSPLIT, $0-24
	GO_ARGS
	JMP	sync∕atomic·SwapInt64(SB)

// Add

TEXT	sync∕atomic·AddInt32(SB), NOSPLIT, $0-20
	GO_ARGS
	MOVD	$__tsan_go_atomic32_fetch_add(SB), R1
	BL	racecallatomic<>(SB)
	// TSan performed fetch_add, but Go needs add_fetch.
	MOVW	add+8(FP), R0
	MOVW	ret+16(FP), R1
	ADD	R0, R1, R0
	MOVW	R0, ret+16(FP)
	RET

TEXT	sync∕atomic·AddInt64(SB), NOSPLIT, $0-24
	GO_ARGS
	MOVD	$__tsan_go_atomic64_fetch_add(SB), R1
	BL	racecallatomic<>(SB)
	// TSan performed fetch_add, but Go needs add_fetch.
	MOVD	add+8(FP), R0
	MOVD	ret+16(FP), R1
	ADD	R0, R1, R0
	MOVD	R0, ret+16(FP)
	RET

TEXT	sync∕atomic·AddUint32(SB), NOSPLIT, $0-20
	GO_ARGS
	JMP	sync∕atomic·AddInt32(SB)

TEXT	sync∕atomic·AddUint64(SB), NOSPLIT, $0-24
	GO_ARGS
	JMP	sync∕atomic·AddInt64(SB)

TEXT	sync∕atomic·AddUintptr(SB), NOSPLIT, $0-24
	GO_ARGS
	JMP	sync∕atomic·AddInt64(SB)

// CompareAndSwap

TEXT	sync∕atomic·CompareAndSwapInt32(SB), NOSPLIT, $0-17
	GO_ARGS
	MOVD	$__tsan_go_atomic32_compare_exchange(SB), R1
	BL	racecallatomic<>(SB)
	RET

TEXT	sync∕atomic·CompareAndSwapInt64(SB), NOSPLIT, $0-25
	GO_ARGS
	MOVD	$__tsan_go_atomic64_compare_exchange(SB), R1
	BL	racecallatomic<>(SB)
	RET

TEXT	sync∕atomic·CompareAndSwapUint32(SB), NOSPLIT, $0-17
	GO_ARGS
	JMP	sync∕atomic·CompareAndSwapInt32(SB)

TEXT	sync∕atomic·CompareAndSwapUint64(SB), NOSPLIT, $0-25
	GO_ARGS
	JMP	sync∕atomic·CompareAndSwapInt64(SB)

TEXT	sync∕atomic·CompareAndSwapUintptr(SB), NOSPLIT, $0-25
	GO_ARGS
	JMP	sync∕atomic·CompareAndSwapInt64(SB)

// Common code for atomic operations. Calls R1.
TEXT	racecallatomic<>(SB), NOSPLIT, $0
	MOVD	24(R15), R5			// Address (arg1, after 2xBL).
	// If we pass an invalid pointer to the TSan runtime, it will cause a
	// "fatal error: unknown caller pc". So trigger a SEGV here instead.
	MOVB	(R5), R0
	MOVD	runtime·racearenastart(SB), R0
	CMPUBLT	R5, R0, racecallatomic_data	// Before racearena start?
	MOVD	runtime·racearenaend(SB), R0
	CMPUBLT	R5, R0, racecallatomic_ok	// Before racearena end?
racecallatomic_data:
	MOVD	runtime·racedatastart(SB), R0
	CMPUBLT	R5, R0, racecallatomic_ignore	// Before racedata start?
	MOVD	runtime·racedataend(SB), R0
	CMPUBGE	R5, R0,	racecallatomic_ignore	// At or after racearena end?
racecallatomic_ok:
	MOVD	g_racectx(g), R2		// ThreadState *.
	MOVD	8(R15), R3			// Caller PC.
	MOVD	R14, R4				// PC.
	ADD	$24, R15, R5			// Arguments.
	// Tail call fails to restore R15, so use a normal one.
	BL	racecall<>(SB)
	RET
racecallatomic_ignore:
	// Call __tsan_go_ignore_sync_begin to ignore synchronization during
	// the atomic op. An attempt to synchronize on the address would cause
	// a crash.
	MOVD	R1, R6				// Save target function.
	MOVD	R14, R7				// Save PC.
	MOVD	$__tsan_go_ignore_sync_begin(SB), R1
	MOVD	g_racectx(g), R2		// ThreadState *.
	BL	racecall<>(SB)
	MOVD	R6, R1				// Restore target function.
	MOVD	g_racectx(g), R2		// ThreadState *.
	MOVD	8(R15), R3			// Caller PC.
	MOVD	R7, R4				// PC.
	ADD	$24, R15, R5			// Arguments.
	BL	racecall<>(SB)
	MOVD	$__tsan_go_ignore_sync_end(SB), R1
	MOVD	g_racectx(g), R2		// ThreadState *.
	BL	racecall<>(SB)
	RET

// func runtime·racecall(void(*f)(...), ...)
// Calls C function f from race runtime and passes up to 4 arguments to it.
// The arguments are never heap-object-preserving pointers, so we pretend there
// are no arguments.
TEXT	runtime·racecall(SB), NOSPLIT, $0-0
	MOVD	fn+0(FP), R1
	MOVD	arg0+8(FP), R2
	MOVD	arg1+16(FP), R3
	MOVD	arg2+24(FP), R4
	MOVD	arg3+32(FP), R5
	JMP	racecall<>(SB)

// Switches SP to g0 stack and calls R1. Arguments are already set.
TEXT	racecall<>(SB), NOSPLIT, $0-0
	BL	runtime·save_g(SB)		// Save g for callbacks.
	MOVD	R15, R7				// Save SP.
	MOVD	g_m(g), R8			// R8 = thread.
	MOVD	m_g0(R8), R8			// R8 = g0.
	CMPBEQ	R8, g, call			// Already on g0?
	MOVD	(g_sched+gobuf_sp)(R8), R15	// Switch SP to g0.
call:	SUB	$160, R15			// Allocate C frame.
	BL	R1				// Call C code.
	MOVD	R7, R15				// Restore SP.
	RET					// Return to Go.

// C->Go callback thunk that allows to call runtime·racesymbolize from C
// code. racecall has only switched SP, finish g->g0 switch by setting correct
// g. R2 contains command code, R3 contains command-specific context. See
// racecallback for command codes.
TEXT	runtime·racecallbackthunk(SB), NOSPLIT|NOFRAME, $0
	STMG	R6, R15, 48(R15)		// Save non-volatile regs.
	BL	runtime·load_g(SB)		// Saved by racecall.
	CMPBNE	R2, $0, rest			// raceGetProcCmd?
	MOVD	g_m(g), R2			// R2 = thread.
	MOVD	m_p(R2), R2			// R2 = processor.
	MVC	$8, p_raceprocctx(R2), (R3)	// *R3 = ThreadState *.
	LMG	48(R15), R6, R15		// Restore non-volatile regs.
	BR	R14				// Return to C.
rest:	MOVD	g_m(g), R4			// R4 = current thread.
	MOVD	m_g0(R4), g			// Switch to g0.
	SUB	$24, R15			// Allocate Go argument slots.
	STMG	R2, R3, 8(R15)			// Fill Go frame.
	BL	runtime·racecallback(SB)	// Call Go code.
	LMG	72(R15), R6, R15		// Restore non-volatile regs.
	BR	R14				// Return to C.
