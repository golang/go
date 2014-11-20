// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build race

#include "zasm_GOOS_GOARCH.h"
#include "funcdata.h"
#include "textflag.h"

// The following thunks allow calling the gcc-compiled race runtime directly
// from Go code without going all the way through cgo.
// First, it's much faster (up to 50% speedup for real Go programs).
// Second, it eliminates race-related special cases from cgocall and scheduler.
// Third, in long-term it will allow to remove cyclic runtime/race dependency on cmd/go.

// A brief recap of the amd64 calling convention.
// Arguments are passed in DI, SI, DX, CX, R8, R9, the rest is on stack.
// Callee-saved registers are: BX, BP, R12-R15.
// SP must be 16-byte aligned.
// On Windows:
// Arguments are passed in CX, DX, R8, R9, the rest is on stack.
// Callee-saved registers are: BX, BP, DI, SI, R12-R15.
// SP must be 16-byte aligned. Windows also requires "stack-backing" for the 4 register arguments:
// http://msdn.microsoft.com/en-us/library/ms235286.aspx
// We do not do this, because it seems to be intended for vararg/unprototyped functions.
// Gcc-compiled race runtime does not try to use that space.

#ifdef GOOS_windows
#define RARG0 CX
#define RARG1 DX
#define RARG2 R8
#define RARG3 R9
#else
#define RARG0 DI
#define RARG1 SI
#define RARG2 DX
#define RARG3 CX
#endif

// func runtime·raceread(addr uintptr)
// Called from instrumented code.
TEXT	runtime·raceread(SB), NOSPLIT, $0-8
	MOVQ	addr+0(FP), RARG1
	MOVQ	(SP), RARG2
	// void __tsan_read(ThreadState *thr, void *addr, void *pc);
	MOVQ	$__tsan_read(SB), AX
	JMP	racecalladdr<>(SB)

// func runtime·RaceRead(addr uintptr)
TEXT	runtime·RaceRead(SB), NOSPLIT, $0-8
	// This needs to be a tail call, because raceread reads caller pc.
	JMP	runtime·raceread(SB)

// void runtime·racereadpc(void *addr, void *callpc, void *pc)
TEXT	runtime·racereadpc(SB), NOSPLIT, $0-24
	MOVQ	addr+0(FP), RARG1
	MOVQ	callpc+8(FP), RARG2
	MOVQ	pc+16(FP), RARG3
	// void __tsan_read_pc(ThreadState *thr, void *addr, void *callpc, void *pc);
	MOVQ	$__tsan_read_pc(SB), AX
	JMP	racecalladdr<>(SB)

// func runtime·racewrite(addr uintptr)
// Called from instrumented code.
TEXT	runtime·racewrite(SB), NOSPLIT, $0-8
	MOVQ	addr+0(FP), RARG1
	MOVQ	(SP), RARG2
	// void __tsan_write(ThreadState *thr, void *addr, void *pc);
	MOVQ	$__tsan_write(SB), AX
	JMP	racecalladdr<>(SB)

// func runtime·RaceWrite(addr uintptr)
TEXT	runtime·RaceWrite(SB), NOSPLIT, $0-8
	// This needs to be a tail call, because racewrite reads caller pc.
	JMP	runtime·racewrite(SB)

// void runtime·racewritepc(void *addr, void *callpc, void *pc)
TEXT	runtime·racewritepc(SB), NOSPLIT, $0-24
	MOVQ	addr+0(FP), RARG1
	MOVQ	callpc+8(FP), RARG2
	MOVQ	pc+16(FP), RARG3
	// void __tsan_write_pc(ThreadState *thr, void *addr, void *callpc, void *pc);
	MOVQ	$__tsan_write_pc(SB), AX
	JMP	racecalladdr<>(SB)

// func runtime·racereadrange(addr, size uintptr)
// Called from instrumented code.
TEXT	runtime·racereadrange(SB), NOSPLIT, $0-16
	MOVQ	addr+0(FP), RARG1
	MOVQ	size+8(FP), RARG2
	MOVQ	(SP), RARG3
	// void __tsan_read_range(ThreadState *thr, void *addr, uintptr size, void *pc);
	MOVQ	$__tsan_read_range(SB), AX
	JMP	racecalladdr<>(SB)

// func runtime·RaceReadRange(addr, size uintptr)
TEXT	runtime·RaceReadRange(SB), NOSPLIT, $0-16
	// This needs to be a tail call, because racereadrange reads caller pc.
	JMP	runtime·racereadrange(SB)

// void runtime·racereadrangepc1(void *addr, uintptr sz, void *pc)
TEXT	runtime·racereadrangepc1(SB), NOSPLIT, $0-24
	MOVQ	addr+0(FP), RARG1
	MOVQ	size+8(FP), RARG2
	MOVQ	pc+16(FP), RARG3
	// void __tsan_read_range(ThreadState *thr, void *addr, uintptr size, void *pc);
	MOVQ	$__tsan_read_range(SB), AX
	JMP	racecalladdr<>(SB)

// func runtime·racewriterange(addr, size uintptr)
// Called from instrumented code.
TEXT	runtime·racewriterange(SB), NOSPLIT, $0-16
	MOVQ	addr+0(FP), RARG1
	MOVQ	size+8(FP), RARG2
	MOVQ	(SP), RARG3
	// void __tsan_write_range(ThreadState *thr, void *addr, uintptr size, void *pc);
	MOVQ	$__tsan_write_range(SB), AX
	JMP	racecalladdr<>(SB)

// func runtime·RaceWriteRange(addr, size uintptr)
TEXT	runtime·RaceWriteRange(SB), NOSPLIT, $0-16
	// This needs to be a tail call, because racewriterange reads caller pc.
	JMP	runtime·racewriterange(SB)

// void runtime·racewriterangepc1(void *addr, uintptr sz, void *pc)
TEXT	runtime·racewriterangepc1(SB), NOSPLIT, $0-24
	MOVQ	addr+0(FP), RARG1
	MOVQ	size+8(FP), RARG2
	MOVQ	pc+16(FP), RARG3
	// void __tsan_write_range(ThreadState *thr, void *addr, uintptr size, void *pc);
	MOVQ	$__tsan_write_range(SB), AX
	JMP	racecalladdr<>(SB)

// If addr (RARG1) is out of range, do nothing.
// Otherwise, setup goroutine context and invoke racecall. Other arguments already set.
TEXT	racecalladdr<>(SB), NOSPLIT, $0-0
	get_tls(R12)
	MOVQ	g(R12), R14
	MOVQ	g_racectx(R14), RARG0	// goroutine context
	// Check that addr is within [arenastart, arenaend) or within [racedatastart, racedataend).
	CMPQ	RARG1, runtime·racearenastart(SB)
	JB	racecalladdr_data
	CMPQ	RARG1, runtime·racearenaend(SB)
	JB	racecalladdr_call
racecalladdr_data:
	CMPQ	RARG1, runtime·racedatastart(SB)
	JB	racecalladdr_ret
	CMPQ	RARG1, runtime·racedataend(SB)
	JAE	racecalladdr_ret
racecalladdr_call:
	MOVQ	AX, AX		// w/o this 6a miscompiles this function
	JMP	racecall<>(SB)
racecalladdr_ret:
	RET

// func runtime·racefuncenter(pc uintptr)
// Called from instrumented code.
TEXT	runtime·racefuncenter(SB), NOSPLIT, $0-8
	MOVQ	DX, R15		// save function entry context (for closures)
	get_tls(R12)
	MOVQ	g(R12), R14
	MOVQ	g_racectx(R14), RARG0	// goroutine context
	MOVQ	callpc+0(FP), RARG1
	// void __tsan_func_enter(ThreadState *thr, void *pc);
	MOVQ	$__tsan_func_enter(SB), AX
	// racecall<> preserves R15
	CALL	racecall<>(SB)
	MOVQ	R15, DX	// restore function entry context
	RET

// func runtime·racefuncexit()
// Called from instrumented code.
TEXT	runtime·racefuncexit(SB), NOSPLIT, $0-0
	get_tls(R12)
	MOVQ	g(R12), R14
	MOVQ	g_racectx(R14), RARG0	// goroutine context
	// void __tsan_func_exit(ThreadState *thr);
	MOVQ	$__tsan_func_exit(SB), AX
	JMP	racecall<>(SB)

// Atomic operations for sync/atomic package.

// Load
TEXT	sync∕atomic·LoadInt32(SB), NOSPLIT, $0-0
	MOVQ	$__tsan_go_atomic32_load(SB), AX
	CALL	racecallatomic<>(SB)
	RET

TEXT	sync∕atomic·LoadInt64(SB), NOSPLIT, $0-0
	MOVQ	$__tsan_go_atomic64_load(SB), AX
	CALL	racecallatomic<>(SB)
	RET

TEXT	sync∕atomic·LoadUint32(SB), NOSPLIT, $0-0
	JMP	sync∕atomic·LoadInt32(SB)

TEXT	sync∕atomic·LoadUint64(SB), NOSPLIT, $0-0
	JMP	sync∕atomic·LoadInt64(SB)

TEXT	sync∕atomic·LoadUintptr(SB), NOSPLIT, $0-0
	JMP	sync∕atomic·LoadInt64(SB)

TEXT	sync∕atomic·LoadPointer(SB), NOSPLIT, $0-0
	JMP	sync∕atomic·LoadInt64(SB)

// Store
TEXT	sync∕atomic·StoreInt32(SB), NOSPLIT, $0-0
	MOVQ	$__tsan_go_atomic32_store(SB), AX
	CALL	racecallatomic<>(SB)
	RET

TEXT	sync∕atomic·StoreInt64(SB), NOSPLIT, $0-0
	MOVQ	$__tsan_go_atomic64_store(SB), AX
	CALL	racecallatomic<>(SB)
	RET

TEXT	sync∕atomic·StoreUint32(SB), NOSPLIT, $0-0
	JMP	sync∕atomic·StoreInt32(SB)

TEXT	sync∕atomic·StoreUint64(SB), NOSPLIT, $0-0
	JMP	sync∕atomic·StoreInt64(SB)

TEXT	sync∕atomic·StoreUintptr(SB), NOSPLIT, $0-0
	JMP	sync∕atomic·StoreInt64(SB)

TEXT	sync∕atomic·StorePointer(SB), NOSPLIT, $0-0
	JMP	sync∕atomic·StoreInt64(SB)

// Swap
TEXT	sync∕atomic·SwapInt32(SB), NOSPLIT, $0-0
	MOVQ	$__tsan_go_atomic32_exchange(SB), AX
	CALL	racecallatomic<>(SB)
	RET

TEXT	sync∕atomic·SwapInt64(SB), NOSPLIT, $0-0
	MOVQ	$__tsan_go_atomic64_exchange(SB), AX
	CALL	racecallatomic<>(SB)
	RET

TEXT	sync∕atomic·SwapUint32(SB), NOSPLIT, $0-0
	JMP	sync∕atomic·SwapInt32(SB)

TEXT	sync∕atomic·SwapUint64(SB), NOSPLIT, $0-0
	JMP	sync∕atomic·SwapInt64(SB)

TEXT	sync∕atomic·SwapUintptr(SB), NOSPLIT, $0-0
	JMP	sync∕atomic·SwapInt64(SB)

TEXT	sync∕atomic·SwapPointer(SB), NOSPLIT, $0-0
	JMP	sync∕atomic·SwapInt64(SB)

// Add
TEXT	sync∕atomic·AddInt32(SB), NOSPLIT, $0-0
	MOVQ	$__tsan_go_atomic32_fetch_add(SB), AX
	CALL	racecallatomic<>(SB)
	MOVL	add+8(FP), AX	// convert fetch_add to add_fetch
	ADDL	AX, ret+16(FP)
	RET

TEXT	sync∕atomic·AddInt64(SB), NOSPLIT, $0-0
	MOVQ	$__tsan_go_atomic64_fetch_add(SB), AX
	CALL	racecallatomic<>(SB)
	MOVQ	add+8(FP), AX	// convert fetch_add to add_fetch
	ADDQ	AX, ret+16(FP)
	RET

TEXT	sync∕atomic·AddUint32(SB), NOSPLIT, $0-0
	JMP	sync∕atomic·AddInt32(SB)

TEXT	sync∕atomic·AddUint64(SB), NOSPLIT, $0-0
	JMP	sync∕atomic·AddInt64(SB)

TEXT	sync∕atomic·AddUintptr(SB), NOSPLIT, $0-0
	JMP	sync∕atomic·AddInt64(SB)

TEXT	sync∕atomic·AddPointer(SB), NOSPLIT, $0-0
	JMP	sync∕atomic·AddInt64(SB)

// CompareAndSwap
TEXT	sync∕atomic·CompareAndSwapInt32(SB), NOSPLIT, $0-0
	MOVQ	$__tsan_go_atomic32_compare_exchange(SB), AX
	CALL	racecallatomic<>(SB)
	RET

TEXT	sync∕atomic·CompareAndSwapInt64(SB), NOSPLIT, $0-0
	MOVQ	$__tsan_go_atomic64_compare_exchange(SB), AX
	CALL	racecallatomic<>(SB)
	RET

TEXT	sync∕atomic·CompareAndSwapUint32(SB), NOSPLIT, $0-0
	JMP	sync∕atomic·CompareAndSwapInt32(SB)

TEXT	sync∕atomic·CompareAndSwapUint64(SB), NOSPLIT, $0-0
	JMP	sync∕atomic·CompareAndSwapInt64(SB)

TEXT	sync∕atomic·CompareAndSwapUintptr(SB), NOSPLIT, $0-0
	JMP	sync∕atomic·CompareAndSwapInt64(SB)

TEXT	sync∕atomic·CompareAndSwapPointer(SB), NOSPLIT, $0-0
	JMP	sync∕atomic·CompareAndSwapInt64(SB)

// Generic atomic operation implementation.
// AX already contains target function.
TEXT	racecallatomic<>(SB), NOSPLIT, $0-0
	// Trigger SIGSEGV early.
	MOVQ	16(SP), R12
	MOVL	(R12), R13
	// Check that addr is within [arenastart, arenaend) or within [racedatastart, racedataend).
	CMPQ	R12, runtime·racearenastart(SB)
	JB	racecallatomic_data
	CMPQ	R12, runtime·racearenaend(SB)
	JB	racecallatomic_ok
racecallatomic_data:
	CMPQ	R12, runtime·racedatastart(SB)
	JB	racecallatomic_ignore
	CMPQ	R12, runtime·racedataend(SB)
	JAE	racecallatomic_ignore
racecallatomic_ok:
	// Addr is within the good range, call the atomic function.
	get_tls(R12)
	MOVQ	g(R12), R14
	MOVQ	g_racectx(R14), RARG0	// goroutine context
	MOVQ	8(SP), RARG1	// caller pc
	MOVQ	(SP), RARG2	// pc
	LEAQ	16(SP), RARG3	// arguments
	JMP	racecall<>(SB)	// does not return
racecallatomic_ignore:
	// Addr is outside the good range.
	// Call __tsan_go_ignore_sync_begin to ignore synchronization during the atomic op.
	// An attempt to synchronize on the address would cause crash.
	MOVQ	AX, R15	// remember the original function
	MOVQ	$__tsan_go_ignore_sync_begin(SB), AX
	MOVQ	g(R12), R14
	MOVQ	g_racectx(R14), RARG0	// goroutine context
	CALL	racecall<>(SB)
	MOVQ	R15, AX	// restore the original function
	// Call the atomic function.
	MOVQ	g_racectx(R14), RARG0	// goroutine context
	MOVQ	8(SP), RARG1	// caller pc
	MOVQ	(SP), RARG2	// pc
	LEAQ	16(SP), RARG3	// arguments
	CALL	racecall<>(SB)
	// Call __tsan_go_ignore_sync_end.
	MOVQ	$__tsan_go_ignore_sync_end(SB), AX
	MOVQ	g_racectx(R14), RARG0	// goroutine context
	JMP	racecall<>(SB)

// void runtime·racecall(void(*f)(...), ...)
// Calls C function f from race runtime and passes up to 4 arguments to it.
// The arguments are never heap-object-preserving pointers, so we pretend there are no arguments.
TEXT	runtime·racecall(SB), NOSPLIT, $0-0
	MOVQ	fn+0(FP), AX
	MOVQ	arg0+8(FP), RARG0
	MOVQ	arg1+16(FP), RARG1
	MOVQ	arg2+24(FP), RARG2
	MOVQ	arg3+32(FP), RARG3
	JMP	racecall<>(SB)

// Switches SP to g0 stack and calls (AX). Arguments already set.
TEXT	racecall<>(SB), NOSPLIT, $0-0
	get_tls(R12)
	MOVQ	g(R12), R14
	MOVQ	g_m(R14), R13
	// Switch to g0 stack.
	MOVQ	SP, R12		// callee-saved, preserved across the CALL
	MOVQ	m_g0(R13), R10
	CMPQ	R10, R14
	JE	racecall_cont	// already on g0
	MOVQ	(g_sched+gobuf_sp)(R10), SP
racecall_cont:
	ANDQ	$~15, SP	// alignment for gcc ABI
	CALL	AX
	MOVQ	R12, SP
	RET

// C->Go callback thunk that allows to call runtime·racesymbolize from C code.
// Direct Go->C race call has only switched SP, finish g->g0 switch by setting correct g.
// The overall effect of Go->C->Go call chain is similar to that of mcall.
TEXT	runtime·racesymbolizethunk(SB), NOSPLIT, $56-8
	// Save callee-saved registers (Go code won't respect that).
	// This is superset of darwin/linux/windows registers.
	PUSHQ	BX
	PUSHQ	BP
	PUSHQ	DI
	PUSHQ	SI
	PUSHQ	R12
	PUSHQ	R13
	PUSHQ	R14
	PUSHQ	R15
	// Set g = g0.
	get_tls(R12)
	MOVQ	g(R12), R13
	MOVQ	g_m(R13), R13
	MOVQ	m_g0(R13), R14
	MOVQ	R14, g(R12)	// g = m->g0
	MOVQ	RARG0, 0(SP)	// func arg
	CALL	runtime·racesymbolize(SB)
	// All registers are smashed after Go code, reload.
	get_tls(R12)
	MOVQ	g(R12), R13
	MOVQ	g_m(R13), R13
	MOVQ	m_curg(R13), R14
	MOVQ	R14, g(R12)	// g = m->curg
	// Restore callee-saved registers.
	POPQ	R15
	POPQ	R14
	POPQ	R13
	POPQ	R12
	POPQ	SI
	POPQ	DI
	POPQ	BP
	POPQ	BX
	RET
