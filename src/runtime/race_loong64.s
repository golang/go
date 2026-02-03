// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build race

#include "go_asm.h"
#include "funcdata.h"
#include "textflag.h"
#include "cgo/abi_loong64.h"

// The following thunks allow calling the gcc-compiled race runtime directly
// from Go code without going all the way through cgo.
// First, it's much faster (up to 50% speedup for real Go programs).
// Second, it eliminates race-related special cases from cgocall and scheduler.
// Third, in long-term it will allow to remove cyclic runtime/race dependency on cmd/go.

// A brief recap of the loong64 calling convention.
// Arguments are passed in R4...R11, the rest is on stack.
// Callee-saved registers are: R23...R30.
// Temporary registers are: R12...R20
// SP must be 16-byte aligned.

// When calling racecalladdr, R20 is the call target address.

// The race ctx, ThreadState *thr below, is passed in R4 and loaded in racecalladdr.

// Load g from TLS. (See tls_loong64.s)
#define load_g \
	MOVV	runtime·tls_g(SB), g

#define RARG0	R4
#define RARG1	R5
#define RARG2	R6
#define RARG3	R7
#define RCALL	R20

// func runtime·raceread(addr uintptr)
// Called from instrumented code.
// Defined as ABIInternal so as to avoid introducing a wrapper,
// which would make caller's PC ineffective.
TEXT	runtime·raceread<ABIInternal>(SB), NOSPLIT, $0-8
	MOVV	R4, RARG1
	MOVV	R1, RARG2
	// void __tsan_read(ThreadState *thr, void *addr, void *pc);
	MOVV	$__tsan_read(SB), RCALL
	JMP	racecalladdr<>(SB)

// func runtime·RaceRead(addr uintptr)
TEXT	runtime·RaceRead(SB), NOSPLIT, $0-8
	// This needs to be a tail call, because raceread reads caller pc.
	JMP	runtime·raceread(SB)

// func runtime·racereadpc(void *addr, void *callpc, void *pc)
TEXT	runtime·racereadpc(SB), NOSPLIT, $0-24
	MOVV	addr+0(FP), RARG1
	MOVV	callpc+8(FP), RARG2
	MOVV	pc+16(FP), RARG3
	// void __tsan_read_pc(ThreadState *thr, void *addr, void *callpc, void *pc);
	MOVV	$__tsan_read_pc(SB), RCALL
	JMP	racecalladdr<>(SB)

// func runtime·racewrite(addr uintptr)
// Called from instrumented code.
// Defined as ABIInternal so as to avoid introducing a wrapper,
// which would make caller's PC ineffective.
TEXT	runtime·racewrite<ABIInternal>(SB), NOSPLIT, $0-8
	MOVV	R4, RARG1
	MOVV	R1, RARG2
	// void __tsan_write(ThreadState *thr, void *addr, void *pc);
	MOVV	$__tsan_write(SB), RCALL
	JMP	racecalladdr<>(SB)

// func runtime·RaceWrite(addr uintptr)
TEXT	runtime·RaceWrite(SB), NOSPLIT, $0-8
	// This needs to be a tail call, because racewrite reads caller pc.
	JMP	runtime·racewrite(SB)

// func runtime·racewritepc(void *addr, void *callpc, void *pc)
TEXT	runtime·racewritepc(SB), NOSPLIT, $0-24
	MOVV	addr+0(FP), RARG1
	MOVV	callpc+8(FP), RARG2
	MOVV	pc+16(FP), RARG3
	// void __tsan_write_pc(ThreadState *thr, void *addr, void *callpc, void *pc);
	MOVV	$__tsan_write_pc(SB), RCALL
	JMP	racecalladdr<>(SB)

// func runtime·racereadrange(addr, size uintptr)
// Called from instrumented code.
// Defined as ABIInternal so as to avoid introducing a wrapper,
// which would make caller's PC ineffective.
TEXT	runtime·racereadrange<ABIInternal>(SB), NOSPLIT, $0-16
	MOVV	R5, RARG2
	MOVV	R4, RARG1
	MOVV	R1, RARG3
	// void __tsan_read_range(ThreadState *thr, void *addr, uintptr size, void *pc);
	MOVV	$__tsan_read_range(SB), RCALL
	JMP	racecalladdr<>(SB)

// func runtime·RaceReadRange(addr, size uintptr)
TEXT	runtime·RaceReadRange(SB), NOSPLIT, $0-16
	// This needs to be a tail call, because racereadrange reads caller pc.
	JMP	runtime·racereadrange(SB)

// func runtime·racereadrangepc1(void *addr, uintptr sz, void *pc)
TEXT	runtime·racereadrangepc1(SB), NOSPLIT, $0-24
	MOVV	addr+0(FP), RARG1
	MOVV	size+8(FP), RARG2
	MOVV	pc+16(FP), RARG3
	ADDV	$4, RARG3	// pc is function start, tsan wants return address.
	// void __tsan_read_range(ThreadState *thr, void *addr, uintptr size, void *pc);
	MOVV	$__tsan_read_range(SB), RCALL
	JMP	racecalladdr<>(SB)

// func runtime·racewriterange(addr, size uintptr)
// Called from instrumented code.
// Defined as ABIInternal so as to avoid introducing a wrapper,
// which would make caller's PC ineffective.
TEXT	runtime·racewriterange<ABIInternal>(SB), NOSPLIT, $0-16
	MOVV	R5, RARG2
	MOVV	R4, RARG1
	MOVV	R1, RARG3
	// void __tsan_write_range(ThreadState *thr, void *addr, uintptr size, void *pc);
	MOVV	$__tsan_write_range(SB), RCALL
	JMP	racecalladdr<>(SB)

// func runtime·RaceWriteRange(addr, size uintptr)
TEXT	runtime·RaceWriteRange(SB), NOSPLIT, $0-16
	// This needs to be a tail call, because racewriterange reads caller pc.
	JMP	runtime·racewriterange(SB)

// func runtime·racewriterangepc1(void *addr, uintptr sz, void *pc)
TEXT	runtime·racewriterangepc1(SB), NOSPLIT, $0-24
	MOVV	addr+0(FP), RARG1
	MOVV	size+8(FP), RARG2
	MOVV	pc+16(FP), RARG3
	ADDV	$4, RARG3	// pc is function start, tsan wants return address.
	// void __tsan_write_range(ThreadState *thr, void *addr, uintptr size, void *pc);
	MOVV	$__tsan_write_range(SB), RCALL
	JMP	racecalladdr<>(SB)

// Call a __tsan function from Go code.
//
// RCALL = tsan function address
// RARG0 = *ThreadState a.k.a. g_racectx from g
// RARG1 = addr passed to __tsan function
//
// If addr (RARG1) is out of range, do nothing. Otherwise, setup goroutine
// context and invoke racecall. Other arguments already set.
TEXT	racecalladdr<>(SB), NOSPLIT, $0-0
	// Check that addr is within [arenastart, arenaend) or within [racedatastart, racedataend).
	MOVV	runtime·racearenastart(SB), R12
	BLT	RARG1, R12, data
	MOVV	runtime·racearenaend(SB), R12
	BLT	RARG1, R12, call
data:
	MOVV	runtime·racedatastart(SB), R12
	BLT	RARG1, R12, ret
	MOVV	runtime·racedataend(SB), R12
	BGE	RARG1, R12, ret
call:
	load_g
	MOVV	g_racectx(g), RARG0
	JMP	racecall<>(SB)
ret:
	RET

// func runtime·racefuncenter(pc uintptr)
// Called from instrumented code.
TEXT	runtime·racefuncenter<ABIInternal>(SB), NOSPLIT, $0-8
	MOVV	R4, RCALL
	JMP	racefuncenter<>(SB)

// Common code for racefuncenter
// RCALL = caller's return address
TEXT	racefuncenter<>(SB), NOSPLIT, $0-0
	load_g
	MOVV	g_racectx(g), RARG0	// goroutine racectx
	MOVV	RCALL, RARG1
	// void __tsan_func_enter(ThreadState *thr, void *pc);
	MOVV	$__tsan_func_enter(SB), RCALL
	JAL	racecall<>(SB)
	RET

// func runtime·racefuncexit()
// Called from instrumented code.
TEXT	runtime·racefuncexit<ABIInternal>(SB), NOSPLIT, $0-0
	load_g
	MOVV	g_racectx(g), RARG0	// race context
	// void __tsan_func_exit(ThreadState *thr);
	MOVV	$__tsan_func_exit(SB), RCALL
	JMP	racecall<>(SB)

// Atomic operations for sync/atomic package.
// R7 = addr of arguments passed to this function, it can
// be fetched at 24(R3) in racecallatomic after two times JAL
// RARG0, RARG1, RARG2 set in racecallatomic

// Load
TEXT	sync∕atomic·LoadInt32(SB), NOSPLIT, $0-12
	GO_ARGS
	MOVV	$__tsan_go_atomic32_load(SB), RCALL
	JAL	racecallatomic<>(SB)
	RET

TEXT	sync∕atomic·LoadInt64(SB), NOSPLIT, $0-16
	GO_ARGS
	MOVV	$__tsan_go_atomic64_load(SB), RCALL
	JAL	racecallatomic<>(SB)
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
	MOVV	$__tsan_go_atomic32_store(SB), RCALL
	JAL	racecallatomic<>(SB)
	RET

TEXT	sync∕atomic·StoreInt64(SB), NOSPLIT, $0-16
	GO_ARGS
	MOVV	$__tsan_go_atomic64_store(SB), RCALL
	JAL	racecallatomic<>(SB)
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
	MOVV	$__tsan_go_atomic32_exchange(SB), RCALL
	JAL	racecallatomic<>(SB)
	RET

TEXT	sync∕atomic·SwapInt64(SB), NOSPLIT, $0-24
	GO_ARGS
	MOVV	$__tsan_go_atomic64_exchange(SB), RCALL
	JAL	racecallatomic<>(SB)
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
	MOVV	$__tsan_go_atomic32_fetch_add(SB), RCALL
	JAL	racecallatomic<>(SB)
	MOVW	add+8(FP), RARG0	// convert fetch_add to add_fetch
	MOVW	ret+16(FP), RARG1
	ADD	RARG0, RARG1, RARG0
	MOVW	RARG0, ret+16(FP)
	RET

TEXT	sync∕atomic·AddInt64(SB), NOSPLIT, $0-24
	GO_ARGS
	MOVV	$__tsan_go_atomic64_fetch_add(SB), RCALL
	JAL	racecallatomic<>(SB)
	MOVV	add+8(FP), RARG0	// convert fetch_add to add_fetch
	MOVV	ret+16(FP), RARG1
	ADDV	RARG0, RARG1, RARG0
	MOVV	RARG0, ret+16(FP)
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

// And
TEXT	sync∕atomic·AndInt32(SB), NOSPLIT, $0-20
	GO_ARGS
	MOVV	$__tsan_go_atomic32_fetch_and(SB), RCALL
	JAL	racecallatomic<>(SB)
	RET

TEXT	sync∕atomic·AndInt64(SB), NOSPLIT, $0-24
	GO_ARGS
	MOVV	$__tsan_go_atomic64_fetch_and(SB), RCALL
	JAL	racecallatomic<>(SB)
	RET

TEXT	sync∕atomic·AndUint32(SB), NOSPLIT, $0-20
	GO_ARGS
	JMP	sync∕atomic·AndInt32(SB)

TEXT	sync∕atomic·AndUint64(SB), NOSPLIT, $0-24
	GO_ARGS
	JMP	sync∕atomic·AndInt64(SB)

TEXT	sync∕atomic·AndUintptr(SB), NOSPLIT, $0-24
	GO_ARGS
	JMP	sync∕atomic·AndInt64(SB)

// Or
TEXT	sync∕atomic·OrInt32(SB), NOSPLIT, $0-20
	GO_ARGS
	MOVV	$__tsan_go_atomic32_fetch_or(SB), RCALL
	JAL	racecallatomic<>(SB)
	RET

TEXT	sync∕atomic·OrInt64(SB), NOSPLIT, $0-24
	GO_ARGS
	MOVV	$__tsan_go_atomic64_fetch_or(SB), RCALL
	JAL	racecallatomic<>(SB)
	RET

TEXT	sync∕atomic·OrUint32(SB), NOSPLIT, $0-20
	GO_ARGS
	JMP	sync∕atomic·OrInt32(SB)

TEXT	sync∕atomic·OrUint64(SB), NOSPLIT, $0-24
	GO_ARGS
	JMP	sync∕atomic·OrInt64(SB)

TEXT	sync∕atomic·OrUintptr(SB), NOSPLIT, $0-24
	GO_ARGS
	JMP	sync∕atomic·OrInt64(SB)

// CompareAndSwap
TEXT	sync∕atomic·CompareAndSwapInt32(SB), NOSPLIT, $0-17
	GO_ARGS
	MOVV	$__tsan_go_atomic32_compare_exchange(SB), RCALL
	JAL	racecallatomic<>(SB)
	RET

TEXT	sync∕atomic·CompareAndSwapInt64(SB), NOSPLIT, $0-25
	GO_ARGS
	MOVV	$__tsan_go_atomic64_compare_exchange(SB), RCALL
	JAL	racecallatomic<>(SB)
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

// Generic atomic operation implementation.
// RCALL = addr of target function
TEXT	racecallatomic<>(SB), NOSPLIT, $0
	// Set up these registers
	// RARG0 = *ThreadState
	// RARG1 = caller pc
	// RARG2 = pc
	// RARG3 = addr of incoming arg list

	// Trigger SIGSEGV early.
	MOVV	24(R3), RARG3	// 1st arg is addr. after two times JAL, get it at 24(R3)
	MOVB	(RARG3), R12	// segv here if addr is bad

	// Check that addr is within [arenastart, arenaend) or within [racedatastart, racedataend).
	MOVV	runtime·racearenastart(SB), R12
	BLT	RARG3, R12, racecallatomic_data
	MOVV	runtime·racearenaend(SB), R12
	BLT	RARG3, R12, racecallatomic_ok

racecallatomic_data:
	MOVV	runtime·racedatastart(SB), R12
	BLT	RARG3, R12, racecallatomic_ignore
	MOVV	runtime·racedataend(SB), R12
	BGE	RARG3, R12, racecallatomic_ignore

racecallatomic_ok:
	// Addr is within the good range, call the atomic function.
	load_g
	MOVV	g_racectx(g), RARG0	// goroutine context
	MOVV	8(R3), RARG1	// caller pc
	MOVV	RCALL, RARG2	// pc
	ADDV	$24, R3, RARG3
	JAL	racecall<>(SB)	// does not return
	RET

racecallatomic_ignore:
	// Addr is outside the good range.
	// Call __tsan_go_ignore_sync_begin to ignore synchronization during the atomic op.
	// An attempt to synchronize on the address would cause crash.
	MOVV	RCALL, R25	// remember the original function
	MOVV	$__tsan_go_ignore_sync_begin(SB), RCALL
	load_g
	MOVV	g_racectx(g), RARG0	// goroutine context
	JAL	racecall<>(SB)
	MOVV	R25, RCALL	// restore the original function

	// Call the atomic function.
	// racecall will call LLVM race code which might clobber R22 (g)
	load_g
	MOVV	g_racectx(g), RARG0	// goroutine context
	MOVV	8(R3), RARG1	// caller pc
	MOVV	RCALL, RARG2	// pc
	ADDV	$24, R3, RARG3	// arguments
	JAL	racecall<>(SB)

	// Call __tsan_go_ignore_sync_end.
	MOVV	$__tsan_go_ignore_sync_end(SB), RCALL
	MOVV	g_racectx(g), RARG0	// goroutine context
	JAL	racecall<>(SB)
	RET

// func runtime·racecall(void(*f)(...), ...)
// Calls C function f from race runtime and passes up to 4 arguments to it.
// The arguments are never heap-object-preserving pointers, so we pretend there are no arguments.
TEXT	runtime·racecall(SB), NOSPLIT, $0-0
	MOVV	fn+0(FP), RCALL
	MOVV	arg0+8(FP), RARG0
	MOVV	arg1+16(FP), RARG1
	MOVV	arg2+24(FP), RARG2
	MOVV	arg3+32(FP), RARG3
	JMP	racecall<>(SB)

// Switches SP to g0 stack and calls (RCALL). Arguments already set.
TEXT	racecall<>(SB), NOSPLIT|NOFRAME, $0-0
	MOVV	g_m(g), R12
	// Switch to g0 stack.
	MOVV	R3, R23	// callee-saved, preserved across the CALL
	MOVV	R1, R24	// callee-saved, preserved across the CALL

	// Switch to g0 stack if we aren't already on g0 or gsignal.
	MOVV	m_gsignal(R12), R13
	BEQ	R13, g, call

	MOVV	m_g0(R12), R13
	BEQ	R13, g, call
	MOVV	(g_sched+gobuf_sp)(R13), R3
call:
	JAL	(RCALL)
	MOVV	R23, R3
	JAL	(R24)
	RET

// C->Go callback thunk that allows to call runtime·racesymbolize from C code.
// Direct Go->C race call has only switched SP, finish g->g0 switch by setting correct g.
// The overall effect of Go->C->Go call chain is similar to that of mcall.
// RARG0 contains command code. RARG1 contains command-specific context.
// See racecallback for command codes.
TEXT	runtime·racecallbackthunk(SB), NOSPLIT|NOFRAME, $0
	// Handle command raceGetProcCmd (0) here.
	// First, code below assumes that we are on curg, while raceGetProcCmd
	// can be executed on g0. Second, it is called frequently, so will
	// benefit from this fast path.
	BNE	RARG0, R0, rest
	MOVV	g, R15
	load_g
	MOVV	g_m(g), RARG0
	MOVV	m_p(RARG0), RARG0
	MOVV	p_raceprocctx(RARG0), RARG0
	MOVV	RARG0, (RARG1)
	MOVV	R15, g
	JMP	(R1)
rest:
	// Save callee-saved registers (Go code won't respect that).
	// 8(R3) and 16(R3) are for args passed through racecallback
	ADDV	$-176, R3
	MOVV	R1, 0(R3)

	SAVE_R22_TO_R31(8*3)
	SAVE_F24_TO_F31(8*13)
	// Set g = g0.
	load_g
	MOVV	g_m(g), R15
	MOVV	m_g0(R15), R14
	BEQ	R14, g, noswitch	// branch if already on g0
	MOVV	R14, g

	JAL	runtime·racecallback<ABIInternal>(SB)
	// All registers are smashed after Go code, reload.
	MOVV	g_m(g), R15
	MOVV	m_curg(R15), g	// g = m->curg
ret:
	// Restore callee-saved registers.
	MOVV	0(R3), R1
	RESTORE_F24_TO_F31(8*13)
	RESTORE_R22_TO_R31(8*3)
	ADDV	$176, R3
	JMP	(R1)

noswitch:
	// already on g0
	JAL	runtime·racecallback<ABIInternal>(SB)
	JMP	ret

// tls_g, g value for each thread in TLS
GLOBL runtime·tls_g+0(SB), TLSBSS+DUPOK, $8
