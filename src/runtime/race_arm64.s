// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build race

#include "go_asm.h"
#include "funcdata.h"
#include "textflag.h"
#include "tls_arm64.h"
#include "cgo/abi_arm64.h"

// The following thunks allow calling the gcc-compiled race runtime directly
// from Go code without going all the way through cgo.
// First, it's much faster (up to 50% speedup for real Go programs).
// Second, it eliminates race-related special cases from cgocall and scheduler.
// Third, in long-term it will allow to remove cyclic runtime/race dependency on cmd/go.

// A brief recap of the arm64 calling convention.
// Arguments are passed in R0...R7, the rest is on stack.
// Callee-saved registers are: R19...R28.
// Temporary registers are: R9...R15
// SP must be 16-byte aligned.

// When calling racecalladdr, R9 is the call target address.

// The race ctx, ThreadState *thr below, is passed in R0 and loaded in racecalladdr.

// Darwin may return unaligned thread pointer. Align it. (See tls_arm64.s)
// No-op on other OSes.
#ifdef TLS_darwin
#define TP_ALIGN	AND	$~7, R0
#else
#define TP_ALIGN
#endif

// Load g from TLS. (See tls_arm64.s)
#define load_g \
	MRS_TPIDR_R0 \
	TP_ALIGN \
	MOVD    runtime·tls_g(SB), R11 \
	MOVD    (R0)(R11), g

// func runtime·raceread(addr uintptr)
// Called from instrumented code.
// Defined as ABIInternal so as to avoid introducing a wrapper,
// which would make caller's PC ineffective.
TEXT	runtime·raceread<ABIInternal>(SB), NOSPLIT, $0-8
	MOVD	R0, R1	// addr
	MOVD	LR, R2
	// void __tsan_read(ThreadState *thr, void *addr, void *pc);
	MOVD	$__tsan_read(SB), R9
	JMP	racecalladdr<>(SB)

// func runtime·RaceRead(addr uintptr)
TEXT	runtime·RaceRead(SB), NOSPLIT, $0-8
	// This needs to be a tail call, because raceread reads caller pc.
	JMP	runtime·raceread(SB)

// func runtime·racereadpc(void *addr, void *callpc, void *pc)
TEXT	runtime·racereadpc(SB), NOSPLIT, $0-24
	MOVD	addr+0(FP), R1
	MOVD	callpc+8(FP), R2
	MOVD	pc+16(FP), R3
	// void __tsan_read_pc(ThreadState *thr, void *addr, void *callpc, void *pc);
	MOVD	$__tsan_read_pc(SB), R9
	JMP	racecalladdr<>(SB)

// func runtime·racewrite(addr uintptr)
// Called from instrumented code.
// Defined as ABIInternal so as to avoid introducing a wrapper,
// which would make caller's PC ineffective.
TEXT	runtime·racewrite<ABIInternal>(SB), NOSPLIT, $0-8
	MOVD	R0, R1	// addr
	MOVD	LR, R2
	// void __tsan_write(ThreadState *thr, void *addr, void *pc);
	MOVD	$__tsan_write(SB), R9
	JMP	racecalladdr<>(SB)

// func runtime·RaceWrite(addr uintptr)
TEXT	runtime·RaceWrite(SB), NOSPLIT, $0-8
	// This needs to be a tail call, because racewrite reads caller pc.
	JMP	runtime·racewrite(SB)

// func runtime·racewritepc(void *addr, void *callpc, void *pc)
TEXT	runtime·racewritepc(SB), NOSPLIT, $0-24
	MOVD	addr+0(FP), R1
	MOVD	callpc+8(FP), R2
	MOVD	pc+16(FP), R3
	// void __tsan_write_pc(ThreadState *thr, void *addr, void *callpc, void *pc);
	MOVD	$__tsan_write_pc(SB), R9
	JMP	racecalladdr<>(SB)

// func runtime·racereadrange(addr, size uintptr)
// Called from instrumented code.
// Defined as ABIInternal so as to avoid introducing a wrapper,
// which would make caller's PC ineffective.
TEXT	runtime·racereadrange<ABIInternal>(SB), NOSPLIT, $0-16
	MOVD	R1, R2	// size
	MOVD	R0, R1	// addr
	MOVD	LR, R3
	// void __tsan_read_range(ThreadState *thr, void *addr, uintptr size, void *pc);
	MOVD	$__tsan_read_range(SB), R9
	JMP	racecalladdr<>(SB)

// func runtime·RaceReadRange(addr, size uintptr)
TEXT	runtime·RaceReadRange(SB), NOSPLIT, $0-16
	// This needs to be a tail call, because racereadrange reads caller pc.
	JMP	runtime·racereadrange(SB)

// func runtime·racereadrangepc1(void *addr, uintptr sz, void *pc)
TEXT	runtime·racereadrangepc1(SB), NOSPLIT, $0-24
	MOVD	addr+0(FP), R1
	MOVD	size+8(FP), R2
	MOVD	pc+16(FP), R3
	ADD	$4, R3	// pc is function start, tsan wants return address.
	// void __tsan_read_range(ThreadState *thr, void *addr, uintptr size, void *pc);
	MOVD	$__tsan_read_range(SB), R9
	JMP	racecalladdr<>(SB)

// func runtime·racewriterange(addr, size uintptr)
// Called from instrumented code.
// Defined as ABIInternal so as to avoid introducing a wrapper,
// which would make caller's PC ineffective.
TEXT	runtime·racewriterange<ABIInternal>(SB), NOSPLIT, $0-16
	MOVD	R1, R2	// size
	MOVD	R0, R1	// addr
	MOVD	LR, R3
	// void __tsan_write_range(ThreadState *thr, void *addr, uintptr size, void *pc);
	MOVD	$__tsan_write_range(SB), R9
	JMP	racecalladdr<>(SB)

// func runtime·RaceWriteRange(addr, size uintptr)
TEXT	runtime·RaceWriteRange(SB), NOSPLIT, $0-16
	// This needs to be a tail call, because racewriterange reads caller pc.
	JMP	runtime·racewriterange(SB)

// func runtime·racewriterangepc1(void *addr, uintptr sz, void *pc)
TEXT	runtime·racewriterangepc1(SB), NOSPLIT, $0-24
	MOVD	addr+0(FP), R1
	MOVD	size+8(FP), R2
	MOVD	pc+16(FP), R3
	ADD	$4, R3	// pc is function start, tsan wants return address.
	// void __tsan_write_range(ThreadState *thr, void *addr, uintptr size, void *pc);
	MOVD	$__tsan_write_range(SB), R9
	JMP	racecalladdr<>(SB)

// If addr (R1) is out of range, do nothing.
// Otherwise, setup goroutine context and invoke racecall. Other arguments already set.
TEXT	racecalladdr<>(SB), NOSPLIT, $0-0
	load_g
	MOVD	g_racectx(g), R0
	// Check that addr is within [arenastart, arenaend) or within [racedatastart, racedataend).
	MOVD	runtime·racearenastart(SB), R10
	CMP	R10, R1
	BLT	data
	MOVD	runtime·racearenaend(SB), R10
	CMP	R10, R1
	BLT	call
data:
	MOVD	runtime·racedatastart(SB), R10
	CMP	R10, R1
	BLT	ret
	MOVD	runtime·racedataend(SB), R10
	CMP	R10, R1
	BGE	ret
call:
	JMP	racecall<>(SB)
ret:
	RET

// func runtime·racefuncenter(pc uintptr)
// Called from instrumented code.
TEXT	runtime·racefuncenter<ABIInternal>(SB), NOSPLIT, $0-8
	MOVD	R0, R9	// callpc
	JMP	racefuncenter<>(SB)

// Common code for racefuncenter
// R9 = caller's return address
TEXT	racefuncenter<>(SB), NOSPLIT, $0-0
	load_g
	MOVD	g_racectx(g), R0	// goroutine racectx
	MOVD	R9, R1
	// void __tsan_func_enter(ThreadState *thr, void *pc);
	MOVD	$__tsan_func_enter(SB), R9
	BL	racecall<>(SB)
	RET

// func runtime·racefuncexit()
// Called from instrumented code.
TEXT	runtime·racefuncexit<ABIInternal>(SB), NOSPLIT, $0-0
	load_g
	MOVD	g_racectx(g), R0	// race context
	// void __tsan_func_exit(ThreadState *thr);
	MOVD	$__tsan_func_exit(SB), R9
	JMP	racecall<>(SB)

// Atomic operations for sync/atomic package.
// R3 = addr of arguments passed to this function, it can
// be fetched at 40(RSP) in racecallatomic after two times BL
// R0, R1, R2 set in racecallatomic

// Load
TEXT	sync∕atomic·LoadInt32(SB), NOSPLIT, $0-12
	GO_ARGS
	MOVD	$__tsan_go_atomic32_load(SB), R9
	BL	racecallatomic<>(SB)
	RET

TEXT	sync∕atomic·LoadInt64(SB), NOSPLIT, $0-16
	GO_ARGS
	MOVD	$__tsan_go_atomic64_load(SB), R9
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
	MOVD	$__tsan_go_atomic32_store(SB), R9
	BL	racecallatomic<>(SB)
	RET

TEXT	sync∕atomic·StoreInt64(SB), NOSPLIT, $0-16
	GO_ARGS
	MOVD	$__tsan_go_atomic64_store(SB), R9
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
	MOVD	$__tsan_go_atomic32_exchange(SB), R9
	BL	racecallatomic<>(SB)
	RET

TEXT	sync∕atomic·SwapInt64(SB), NOSPLIT, $0-24
	GO_ARGS
	MOVD	$__tsan_go_atomic64_exchange(SB), R9
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
	MOVD	$__tsan_go_atomic32_fetch_add(SB), R9
	BL	racecallatomic<>(SB)
	MOVW	add+8(FP), R0	// convert fetch_add to add_fetch
	MOVW	ret+16(FP), R1
	ADD	R0, R1, R0
	MOVW	R0, ret+16(FP)
	RET

TEXT	sync∕atomic·AddInt64(SB), NOSPLIT, $0-24
	GO_ARGS
	MOVD	$__tsan_go_atomic64_fetch_add(SB), R9
	BL	racecallatomic<>(SB)
	MOVD	add+8(FP), R0	// convert fetch_add to add_fetch
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

// And
TEXT	sync∕atomic·AndInt32(SB), NOSPLIT, $0-20
	GO_ARGS
	MOVD	$__tsan_go_atomic32_fetch_and(SB), R9
	BL	racecallatomic<>(SB)
	RET

TEXT	sync∕atomic·AndInt64(SB), NOSPLIT, $0-24
	GO_ARGS
	MOVD	$__tsan_go_atomic64_fetch_and(SB), R9
	BL	racecallatomic<>(SB)
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
	MOVD	$__tsan_go_atomic32_fetch_or(SB), R9
	BL	racecallatomic<>(SB)
	RET

TEXT	sync∕atomic·OrInt64(SB), NOSPLIT, $0-24
	GO_ARGS
	MOVD	$__tsan_go_atomic64_fetch_or(SB), R9
	BL	racecallatomic<>(SB)
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
	MOVD	$__tsan_go_atomic32_compare_exchange(SB), R9
	BL	racecallatomic<>(SB)
	RET

TEXT	sync∕atomic·CompareAndSwapInt64(SB), NOSPLIT, $0-25
	GO_ARGS
	MOVD	$__tsan_go_atomic64_compare_exchange(SB), R9
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

// Generic atomic operation implementation.
// R9 = addr of target function
TEXT	racecallatomic<>(SB), NOSPLIT, $0
	// Set up these registers
	// R0 = *ThreadState
	// R1 = caller pc
	// R2 = pc
	// R3 = addr of incoming arg list

	// Trigger SIGSEGV early.
	MOVD	72(RSP), R3	// 1st arg is addr. after two small frames (32 bytes each), get it at 72(RSP)
	MOVB	(R3), R13	// segv here if addr is bad
	// Check that addr is within [arenastart, arenaend) or within [racedatastart, racedataend).
	MOVD	runtime·racearenastart(SB), R10
	CMP	R10, R3
	BLT	racecallatomic_data
	MOVD	runtime·racearenaend(SB), R10
	CMP	R10, R3
	BLT	racecallatomic_ok
racecallatomic_data:
	MOVD	runtime·racedatastart(SB), R10
	CMP	R10, R3
	BLT	racecallatomic_ignore
	MOVD	runtime·racedataend(SB), R10
	CMP	R10, R3
	BGE	racecallatomic_ignore
racecallatomic_ok:
	// Addr is within the good range, call the atomic function.
	load_g
	MOVD	g_racectx(g), R0	// goroutine context
	MOVD	56(RSP), R1	// caller pc
	MOVD	R9, R2	// pc
	ADD	$72, RSP, R3
	BL	racecall<>(SB)
	RET
racecallatomic_ignore:
	// Addr is outside the good range.
	// Call __tsan_go_ignore_sync_begin to ignore synchronization during the atomic op.
	// An attempt to synchronize on the address would cause crash.
	MOVD	R9, R21	// remember the original function
	MOVD	$__tsan_go_ignore_sync_begin(SB), R9
	load_g
	MOVD	g_racectx(g), R0	// goroutine context
	BL	racecall<>(SB)
	MOVD	R21, R9	// restore the original function
	// Call the atomic function.
	// racecall will call LLVM race code which might clobber R28 (g)
	load_g
	MOVD	g_racectx(g), R0	// goroutine context
	MOVD	56(RSP), R1	// caller pc
	MOVD	R9, R2	// pc
	ADD	$72, RSP, R3	// arguments
	BL	racecall<>(SB)
	// Call __tsan_go_ignore_sync_end.
	MOVD	$__tsan_go_ignore_sync_end(SB), R9
	MOVD	g_racectx(g), R0	// goroutine context
	BL	racecall<>(SB)
	RET

// func runtime·racecall(void(*f)(...), ...)
// Calls C function f from race runtime and passes up to 4 arguments to it.
// The arguments are never heap-object-preserving pointers, so we pretend there are no arguments.
TEXT	runtime·racecall(SB), NOSPLIT, $0-0
	MOVD	fn+0(FP), R9
	MOVD	arg0+8(FP), R0
	MOVD	arg1+16(FP), R1
	MOVD	arg2+24(FP), R2
	MOVD	arg3+32(FP), R3
	JMP	racecall<>(SB)

// Switches SP to g0 stack and calls (R9). Arguments already set.
// Clobbers R19, R20.
TEXT	racecall<>(SB), NOSPLIT|NOFRAME, $0-0
	MOVD	g_m(g), R10
	// Switch to g0 stack.
	MOVD	RSP, R19	// callee-saved, preserved across the CALL
	MOVD	R30, R20	// callee-saved, preserved across the CALL

	// Switch to g0 stack if we aren't already on g0 or gsignal.
	MOVD	m_gsignal(R10), R11
	CMP	R11, g
	BEQ	call

	MOVD	m_g0(R10), R11
	CMP	R11, g
	BEQ	call

	MOVD	(g_sched+gobuf_sp)(R11), R12
	MOVD	R12, RSP
call:
	BL	R9
	MOVD	R19, RSP
	JMP	(R20)

// C->Go callback thunk that allows to call runtime·racesymbolize from C code.
// Direct Go->C race call has only switched SP, finish g->g0 switch by setting correct g.
// The overall effect of Go->C->Go call chain is similar to that of mcall.
// R0 contains command code. R1 contains command-specific context.
// See racecallback for command codes.
TEXT	runtime·racecallbackthunk(SB), NOSPLIT|NOFRAME, $0
	// Handle command raceGetProcCmd (0) here.
	// First, code below assumes that we are on curg, while raceGetProcCmd
	// can be executed on g0. Second, it is called frequently, so will
	// benefit from this fast path.
	CBNZ	R0, rest
	MOVD	g, R13
#ifdef TLS_darwin
	MOVD	R27, R12 // save R27 a.k.a. REGTMP (callee-save in C). load_g clobbers it
#endif
	load_g
#ifdef TLS_darwin
	MOVD	R12, R27
#endif
	MOVD	g_m(g), R0
	MOVD	m_p(R0), R0
	MOVD	p_raceprocctx(R0), R0
	MOVD	R0, (R1)
	MOVD	R13, g
	JMP	(LR)
rest:
	// Save callee-saved registers (Go code won't respect that).
	// 8(RSP) and 16(RSP) are for args passed through racecallback
	SUB	$176, RSP
	MOVD	LR, 0(RSP)

	SAVE_R19_TO_R28(8*3)
	SAVE_F8_TO_F15(8*13)
	MOVD	R29, (8*21)(RSP)
	// Set g = g0.
	// load_g will clobber R0, Save R0
	MOVD	R0, R13
	load_g
	// restore R0
	MOVD	R13, R0
	MOVD	g_m(g), R13
	MOVD	m_g0(R13), R14
	CMP	R14, g
	BEQ	noswitch	// branch if already on g0
	MOVD	R14, g

	MOVD	R0, 8(RSP)	// func arg
	MOVD	R1, 16(RSP)	// func arg
	BL	runtime·racecallback(SB)

	// All registers are smashed after Go code, reload.
	MOVD	g_m(g), R13
	MOVD	m_curg(R13), g	// g = m->curg
ret:
	// Restore callee-saved registers.
	MOVD	0(RSP), LR
	MOVD	(8*21)(RSP), R29
	RESTORE_F8_TO_F15(8*13)
	RESTORE_R19_TO_R28(8*3)
	ADD	$176, RSP
	JMP	(LR)

noswitch:
	// already on g0
	MOVD	R0, 8(RSP)	// func arg
	MOVD	R1, 16(RSP)	// func arg
	BL	runtime·racecallback(SB)
	JMP	ret

#ifndef TLSG_IS_VARIABLE
// tls_g, g value for each thread in TLS
GLOBL runtime·tls_g+0(SB), TLSBSS+DUPOK, $8
#endif
