// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build race

#include "go_asm.h"
#include "funcdata.h"
#include "textflag.h"
#include "cgo/abi_riscv64.h"

// The following thunks allow calling the gcc-compiled race runtime directly
// from Go code without going all the way through cgo.
// First, it's much faster (up to 50% speedup for real Go programs).
// Second, it eliminates race-related special cases from cgocall and scheduler.
// Third, in long-term it will allow to remove cyclic runtime/race dependency on cmd/go.

// A brief recap of the riscv C calling convention.
// Arguments are passed in X10...X17
// Callee-saved registers are: X8, X9, X18..X27
// Temporary registers are: X5..X7, X28..X31

// When calling racecalladdr, X11 is the call target address.

// The race ctx, ThreadState *thr below, is passed in X10 and loaded in racecalladdr.

// func runtime·raceread(addr uintptr)
// Called from instrumented code.
TEXT	runtime·raceread<ABIInternal>(SB), NOSPLIT, $0-8
	// void __tsan_read(ThreadState *thr, void *addr, void *pc);
	MOV	$__tsan_read(SB), X5
	MOV	X10, X11
	MOV	X1, X12
	JMP	racecalladdr<>(SB)

// func runtime·RaceRead(addr uintptr)
TEXT	runtime·RaceRead(SB), NOSPLIT, $0-8
	// This needs to be a tail call, because raceread reads caller pc.
	JMP	runtime·raceread(SB)

// func runtime·racereadpc(void *addr, void *callpc, void *pc)
TEXT	runtime·racereadpc(SB), NOSPLIT, $0-24
	// void __tsan_read_pc(ThreadState *thr, void *addr, void *callpc, void *pc);
	MOV	$__tsan_read_pc(SB), X5
	MOV	addr+0(FP), X11
	MOV	callpc+8(FP), X12
	MOV	pc+16(FP), X13
	JMP	racecalladdr<>(SB)

// func runtime·racewrite(addr uintptr)
// Called from instrumented code.
TEXT	runtime·racewrite<ABIInternal>(SB), NOSPLIT, $0-8
	// void __tsan_write(ThreadState *thr, void *addr, void *pc);
	MOV	$__tsan_write(SB), X5
	MOV	X10, X11
	MOV	X1, X12
	JMP	racecalladdr<>(SB)

// func runtime·RaceWrite(addr uintptr)
TEXT	runtime·RaceWrite(SB), NOSPLIT, $0-8
	// This needs to be a tail call, because racewrite reads caller pc.
	JMP	runtime·racewrite(SB)

// func runtime·racewritepc(void *addr, void *callpc, void *pc)
TEXT	runtime·racewritepc(SB), NOSPLIT, $0-24
	// void __tsan_write_pc(ThreadState *thr, void *addr, void *callpc, void *pc);
	MOV	$__tsan_write_pc(SB), X5
	MOV	addr+0(FP), X11
	MOV	callpc+8(FP), X12
	MOV	pc+16(FP), X13
	JMP	racecalladdr<>(SB)

// func runtime·racereadrange(addr, size uintptr)
// Called from instrumented code.
TEXT	runtime·racereadrange<ABIInternal>(SB), NOSPLIT, $0-16
	// void __tsan_read_range(ThreadState *thr, void *addr, uintptr size, void *pc);
	MOV	$__tsan_read_range(SB), X5
	MOV	X11, X12
	MOV	X10, X11
	MOV	X1, X13
	JMP	racecalladdr<>(SB)

// func runtime·RaceReadRange(addr, size uintptr)
TEXT	runtime·RaceReadRange(SB), NOSPLIT, $0-16
	// This needs to be a tail call, because racereadrange reads caller pc.
	JMP	runtime·racereadrange(SB)

// func runtime·racereadrangepc1(void *addr, uintptr sz, void *pc)
TEXT	runtime·racereadrangepc1(SB), NOSPLIT, $0-24
	// void __tsan_read_range(ThreadState *thr, void *addr, uintptr size, void *pc);
	MOV	$__tsan_read_range(SB), X5
	MOV	addr+0(FP), X11
	MOV	size+8(FP), X12
	MOV	pc+16(FP), X13

	// pc is an interceptor address, but TSan expects it to point to the
	// middle of an interceptor (see LLVM's SCOPED_INTERCEPTOR_RAW).
	ADD	$4, X13
	JMP	racecalladdr<>(SB)

// func runtime·racewriterange(addr, size uintptr)
// Called from instrumented code.
TEXT	runtime·racewriterange<ABIInternal>(SB), NOSPLIT, $0-16
	// void __tsan_write_range(ThreadState *thr, void *addr, uintptr size, void *pc);
	MOV	$__tsan_write_range(SB), X5
	MOV	X11, X12
	MOV	X10, X11
	MOV	X1, X13
	JMP	racecalladdr<>(SB)

// func runtime·RaceWriteRange(addr, size uintptr)
TEXT	runtime·RaceWriteRange(SB), NOSPLIT, $0-16
	// This needs to be a tail call, because racewriterange reads caller pc.
	JMP	runtime·racewriterange(SB)

// func runtime·racewriterangepc1(void *addr, uintptr sz, void *pc)
TEXT	runtime·racewriterangepc1(SB), NOSPLIT, $0-24
	// void __tsan_write_range(ThreadState *thr, void *addr, uintptr size, void *pc);
	MOV	$__tsan_write_range(SB), X5
	MOV	addr+0(FP), X11
	MOV	size+8(FP), X12
	MOV	pc+16(FP), X13
	// pc is an interceptor address, but TSan expects it to point to the
	// middle of an interceptor (see LLVM's SCOPED_INTERCEPTOR_RAW).
	ADD	$4, X13
	JMP	racecalladdr<>(SB)

// If addr (X11) is out of range, do nothing. Otherwise, setup goroutine context and
// invoke racecall. Other arguments are already set.
TEXT	racecalladdr<>(SB), NOSPLIT, $0-0
	MOV	runtime·racearenastart(SB), X7
	BLT	X11, X7, data			// Before racearena start?
	MOV	runtime·racearenaend(SB), X7
	BLT	X11, X7, call			// Before racearena end?
data:
	MOV	runtime·racedatastart(SB), X7
	BLT	X11, X7, ret			// Before racedata start?
	MOV	runtime·racedataend(SB), X7
	BGE	X11, X7, ret			// At or after racedata end?
call:
	MOV	g_racectx(g), X10
	JMP	racecall<>(SB)
ret:
	RET

// func runtime·racefuncenter(pc uintptr)
// Called from instrumented code.
TEXT	runtime·racefuncenter<ABIInternal>(SB), NOSPLIT, $0-8
	MOV	$__tsan_func_enter(SB), X5
	MOV	X10, X11
	MOV	g_racectx(g), X10
	JMP	racecall<>(SB)

// Common code for racefuncenter
// X1 = caller's return address
TEXT	racefuncenter<>(SB), NOSPLIT, $0-0
	// void __tsan_func_enter(ThreadState *thr, void *pc);
	MOV	$__tsan_func_enter(SB), X5
	MOV	g_racectx(g), X10
	MOV	X1, X11
	JMP	racecall<>(SB)

// func runtime·racefuncexit()
// Called from instrumented code.
TEXT	runtime·racefuncexit<ABIInternal>(SB), NOSPLIT, $0-0
	// void __tsan_func_exit(ThreadState *thr);
	MOV	$__tsan_func_exit(SB), X5
	MOV	g_racectx(g), X10
	JMP	racecall<>(SB)

// Atomic operations for sync/atomic package.

// Load

TEXT	sync∕atomic·LoadInt32(SB), NOSPLIT, $0-12
	GO_ARGS
	MOV	$__tsan_go_atomic32_load(SB), X5
	CALL	racecallatomic<>(SB)
	RET

TEXT	sync∕atomic·LoadInt64(SB), NOSPLIT, $0-16
	GO_ARGS
	MOV	$__tsan_go_atomic64_load(SB), X5
	CALL	racecallatomic<>(SB)
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
	MOV	$__tsan_go_atomic32_store(SB), X5
	CALL	racecallatomic<>(SB)
	RET

TEXT	sync∕atomic·StoreInt64(SB), NOSPLIT, $0-16
	GO_ARGS
	MOV	$__tsan_go_atomic64_store(SB), X5
	CALL	racecallatomic<>(SB)
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
	MOV	$__tsan_go_atomic32_exchange(SB), X5
	CALL	racecallatomic<>(SB)
	RET

TEXT	sync∕atomic·SwapInt64(SB), NOSPLIT, $0-24
	GO_ARGS
	MOV	$__tsan_go_atomic64_exchange(SB), X5
	CALL	racecallatomic<>(SB)
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
	MOV	$__tsan_go_atomic32_fetch_add(SB), X5
	CALL	racecallatomic<>(SB)
	// TSan performed fetch_add, but Go needs add_fetch.
	MOVW	add+8(FP), X5
	MOVW	ret+16(FP), X6
	ADD	X5, X6, X5
	MOVW	X5, ret+16(FP)
	RET

TEXT	sync∕atomic·AddInt64(SB), NOSPLIT, $0-24
	GO_ARGS
	MOV	$__tsan_go_atomic64_fetch_add(SB), X5
	CALL	racecallatomic<>(SB)
	// TSan performed fetch_add, but Go needs add_fetch.
	MOV	add+8(FP), X5
	MOV	ret+16(FP), X6
	ADD	X5, X6, X5
	MOV	X5, ret+16(FP)
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
	MOV	$__tsan_go_atomic32_fetch_and(SB), X5
	CALL	racecallatomic<>(SB)
	RET

TEXT	sync∕atomic·AndInt64(SB), NOSPLIT, $0-24
	GO_ARGS
	MOV	$__tsan_go_atomic64_fetch_and(SB), X5
	CALL	racecallatomic<>(SB)
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
	MOV	$__tsan_go_atomic32_fetch_or(SB), X5
	CALL	racecallatomic<>(SB)
	RET

TEXT	sync∕atomic·OrInt64(SB), NOSPLIT, $0-24
	GO_ARGS
	MOV	$__tsan_go_atomic64_fetch_or(SB), X5
	CALL	racecallatomic<>(SB)
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
	MOV	$__tsan_go_atomic32_compare_exchange(SB), X5
	CALL	racecallatomic<>(SB)
	RET

TEXT	sync∕atomic·CompareAndSwapInt64(SB), NOSPLIT, $0-25
	GO_ARGS
	MOV	$__tsan_go_atomic64_compare_exchange(SB), X5
	CALL	racecallatomic<>(SB)
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
// X5 = addr of target function
TEXT	racecallatomic<>(SB), NOSPLIT, $0
	// Set up these registers
	// X10 = *ThreadState
	// X11 = caller pc
	// X12 = pc
	// X13 = addr of incoming arg list

	// Trigger SIGSEGV early.
	MOV	24(X2), X6	// 1st arg is addr. after two times CALL, get it at 24(X2)
	MOVB	(X6), X0	// segv here if addr is bad
	// Check that addr is within [arenastart, arenaend) or within [racedatastart, racedataend).
	MOV	runtime·racearenastart(SB), X7
	BLT	X6, X7, racecallatomic_data
	MOV	runtime·racearenaend(SB), X7
	BLT	X6, X7, racecallatomic_ok
racecallatomic_data:
	MOV	runtime·racedatastart(SB), X7
	BLT	X6, X7, racecallatomic_ignore
	MOV	runtime·racedataend(SB), X7
	BGE	X6, X7, racecallatomic_ignore
racecallatomic_ok:
	// Addr is within the good range, call the atomic function.
	MOV	g_racectx(g), X10	// goroutine context
	MOV	8(X2), X11		// caller pc
	MOV	X1, X12			// pc
	ADD	$24, X2, X13
	CALL	racecall<>(SB)
	RET
racecallatomic_ignore:
	// Addr is outside the good range.
	// Call __tsan_go_ignore_sync_begin to ignore synchronization during the atomic op.
	// An attempt to synchronize on the address would cause crash.
	MOV	X1, X20			// save PC
	MOV	X5, X21			// save target function
	MOV	$__tsan_go_ignore_sync_begin(SB), X5
	MOV	g_racectx(g), X10	// goroutine context
	CALL	racecall<>(SB)
	MOV	X21, X5			// restore the target function
	// Call the atomic function.
	MOV	g_racectx(g), X10	// goroutine context
	MOV	8(X2), X11		// caller pc
	MOV	X20, X12		// pc
	ADD	$24, X2, X13		// arguments
	CALL	racecall<>(SB)
	// Call __tsan_go_ignore_sync_end.
	MOV	$__tsan_go_ignore_sync_end(SB), X5
	MOV	g_racectx(g), X10	// goroutine context
	CALL	racecall<>(SB)
	RET

// func runtime·racecall(void(*f)(...), ...)
// Calls C function f from race runtime and passes up to 4 arguments to it.
// The arguments are never heap-object-preserving pointers, so we pretend there
// are no arguments.
TEXT	runtime·racecall(SB), NOSPLIT, $0-0
	MOV	fn+0(FP), X5
	MOV	arg0+8(FP), X10
	MOV	arg1+16(FP), X11
	MOV	arg2+24(FP), X12
	MOV	arg3+32(FP), X13
	JMP	racecall<>(SB)

// Switches SP to g0 stack and calls X5. Arguments are already set.
TEXT	racecall<>(SB), NOSPLIT|NOFRAME, $0-0
	MOV	X1, X18				// Save RA in callee save register
	MOV	X2, X19				// Save SP in callee save register
	CALL	runtime·save_g(SB)	// Save g for callbacks

	MOV	g_m(g), X6

	// Switch to g0 stack if we aren't already on g0 or gsignal.
	MOV	m_gsignal(X6), X7
	BEQ	X7, g, call
	MOV	m_g0(X6), X7
	BEQ	X7, g, call

	MOV	(g_sched+gobuf_sp)(X7), X2	// Switch to g0 stack
call:
	JALR	RA, (X5)			// Call C function
	MOV	X19, X2				// Restore SP
	JMP	(X18)				// Return to Go.

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
	BNEZ	X10, rest
	MOV	X1, X5
	MOV	g, X6
	CALL	runtime·load_g(SB)
	MOV	g_m(g), X7
	MOV	m_p(X7), X7
	MOV	p_raceprocctx(X7), X7
	MOV	X7, (X11)
	MOV	X6, g
	JMP	(X5)
rest:
	// Save callee-save registers (X8, X9, X18..X27, F8, F9, F18..F27),
	// since Go code will not respect this.
	// 8(X2) and 16(X2) are for args passed to racecallback
	SUB	$(27*8), X2
	MOV	X1, (0*8)(X2)
	SAVE_GPR((3*8))
	SAVE_FPR((15*8))

	// Set g = g0.
	CALL	runtime·load_g(SB)
	MOV	g_m(g), X5
	MOV	m_g0(X5), X6
	BEQ	X6, g, noswitch	// branch if already on g0
	MOV	X6, g

	MOV	X10, 8(X2)	// func arg
	MOV	X11, 16(X2)	// func arg
	CALL	runtime·racecallback(SB)

	// All registers are smashed after Go code, reload.
	MOV	g_m(g), X5
	MOV	m_curg(X5), g	// g = m->curg
ret:
	// Restore callee-save registers.
	MOV	(0*8)(X2), X1
	RESTORE_GPR((3*8))
	RESTORE_FPR((15*8))

	ADD	$(27*8), X2
	JMP	(X1)

noswitch:
	// already on g0
	MOV	X10, 8(X2)	// func arg
	MOV	X11, 16(X2)	// func arg
	CALL	runtime·racecallback(SB)
	JMP	ret
