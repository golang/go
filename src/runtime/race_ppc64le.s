// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build race

#include "go_asm.h"
#include "go_tls.h"
#include "funcdata.h"
#include "textflag.h"

// The following functions allow calling the clang-compiled race runtime directly
// from Go code without going all the way through cgo.
// First, it's much faster (up to 50% speedup for real Go programs).
// Second, it eliminates race-related special cases from cgocall and scheduler.
// Third, in long-term it will allow to remove cyclic runtime/race dependency on cmd/go.

// A brief recap of the ppc64le calling convention.
// Arguments are passed in R3, R4, R5 ...
// SP must be 16-byte aligned.

// Note that for ppc64x, LLVM follows the standard ABI and
// expects arguments in registers, so these functions move
// the arguments from storage to the registers expected
// by the ABI.

// When calling from Go to Clang tsan code:
// R3 is the 1st argument and is usually the ThreadState*
// R4-? are the 2nd, 3rd, 4th, etc. arguments

// When calling racecalladdr:
// R8 is the call target address

// The race ctx is passed in R3 and loaded in
// racecalladdr.
//
// The sequence used to get the race ctx:
//    MOVD    runtime·tls_g(SB), R10	// offset to TLS
//    MOVD    0(R13)(R10*1), g		// R13=TLS for this thread, g = R30
//    MOVD    g_racectx(g), R3		// racectx == ThreadState

// func runtime·RaceRead(addr uintptr)
// Called from instrumented Go code
TEXT	runtime·raceread(SB), NOSPLIT, $0-8
	MOVD	addr+0(FP), R4
	MOVD	LR, R5 // caller of this?
	// void __tsan_read(ThreadState *thr, void *addr, void *pc);
	MOVD	$__tsan_read(SB), R8
	BR	racecalladdr<>(SB)

TEXT    runtime·RaceRead(SB), NOSPLIT, $0-8
	BR	runtime·raceread(SB)

// void runtime·racereadpc(void *addr, void *callpc, void *pc)
TEXT	runtime·racereadpc(SB), NOSPLIT, $0-24
	MOVD	addr+0(FP), R4
	MOVD	callpc+8(FP), R5
	MOVD	pc+16(FP), R6
	// void __tsan_read_pc(ThreadState *thr, void *addr, void *callpc, void *pc);
	MOVD	$__tsan_read_pc(SB), R8
	BR	racecalladdr<>(SB)

// func runtime·RaceWrite(addr uintptr)
// Called from instrumented Go code
TEXT	runtime·racewrite(SB), NOSPLIT, $0-8
	MOVD	addr+0(FP), R4
	MOVD	LR, R5 // caller has set LR via BL inst
	// void __tsan_write(ThreadState *thr, void *addr, void *pc);
	MOVD	$__tsan_write(SB), R8
	BR	racecalladdr<>(SB)

TEXT    runtime·RaceWrite(SB), NOSPLIT, $0-8
	JMP	runtime·racewrite(SB)

// void runtime·racewritepc(void *addr, void *callpc, void *pc)
TEXT	runtime·racewritepc(SB), NOSPLIT, $0-24
	MOVD	addr+0(FP), R4
	MOVD	callpc+8(FP), R5
	MOVD	pc+16(FP), R6
	// void __tsan_write_pc(ThreadState *thr, void *addr, void *callpc, void *pc);
	MOVD	$__tsan_write_pc(SB), R8
	BR	racecalladdr<>(SB)

// func runtime·RaceReadRange(addr, size uintptr)
// Called from instrumented Go code.
TEXT	runtime·racereadrange(SB), NOSPLIT, $0-16
	MOVD	addr+0(FP), R4
	MOVD	size+8(FP), R5
	MOVD	LR, R6
	// void __tsan_read_range(ThreadState *thr, void *addr, uintptr size, void *pc);
	MOVD	$__tsan_read_range(SB), R8
	BR	racecalladdr<>(SB)

// void runtime·racereadrangepc1(void *addr, uintptr sz, void *pc)
TEXT	runtime·racereadrangepc1(SB), NOSPLIT, $0-24
	MOVD    addr+0(FP), R4
	MOVD    size+8(FP), R5
	MOVD    pc+16(FP), R6
	ADD	$4, R6		// tsan wants return addr
        // void __tsan_read_range(ThreadState *thr, void *addr, uintptr size, void *pc);
        MOVD    $__tsan_read_range(SB), R8
        BR	racecalladdr<>(SB)

TEXT    runtime·RaceReadRange(SB), NOSPLIT, $0-24
	BR	runtime·racereadrange(SB)

// func runtime·RaceWriteRange(addr, size uintptr)
// Called from instrumented Go code.
TEXT	runtime·racewriterange(SB), NOSPLIT, $0-16
	MOVD	addr+0(FP), R4
	MOVD	size+8(FP), R5
	MOVD	LR, R6
	// void __tsan_write_range(ThreadState *thr, void *addr, uintptr size, void *pc);
	MOVD	$__tsan_write_range(SB), R8
	BR	racecalladdr<>(SB)

TEXT    runtime·RaceWriteRange(SB), NOSPLIT, $0-16
	BR	runtime·racewriterange(SB)

// void runtime·racewriterangepc1(void *addr, uintptr sz, void *pc)
// Called from instrumented Go code
TEXT	runtime·racewriterangepc1(SB), NOSPLIT, $0-24
	MOVD	addr+0(FP), R4
	MOVD	size+8(FP), R5
	MOVD	pc+16(FP), R6
	ADD	$4, R6			// add 4 to inst offset?
	// void __tsan_write_range(ThreadState *thr, void *addr, uintptr size, void *pc);
	MOVD	$__tsan_write_range(SB), R8
	BR	racecalladdr<>(SB)

// Call a __tsan function from Go code.
// R8 = tsan function address
// R3 = *ThreadState a.k.a. g_racectx from g
// R4 = addr passed to __tsan function
//
// Otherwise, setup goroutine context and invoke racecall. Other arguments already set.
TEXT	racecalladdr<>(SB), NOSPLIT, $0-0
	MOVD    runtime·tls_g(SB), R10
	MOVD	0(R13)(R10*1), g
	MOVD	g_racectx(g), R3	// goroutine context
	// Check that addr is within [arenastart, arenaend) or within [racedatastart, racedataend).
	MOVD	runtime·racearenastart(SB), R9
	CMP	R4, R9
	BLT	data
	MOVD	runtime·racearenaend(SB), R9
	CMP	R4, R9
	BLT	call
data:
	MOVD	runtime·racedatastart(SB), R9
	CMP	R4, R9
	BLT	ret
	MOVD	runtime·racedataend(SB), R9
	CMP	R4, R9
	BGT	ret
call:
	// Careful!! racecall will save LR on its
	// stack, which is OK as long as racecalladdr
	// doesn't change in a way that generates a stack.
	// racecall should return to the caller of
	// recalladdr.
	BR	racecall<>(SB)
ret:
	RET

// func runtime·racefuncenterfp()
// Called from instrumented Go code.
// Like racefuncenter but doesn't pass an arg, uses the caller pc
// from the first slot on the stack.
TEXT	runtime·racefuncenterfp(SB), NOSPLIT, $0-0
	MOVD	0(R1), R8
	BR	racefuncenter<>(SB)

// func runtime·racefuncenter(pc uintptr)
// Called from instrumented Go code.
// Not used now since gc/racewalk.go doesn't pass the
// correct caller pc and racefuncenterfp can do it.
TEXT	runtime·racefuncenter(SB), NOSPLIT, $0-8
	MOVD	callpc+0(FP), R8
	BR	racefuncenter<>(SB)

// Common code for racefuncenter/racefuncenterfp
// R11 = caller's return address
TEXT	racefuncenter<>(SB), NOSPLIT, $0-0
	MOVD    runtime·tls_g(SB), R10
	MOVD    0(R13)(R10*1), g
	MOVD    g_racectx(g), R3        // goroutine racectx aka *ThreadState
	MOVD	R8, R4			// caller pc set by caller in R8
	// void __tsan_func_enter(ThreadState *thr, void *pc);
	MOVD	$__tsan_func_enter(SB), R8
	BR	racecall<>(SB)
	RET

// func runtime·racefuncexit()
// Called from Go instrumented code.
TEXT	runtime·racefuncexit(SB), NOSPLIT, $0-0
	MOVD    runtime·tls_g(SB), R10
	MOVD    0(R13)(R10*1), g
	MOVD    g_racectx(g), R3        // goroutine racectx aka *ThreadState
	// void __tsan_func_exit(ThreadState *thr);
	MOVD	$__tsan_func_exit(SB), R8
	BR	racecall<>(SB)

// Atomic operations for sync/atomic package.
// Some use the __tsan versions instead
// R6 = addr of arguments passed to this function
// R3, R4, R5 set in racecallatomic

// Load atomic in tsan
TEXT	sync∕atomic·LoadInt32(SB), NOSPLIT, $0-0
	// void __tsan_go_atomic32_load(ThreadState *thr, uptr cpc, uptr pc, u8 *a);
	MOVD	$__tsan_go_atomic32_load(SB), R8
	ADD	$32, R1, R6	// addr of caller's 1st arg
	BR	racecallatomic<>(SB)
	RET

TEXT	sync∕atomic·LoadInt64(SB), NOSPLIT, $0-0
	// void __tsan_go_atomic64_load(ThreadState *thr, uptr cpc, uptr pc, u8 *a);
	MOVD	$__tsan_go_atomic64_load(SB), R8
	ADD	$32, R1, R6	// addr of caller's 1st arg
	BR	racecallatomic<>(SB)
	RET

TEXT	sync∕atomic·LoadUint32(SB), NOSPLIT, $0-0
	BR	sync∕atomic·LoadInt32(SB)

TEXT	sync∕atomic·LoadUint64(SB), NOSPLIT, $0-0
	BR	sync∕atomic·LoadInt64(SB)

TEXT	sync∕atomic·LoadUintptr(SB), NOSPLIT, $0-0
	BR	sync∕atomic·LoadInt64(SB)

TEXT	sync∕atomic·LoadPointer(SB), NOSPLIT, $0-0
	BR	sync∕atomic·LoadInt64(SB)

// Store atomic in tsan
TEXT	sync∕atomic·StoreInt32(SB), NOSPLIT, $0-0
	// void __tsan_go_atomic32_store(ThreadState *thr, uptr cpc, uptr pc, u8 *a);
	MOVD	$__tsan_go_atomic32_store(SB), R8
	ADD	$32, R1, R6	// addr of caller's 1st arg
	BR	racecallatomic<>(SB)

TEXT	sync∕atomic·StoreInt64(SB), NOSPLIT, $0-0
	// void __tsan_go_atomic64_store(ThreadState *thr, uptr cpc, uptr pc, u8 *a);
	MOVD	$__tsan_go_atomic64_store(SB), R8
	ADD	$32, R1, R6	// addr of caller's 1st arg
	BR	racecallatomic<>(SB)

TEXT	sync∕atomic·StoreUint32(SB), NOSPLIT, $0-0
	BR	sync∕atomic·StoreInt32(SB)

TEXT	sync∕atomic·StoreUint64(SB), NOSPLIT, $0-0
	BR	sync∕atomic·StoreInt64(SB)

TEXT	sync∕atomic·StoreUintptr(SB), NOSPLIT, $0-0
	BR	sync∕atomic·StoreInt64(SB)

// Swap in tsan
TEXT	sync∕atomic·SwapInt32(SB), NOSPLIT, $0-0
	// void __tsan_go_atomic32_exchange(ThreadState *thr, uptr cpc, uptr pc, u8 *a);
	MOVD	$__tsan_go_atomic32_exchange(SB), R8
	ADD	$32, R1, R6	// addr of caller's 1st arg
	BR	racecallatomic<>(SB)

TEXT	sync∕atomic·SwapInt64(SB), NOSPLIT, $0-0
	// void __tsan_go_atomic64_exchange(ThreadState *thr, uptr cpc, uptr pc, u8 *a)
	MOVD	$__tsan_go_atomic64_exchange(SB), R8
	ADD	$32, R1, R6	// addr of caller's 1st arg
	BR	racecallatomic<>(SB)

TEXT	sync∕atomic·SwapUint32(SB), NOSPLIT, $0-0
	BR	sync∕atomic·SwapInt32(SB)

TEXT	sync∕atomic·SwapUint64(SB), NOSPLIT, $0-0
	BR	sync∕atomic·SwapInt64(SB)

TEXT	sync∕atomic·SwapUintptr(SB), NOSPLIT, $0-0
	BR	sync∕atomic·SwapInt64(SB)

// Add atomic in tsan
TEXT	sync∕atomic·AddInt32(SB), NOSPLIT, $0-0
	// void __tsan_go_atomic32_fetch_add(ThreadState *thr, uptr cpc, uptr pc, u8 *a);
	MOVD	$__tsan_go_atomic32_fetch_add(SB), R8
	ADD	$64, R1, R6	// addr of caller's 1st arg
	BL	racecallatomic<>(SB)
	// The tsan fetch_add result is not as expected by Go,
	// so the 'add' must be added to the result.
	MOVW	add+8(FP), R3	// The tsa fetch_add does not return the
	MOVW	ret+16(FP), R4	// result as expected by go, so fix it.
	ADD	R3, R4, R3
	MOVW	R3, ret+16(FP)
	RET

TEXT	sync∕atomic·AddInt64(SB), NOSPLIT, $0-0
	// void __tsan_go_atomic64_fetch_add(ThreadState *thr, uptr cpc, uptr pc, u8 *a);
	MOVD	$__tsan_go_atomic64_fetch_add(SB), R8
	ADD	$64, R1, R6	// addr of caller's 1st arg
	BL	racecallatomic<>(SB)
	// The tsan fetch_add result is not as expected by Go,
	// so the 'add' must be added to the result.
	MOVD	add+8(FP), R3
	MOVD	ret+16(FP), R4
	ADD	R3, R4, R3
	MOVD	R3, ret+16(FP)
	RET

TEXT	sync∕atomic·AddUint32(SB), NOSPLIT, $0-0
	BR	sync∕atomic·AddInt32(SB)

TEXT	sync∕atomic·AddUint64(SB), NOSPLIT, $0-0
	BR	sync∕atomic·AddInt64(SB)

TEXT	sync∕atomic·AddUintptr(SB), NOSPLIT, $0-0
	BR	sync∕atomic·AddInt64(SB)

// CompareAndSwap in tsan
TEXT	sync∕atomic·CompareAndSwapInt32(SB), NOSPLIT, $0-0
	// void __tsan_go_atomic32_compare_exchange(
	//   ThreadState *thr, uptr cpc, uptr pc, u8 *a)
	MOVD	$__tsan_go_atomic32_compare_exchange(SB), R8
	ADD	$32, R1, R6	// addr of caller's 1st arg
	BR	racecallatomic<>(SB)

TEXT	sync∕atomic·CompareAndSwapInt64(SB), NOSPLIT, $0-0
	// void __tsan_go_atomic32_compare_exchange(
	//   ThreadState *thr, uptr cpc, uptr pc, u8 *a)
	MOVD	$__tsan_go_atomic64_compare_exchange(SB), R8
	ADD	$32, R1, R6	// addr of caller's 1st arg
	BR	racecallatomic<>(SB)

TEXT	sync∕atomic·CompareAndSwapUint32(SB), NOSPLIT, $0-0
	BR	sync∕atomic·CompareAndSwapInt32(SB)

TEXT	sync∕atomic·CompareAndSwapUint64(SB), NOSPLIT, $0-0
	BR	sync∕atomic·CompareAndSwapInt64(SB)

TEXT	sync∕atomic·CompareAndSwapUintptr(SB), NOSPLIT, $0-0
	BR	sync∕atomic·CompareAndSwapInt64(SB)

// Common function used to call tsan's atomic functions
// R3 = *ThreadState
// R4 = TODO: What's this supposed to be?
// R5 = caller pc
// R6 = addr of incoming arg list
// R8 contains addr of target function.
TEXT	racecallatomic<>(SB), NOSPLIT, $0-0
	// Trigger SIGSEGV early if address passed to atomic function is bad.
	MOVD	(R6), R7	// 1st arg is addr
	MOVD	(R7), R9	// segv here if addr is bad
	// Check that addr is within [arenastart, arenaend) or within [racedatastart, racedataend).
	MOVD	runtime·racearenastart(SB), R9
	CMP	R7, R9
	BLT	racecallatomic_data
	MOVD	runtime·racearenaend(SB), R9
	CMP	R7, R9
	BLT	racecallatomic_ok
racecallatomic_data:
	MOVD	runtime·racedatastart(SB), R9
	CMP	R7, R9
	BLT	racecallatomic_ignore
	MOVD	runtime·racedataend(SB), R9
	CMP	R7, R9
	BGE	racecallatomic_ignore
racecallatomic_ok:
	// Addr is within the good range, call the atomic function.
	MOVD    runtime·tls_g(SB), R10
	MOVD    0(R13)(R10*1), g
	MOVD    g_racectx(g), R3        // goroutine racectx aka *ThreadState
	MOVD	R8, R5			// pc is the function called
	MOVD	(R1), R4		// caller pc from stack
	BL	racecall<>(SB)		// BL needed to maintain stack consistency
	RET				//
racecallatomic_ignore:
	// Addr is outside the good range.
	// Call __tsan_go_ignore_sync_begin to ignore synchronization during the atomic op.
	// An attempt to synchronize on the address would cause crash.
	MOVD	R8, R15	// save the original function
	MOVD	R6, R17 // save the original arg list addr
	MOVD	$__tsan_go_ignore_sync_begin(SB), R8 // func addr to call
	MOVD    runtime·tls_g(SB), R10
	MOVD    0(R13)(R10*1), g
	MOVD    g_racectx(g), R3        // goroutine context
	BL	racecall<>(SB)
	MOVD	R15, R8	// restore the original function
	MOVD	R17, R6 // restore arg list addr
	// Call the atomic function.
	// racecall will call LLVM race code which might clobber r30 (g)
	MOVD    runtime·tls_g(SB), R10
        MOVD    0(R13)(R10*1), g

	MOVD	g_racectx(g), R3
	MOVD	R8, R4		// pc being called same TODO as above
	MOVD	(R1), R5	// caller pc from latest LR
	BL	racecall<>(SB)
	// Call __tsan_go_ignore_sync_end.
	MOVD	$__tsan_go_ignore_sync_end(SB), R8
	MOVD	g_racectx(g), R3	// goroutine context g should sitll be good?
	BL	racecall<>(SB)
	RET

// void runtime·racecall(void(*f)(...), ...)
// Calls C function f from race runtime and passes up to 4 arguments to it.
// The arguments are never heap-object-preserving pointers, so we pretend there are no arguments.
TEXT	runtime·racecall(SB), NOSPLIT, $0-0
	MOVD	fn+0(FP), R8
	MOVD	arg0+8(FP), R3
	MOVD	arg1+16(FP), R4
	MOVD	arg2+24(FP), R5
	MOVD	arg3+32(FP), R6
	JMP	racecall<>(SB)

// Finds g0 and sets its stack
// Arguments were loaded for call from Go to C
TEXT	racecall<>(SB), NOSPLIT, $0-0
	// Set the LR slot for the ppc64 ABI
	MOVD	LR, R10
	MOVD	R10, 0(R1)	// Go expectation
	MOVD	R10, 16(R1)	// C ABI
	// Get info from the current goroutine
	MOVD    runtime·tls_g(SB), R10	// g offset in TLS
	MOVD    0(R13)(R10*1), g	// R13 = current TLS
	MOVD	g_m(g), R7		// m for g
	MOVD	R1, R16			// callee-saved, preserved across C call
	MOVD	m_g0(R7), R10		// g0 for m
	CMP	R10, g			// same g0?
	BEQ	call			// already on g0
	MOVD	(g_sched+gobuf_sp)(R10), R1 // switch R1
call:
	MOVD	R8, CTR			// R8 = caller addr
	MOVD	R8, R12			// expected by PPC64 ABI
	BL	(CTR)
	XOR     R0, R0			// clear R0 on return from Clang
	MOVD	R16, R1			// restore R1; R16 nonvol in Clang
	MOVD    runtime·tls_g(SB), R10	// find correct g
	MOVD    0(R13)(R10*1), g
	MOVD	16(R1), R10		// LR was saved away, restore for return
	MOVD	R10, LR
	RET

// C->Go callback thunk that allows to call runtime·racesymbolize from C code.
// Direct Go->C race call has only switched SP, finish g->g0 switch by setting correct g.
// The overall effect of Go->C->Go call chain is similar to that of mcall.
// RARG0 contains command code. RARG1 contains command-specific context.
// See racecallback for command codes.
TEXT	runtime·racecallbackthunk(SB), NOSPLIT, $-8
	// Handle command raceGetProcCmd (0) here.
	// First, code below assumes that we are on curg, while raceGetProcCmd
	// can be executed on g0. Second, it is called frequently, so will
	// benefit from this fast path.
	XOR	R0, R0		// clear R0 since we came from C code
	CMP	R3, $0
	BNE	rest
	// g0 TODO: Don't modify g here since R30 is nonvolatile
	MOVD	g, R9
	MOVD    runtime·tls_g(SB), R10
	MOVD    0(R13)(R10*1), g
	MOVD	g_m(g), R3
	MOVD	m_p(R3), R3
	MOVD	p_racectx(R3), R3
	MOVD	R3, (R4)
	MOVD	R9, g		// restore R30 ??
	RET

	// This is all similar to what cgo does
	// Save registers according to the ppc64 ABI
rest:
	MOVD	LR, R10	// save link register
	MOVD	R10, 16(R1)
	MOVW	CR, R10
	MOVW	R10, 8(R1)
	MOVDU   R1, -336(R1) // Allocate frame needed for register save area

	MOVD    R14, 40(R1)
	MOVD    R15, 48(R1)
	MOVD    R16, 56(R1)
	MOVD    R17, 64(R1)
	MOVD    R18, 72(R1)
	MOVD    R19, 80(R1)
	MOVD    R20, 88(R1)
	MOVD    R21, 96(R1)
	MOVD    R22, 104(R1)
	MOVD    R23, 112(R1)
	MOVD    R24, 120(R1)
	MOVD    R25, 128(R1)
	MOVD    R26, 136(R1)
	MOVD    R27, 144(R1)
	MOVD    R28, 152(R1)
	MOVD    R29, 160(R1)
	MOVD    g, 168(R1) // R30
	MOVD    R31, 176(R1)
	FMOVD   F14, 184(R1)
	FMOVD   F15, 192(R1)
	FMOVD   F16, 200(R1)
	FMOVD   F17, 208(R1)
	FMOVD   F18, 216(R1)
	FMOVD   F19, 224(R1)
	FMOVD   F20, 232(R1)
	FMOVD   F21, 240(R1)
	FMOVD   F22, 248(R1)
	FMOVD   F23, 256(R1)
	FMOVD   F24, 264(R1)
	FMOVD   F25, 272(R1)
	FMOVD   F26, 280(R1)
	FMOVD   F27, 288(R1)
	FMOVD   F28, 296(R1)
	FMOVD   F29, 304(R1)
	FMOVD   F30, 312(R1)
	FMOVD   F31, 320(R1)

	MOVD    runtime·tls_g(SB), R10
	MOVD    0(R13)(R10*1), g

	MOVD	g_m(g), R7
	MOVD	m_g0(R7), g // set g = m-> g0
	MOVD	R3, cmd+0(FP) // can't use R1 here ?? use input args and assumer caller expects those?
	MOVD	R4, ctx+8(FP) // can't use R1 here ??
	BL	runtime·racecallback(SB)
	// All registers are clobbered after Go code, reload.
	MOVD    runtime·tls_g(SB), R10
        MOVD    0(R13)(R10*1), g

	MOVD	g_m(g), R7
	MOVD	m_curg(R7), g // restore g = m->curg
	MOVD    40(R1), R14
	MOVD    48(R1), R15
	MOVD    56(R1), R16
	MOVD    64(R1), R17
	MOVD    72(R1), R18
	MOVD    80(R1), R19
	MOVD    88(R1), R20
	MOVD    96(R1), R21
	MOVD    104(R1), R22
	MOVD    112(R1), R23
	MOVD    120(R1), R24
	MOVD    128(R1), R25
	MOVD    136(R1), R26
	MOVD    144(R1), R27
	MOVD    152(R1), R28
	MOVD    160(R1), R29
	MOVD    168(R1), g // R30
	MOVD    176(R1), R31
	FMOVD   184(R1), F14
	FMOVD   192(R1), F15
	FMOVD   200(R1), F16
	FMOVD   208(R1), F17
	FMOVD   216(R1), F18
	FMOVD   224(R1), F19
	FMOVD   232(R1), F20
	FMOVD   240(R1), F21
	FMOVD   248(R1), F22
	FMOVD   256(R1), F23
	FMOVD   264(R1), F24
	FMOVD   272(R1), F25
	FMOVD   280(R1), F26
	FMOVD   288(R1), F27
	FMOVD   296(R1), F28
	FMOVD   304(R1), F29
	FMOVD   312(R1), F30
	FMOVD   320(R1), F31

	ADD     $336, R1
	MOVD    8(R1), R10
	MOVFL   R10, $0xff // Restore of CR
	MOVD    16(R1), R10	// needed?
	MOVD    R10, LR
	RET

// tls_g, g value for each thread in TLS
GLOBL runtime·tls_g+0(SB), TLSBSS+DUPOK, $8
