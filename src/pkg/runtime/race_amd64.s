// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build race

#include "zasm_GOOS_GOARCH.h"
#include "funcdata.h"
#include "../../cmd/ld/textflag.h"

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
	MOVQ	cp+16(FP), RARG3
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
	// Check that addr is within [arenastart, arenaend) or within [noptrdata, enoptrbss).
	CMPQ	RARG1, runtime·racearenastart(SB)
	JB	racecalladdr_data
	CMPQ	RARG1, runtime·racearenaend(SB)
	JB	racecalladdr_call
racecalladdr_data:
	MOVQ	$runtime·noptrdata(SB), R13
	CMPQ	RARG1, R13
	JB	racecalladdr_ret
	MOVQ	$runtime·enoptrbss(SB), R13
	CMPQ	RARG1, R13
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
