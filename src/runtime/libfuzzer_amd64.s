// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build libfuzzer

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

// Based on race_amd64.s; see commentary there.

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

// void runtime·libfuzzerCall4(fn, hookId int, s1, s2 unsafe.Pointer, result uintptr)
// Calls C function fn from libFuzzer and passes 4 arguments to it.
TEXT	runtime·libfuzzerCall4(SB), NOSPLIT, $0-40
	MOVQ	fn+0(FP), AX
	MOVQ	hookId+8(FP), RARG0
	MOVQ	s1+16(FP), RARG1
	MOVQ	s2+24(FP), RARG2
	MOVQ	result+32(FP), RARG3

	get_tls(R12)
	MOVQ	g(R12), R14
	MOVQ	g_m(R14), R13

	// Switch to g0 stack.
	MOVQ	SP, R12		// callee-saved, preserved across the CALL
	MOVQ	m_g0(R13), R10
	CMPQ	R10, R14
	JE	call	// already on g0
	MOVQ	(g_sched+gobuf_sp)(R10), SP
call:
	ANDQ	$~15, SP	// alignment for gcc ABI
	CALL	AX
	MOVQ	R12, SP
	RET

// void runtime·libfuzzerCallTraceIntCmp(fn, arg0, arg1, fakePC uintptr)
// Calls C function fn from libFuzzer and passes 2 arguments to it after
// manipulating the return address so that libfuzzer's integer compare hooks
// work
// libFuzzer's compare hooks obtain the caller's address from the compiler
// builtin __builtin_return_address. Since we invoke the hooks always
// from the same native function, this builtin would always return the same
// value. Internally, the libFuzzer hooks call through to the always inlined
// HandleCmp and thus can't be mimicked without patching libFuzzer.
//
// We solve this problem via an inline assembly trampoline construction that
// translates a runtime argument `fake_pc` in the range [0, 512) into a call to
// a hook with a fake return address whose lower 9 bits are `fake_pc` up to a
// constant shift. This is achieved by pushing a return address pointing into
// 512 ret instructions at offset `fake_pc` onto the stack and then jumping
// directly to the address of the hook.
//
// Note: We only set the lowest 9 bits of the return address since only these
// bits are used by the libFuzzer value profiling mode for integer compares, see
// https://github.com/llvm/llvm-project/blob/704d92607d26e696daba596b72cb70effe79a872/compiler-rt/lib/fuzzer/FuzzerTracePC.cpp#L390
// as well as
// https://github.com/llvm/llvm-project/blob/704d92607d26e696daba596b72cb70effe79a872/compiler-rt/lib/fuzzer/FuzzerValueBitMap.h#L34
// ValueProfileMap.AddValue() truncates its argument to 16 bits and shifts the
// PC to the left by log_2(128)=7, which means that only the lowest 16 - 7 bits
// of the return address matter. String compare hooks use the lowest 12 bits,
// but take the return address as an argument and thus don't require the
// indirection through a trampoline.
// TODO: Remove the inline assembly trampoline once a PC argument has been added to libfuzzer's int compare hooks.
TEXT	runtime·libfuzzerCallTraceIntCmp(SB), NOSPLIT, $0-32
	MOVQ	fn+0(FP), AX
	MOVQ	arg0+8(FP), RARG0
	MOVQ	arg1+16(FP), RARG1
	MOVQ	fakePC+24(FP), R8

	get_tls(R12)
	MOVQ	g(R12), R14
	MOVQ	g_m(R14), R13

	// Switch to g0 stack.
	MOVQ	SP, R12		// callee-saved, preserved across the CALL
	MOVQ	m_g0(R13), R10
	CMPQ	R10, R14
	JE	call	// already on g0
	MOVQ	(g_sched+gobuf_sp)(R10), SP
call:
	ANDQ	$~15, SP	// alignment for gcc ABI
	SUBQ	$8, SP
	// Load the address of the end of the function and push it into the stack.
	// This address will be jumped to after executing the return instruction
	// from the return sled. There we reset the stack pointer and return.
	MOVQ    $end_of_function<>(SB), BX
	PUSHQ   BX
	// Load the starting address of the return sled into BX.
	MOVQ    $ret_sled<>(SB), BX
	// Load the address of the i'th return instruction from the return sled.
	// The index is given in the fakePC argument.
	ADDQ    R8, BX
	PUSHQ   BX
	// Call the original function with the fakePC return address on the stack.
	// Function arguments arg0 and arg1 are passed in the registers specified
	// by the x64 calling convention.
	JMP     AX
// This code will not be executed and is only there to satisfy assembler
// check of a balanced stack.
not_reachable:
	POPQ    BX
	POPQ    BX
	RET

TEXT end_of_function<>(SB), NOSPLIT, $0-0
	MOVQ	R12, SP
	RET

#define REPEAT_8(a) a \
  a \
  a \
  a \
  a \
  a \
  a \
  a

#define REPEAT_512(a) REPEAT_8(REPEAT_8(REPEAT_8(a)))

TEXT ret_sled<>(SB), NOSPLIT, $0-0
	REPEAT_512(RET)

// void runtime·libfuzzerCallWithTwoByteBuffers(fn, start, end *byte)
// Calls C function fn from libFuzzer and passes 2 arguments of type *byte to it.
TEXT	runtime·libfuzzerCallWithTwoByteBuffers(SB), NOSPLIT, $0-24
	MOVQ	fn+0(FP), AX
	MOVQ	start+8(FP), RARG0
	MOVQ	end+16(FP), RARG1

	get_tls(R12)
	MOVQ	g(R12), R14
	MOVQ	g_m(R14), R13

	// Switch to g0 stack.
	MOVQ	SP, R12		// callee-saved, preserved across the CALL
	MOVQ	m_g0(R13), R10
	CMPQ	R10, R14
	JE	call	// already on g0
	MOVQ	(g_sched+gobuf_sp)(R10), SP
call:
	ANDQ	$~15, SP	// alignment for gcc ABI
	CALL	AX
	MOVQ	R12, SP
	RET
