// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build libfuzzer

#include "go_asm.h"
#include "textflag.h"

// Based on race_arm64.s; see commentary there.

#define RARG0 R0
#define RARG1 R1
#define RARG2 R2
#define RARG3 R3

#define REPEAT_2(a) a a
#define REPEAT_8(a) REPEAT_2(REPEAT_2(REPEAT_2(a)))
#define REPEAT_128(a) REPEAT_2(REPEAT_8(REPEAT_8(a)))

// void runtime·libfuzzerCallTraceIntCmp(fn, arg0, arg1, fakePC uintptr)
// Calls C function fn from libFuzzer and passes 2 arguments to it after
// manipulating the return address so that libfuzzer's integer compare hooks
// work.
// The problem statement and solution are documented in detail in libfuzzer_amd64.s.
// See commentary there.
TEXT	runtime·libfuzzerCallTraceIntCmp(SB), NOSPLIT, $8-32
	MOVD	fn+0(FP), R9
	MOVD	arg0+8(FP), RARG0
	MOVD	arg1+16(FP), RARG1
	MOVD	fakePC+24(FP), R8
	// Save the original return address in a local variable
	MOVD	R30, savedRetAddr-8(SP)

	MOVD	g_m(g), R10

	// Switch to g0 stack.
	MOVD	RSP, R19	// callee-saved, preserved across the CALL
	MOVD	m_g0(R10), R11
	CMP	R11, g
	BEQ	call	// already on g0
	MOVD	(g_sched+gobuf_sp)(R11), R12
	MOVD	R12, RSP
call:
	// Load address of the ret sled into the default register for the return
	// address.
	ADR	ret_sled, R30
	// Clear the lowest 2 bits of fakePC. All ARM64 instructions are four
	// bytes long, so we cannot get better return address granularity than
	// multiples of 4.
	AND	$-4, R8, R8
	// Add the offset of the fake_pc-th ret.
	ADD	R8, R30, R30
	// Call the function by jumping to it and reusing all registers except
	// for the modified return address register R30.
	JMP	(R9)

// The ret sled for ARM64 consists of 128 br instructions jumping to the
// end of the function. Each instruction is 4 bytes long. The sled thus
// has the same byte length of 4 * 128 = 512 as the x86_64 sled, but
// coarser granularity.
#define RET_SLED \
	JMP	end_of_function;

ret_sled:
	REPEAT_128(RET_SLED);

end_of_function:
	MOVD	R19, RSP
	MOVD	savedRetAddr-8(SP), R30
	RET

// void runtime·libfuzzerCall4(fn, hookId int, s1, s2 unsafe.Pointer, result uintptr)
// Calls C function fn from libFuzzer and passes 4 arguments to it.
TEXT	runtime·libfuzzerCall4(SB), NOSPLIT, $0-40
	MOVD	fn+0(FP), R9
	MOVD	hookId+8(FP), RARG0
	MOVD	s1+16(FP), RARG1
	MOVD	s2+24(FP), RARG2
	MOVD	result+32(FP), RARG3

	MOVD	g_m(g), R10

	// Switch to g0 stack.
	MOVD	RSP, R19	// callee-saved, preserved across the CALL
	MOVD	m_g0(R10), R11
	CMP	R11, g
	BEQ	call	// already on g0
	MOVD	(g_sched+gobuf_sp)(R11), R12
	MOVD	R12, RSP
call:
	BL	R9
	MOVD	R19, RSP
	RET

// void runtime·libfuzzerCallWithTwoByteBuffers(fn, start, end *byte)
// Calls C function fn from libFuzzer and passes 2 arguments of type *byte to it.
TEXT	runtime·libfuzzerCallWithTwoByteBuffers(SB), NOSPLIT, $0-24
	MOVD	fn+0(FP), R9
	MOVD	start+8(FP), R0
	MOVD	end+16(FP), R1

	MOVD	g_m(g), R10

	// Switch to g0 stack.
	MOVD	RSP, R19	// callee-saved, preserved across the CALL
	MOVD	m_g0(R10), R11
	CMP	R11, g
	BEQ	call	// already on g0
	MOVD	(g_sched+gobuf_sp)(R11), R12
	MOVD	R12, RSP
call:
	BL	R9
	MOVD	R19, RSP
	RET
