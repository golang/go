// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build libfuzzer

#include "go_asm.h"
#include "textflag.h"

// Based on race_loong64.s; see commentary there.

#define RARG0 R4
#define RARG1 R5
#define RARG2 R6
#define RARG3 R7

#define REPEAT_2(a) a a
#define REPEAT_8(a) REPEAT_2(REPEAT_2(REPEAT_2(a)))
#define REPEAT_128(a) REPEAT_2(REPEAT_8(REPEAT_8(a)))

// void runtime·libfuzzerCall4(fn, hookId int, s1, s2 unsafe.Pointer, result uintptr)
// Calls C function fn from libFuzzer and passes 4 arguments to it.
TEXT	runtime·libfuzzerCall4<ABIInternal>(SB), NOSPLIT, $0-0
	MOVV	R4, R12	// fn
	MOVV	R5, RARG0	// hookId
	MOVV	R6, RARG1	// s1
	MOVV	R7, RARG2	// s2
	MOVV	R8, RARG3	// result

	MOVV	g_m(g), R13

	// Switch to g0 stack.
	MOVV	R3, R23	// callee-saved, preserved across the CALL
	MOVV	m_g0(R13), R14
	BEQ	R14, g, call	// already on g0
	MOVV	(g_sched+gobuf_sp)(R14), R3

call:
	JAL	(R12)
	MOVV	R23, R3
	RET

// void runtime·libfuzzerCallWithTwoByteBuffers(fn, start, end *byte)
// Calls C function fn from libFuzzer and passes 2 arguments of type *byte to it.
TEXT    runtime·libfuzzerCallWithTwoByteBuffers<ABIInternal>(SB), NOSPLIT, $0-0
	MOVV    R4, R12	// fn
	MOVV    R5, RARG0	// start
	MOVV    R6, RARG1	// end

	MOVV    g_m(g), R13

	// Switch to g0 stack.
	MOVV    R3, R23	// callee-saved, preserved across the CALL
	MOVV    m_g0(R13), R14
	BEQ	R14, g, call	// already on g0
	MOVV    (g_sched+gobuf_sp)(R14), R3

call:
	JAL	(R12)
	MOVV    R23, R3
	RET

// void runtime·libfuzzerCallTraceIntCmp(fn, arg0, arg1, fakePC uintptr)
// Calls C function fn from libFuzzer and passes 2 arguments to it after
// manipulating the return address so that libfuzzer's integer compare hooks
// work.
// The problem statement and solution are documented in detail in libfuzzer_amd64.s.
// See commentary there.
TEXT	runtime·libfuzzerCallTraceIntCmp<ABIInternal>(SB), NOSPLIT, $0-0
	MOVV	R4, R12	// fn
	MOVV	R5, RARG0	// arg0
	MOVV	R6, RARG1	// arg1
	// Save the original return address in a local variable
	MOVV	R1, savedRetAddr-8(SP)

	MOVV	g_m(g), R13

	// Switch to g0 stack.
	MOVV	R3, R23	// callee-saved, preserved across the CALL
	MOVV	m_g0(R13), R14
	BEQ	R14, g, call	// already on g0
	MOVV	(g_sched+gobuf_sp)(R14), R3

call:
	// Load address of the ret sled into the default register for the return
	// address.
	MOVV	$ret_sled(SB), R1
	// Clear the lowest 2 bits of fakePC. All Loong64 instructions are four
	// bytes long, so we cannot get better return address granularity than
	// multiples of 4.
	AND	$-4, R7
	// Load the address of the i'th return instruction from the return sled.
	// The index is given in the fakePC argument.
	ADDV	R7, R1
	// Call the function by jumping to it and reusing all registers except
	// for the modified return address register R1.
	JMP	(R12)

// The ret sled for Loong64 consists of 128 br instructions jumping to the
// end of the function. Each instruction is 4 bytes long. The sled thus has
// the same byte length of 4 * 128 = 512 as the x86_64 sled, but coarser
// granularity.
#define RET_SLED \
	JMP	end_of_function;

TEXT	ret_sled(SB), NOSPLIT, $0-0
	REPEAT_128(RET_SLED);

end_of_function:
	MOVV	R23, R3
	MOVV	savedRetAddr-8(SP), R1
	RET
