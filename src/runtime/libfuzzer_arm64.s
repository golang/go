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

// func runtime·libfuzzerCall(fn, arg0, arg1 uintptr)
// Calls C function fn from libFuzzer and passes 2 arguments to it.
TEXT	runtime·libfuzzerCall(SB), NOSPLIT, $0-24
	MOVD	fn+0(FP), R9
	MOVD	arg0+8(FP), RARG0
	MOVD	arg1+16(FP), RARG1

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
