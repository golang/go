// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !faketime

#include "go_asm.h"
#include "textflag.h"
#include "time_windows.h"

TEXT time·now(SB),NOSPLIT|NOFRAME,$0-24
	MOVB    runtime·useQPCTime(SB), R0
	CMP	$0, R0
	BNE	useQPC

	MOVD	$_INTERRUPT_TIME, R3
	MOVD	time_lo(R3), R0
	MOVD	$100, R1
	MUL	R1, R0
	MOVD	R0, mono+16(FP)

	MOVD	$_SYSTEM_TIME, R3
	MOVD	time_lo(R3), R0
	// convert to Unix epoch (but still 100ns units)
	#define delta 116444736000000000
	SUB	$delta, R0
	// Convert to nSec
	MOVD	$100, R1
	MUL	R1, R0

	// Code stolen from compiler output for:
	//
	//	var x uint64
	//	func f() (sec uint64, nsec uint32) { return x / 1000000000, uint32(x % 1000000000) }
	//
	LSR	$1, R0, R1
	MOVD	$-8543223759426509416, R2
	UMULH	R1, R2, R1
	LSR	$28, R1, R1
	MOVD	R1, sec+0(FP)
	MOVD	$1000000000, R2
	MSUB	R1, R0, R2, R0
	MOVW	R0, nsec+8(FP)
	RET
useQPC:
	B	runtime·nowQPC(SB)		// tail call

