// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !faketime

#include "go_asm.h"
#include "textflag.h"
#include "funcdata.h"
#include "time_windows.h"

TEXT time·now<ABIInternal>(SB),NOSPLIT,$0-0
	NO_LOCAL_POINTERS
	BL	runtime·nanotime1<ABIInternal>(SB)
	MOVD	R0, R2

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
	MOVD	$-8543223759426509416, R3
	UMULH	R1, R3, R1
	LSR	$28, R1, R1
	MOVD	$1000000000, R3
	MSUB	R1, R0, R3, R3
	MOVD	R1, R0
	MOVD	R3, R1
	RET

