// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !faketime

#include "go_asm.h"
#include "textflag.h"
#include "funcdata.h"
#include "time_windows.h"

TEXT time·now(SB),NOSPLIT,$8-20
	NO_LOCAL_POINTERS
	CALL	runtime·nanotime1(SB)
	MOVL	0(SP), AX
	MOVL	4(SP), DX
	MOVL	AX, mono+12(FP)
	MOVL	DX, mono+16(FP)

wall:
	MOVL	(_SYSTEM_TIME+time_hi1), CX
	MOVL	(_SYSTEM_TIME+time_lo), AX
	MOVL	(_SYSTEM_TIME+time_hi2), DX
	CMPL	CX, DX
	JNE	wall

	// w = DX:AX
	// convert to Unix epoch (but still 100ns units)
	#define delta 116444736000000000
	SUBL	$(delta & 0xFFFFFFFF), AX
	SBBL $(delta >> 32), DX

	// nano/100 = DX:AX
	// split into two decimal halves by div 1e9.
	// (decimal point is two spots over from correct place,
	// but we avoid overflow in the high word.)
	MOVL	$1000000000, CX
	DIVL	CX
	MOVL	AX, DI
	MOVL	DX, SI

	// DI = nano/100/1e9 = nano/1e11 = sec/100, DX = SI = nano/100%1e9
	// split DX into seconds and nanoseconds by div 1e7 magic multiply.
	MOVL	DX, AX
	MOVL	$1801439851, CX
	MULL	CX
	SHRL	$22, DX
	MOVL	DX, BX
	IMULL	$10000000, DX
	MOVL	SI, CX
	SUBL	DX, CX

	// DI = sec/100 (still)
	// BX = (nano/100%1e9)/1e7 = (nano/1e9)%100 = sec%100
	// CX = (nano/100%1e9)%1e7 = (nano%1e9)/100 = nsec/100
	// store nsec for return
	IMULL	$100, CX
	MOVL	CX, nsec+8(FP)

	// DI = sec/100 (still)
	// BX = sec%100
	// construct DX:AX = 64-bit sec and store for return
	MOVL	$0, DX
	MOVL	$100, AX
	MULL	DI
	ADDL	BX, AX
	ADCL	$0, DX
	MOVL	AX, sec+0(FP)
	MOVL	DX, sec+4(FP)
	RET
