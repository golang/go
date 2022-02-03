// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !faketime

#include "go_asm.h"
#include "textflag.h"
#include "time_windows.h"

TEXT time·now(SB),NOSPLIT|NOFRAME,$0-20
	MOVW    $0, R0
	MOVB    runtime·useQPCTime(SB), R0
	CMP	$0, R0
	BNE	useQPC
	MOVW	$_INTERRUPT_TIME, R3
loop:
	MOVW	time_hi1(R3), R1
	DMB	MB_ISH
	MOVW	time_lo(R3), R0
	DMB	MB_ISH
	MOVW	time_hi2(R3), R2
	CMP	R1, R2
	BNE	loop

	// wintime = R1:R0, multiply by 100
	MOVW	$100, R2
	MULLU	R0, R2, (R4, R3)    // R4:R3 = R1:R0 * R2
	MULA	R1, R2, R4, R4

	// wintime*100 = R4:R3
	MOVW	R3, mono+12(FP)
	MOVW	R4, mono+16(FP)

	MOVW	$_SYSTEM_TIME, R3
wall:
	MOVW	time_hi1(R3), R1
	DMB	MB_ISH
	MOVW	time_lo(R3), R0
	DMB	MB_ISH
	MOVW	time_hi2(R3), R2
	CMP	R1, R2
	BNE	wall

	// w = R1:R0 in 100ns untis
	// convert to Unix epoch (but still 100ns units)
	#define delta 116444736000000000
	SUB.S   $(delta & 0xFFFFFFFF), R0
	SBC     $(delta >> 32), R1

	// Convert to nSec
	MOVW    $100, R2
	MULLU   R0, R2, (R4, R3)    // R4:R3 = R1:R0 * R2
	MULA    R1, R2, R4, R4
	// w = R2:R1 in nSec
	MOVW    R3, R1	      // R4:R3 -> R2:R1
	MOVW    R4, R2

	// multiply nanoseconds by reciprocal of 10**9 (scaled by 2**61)
	// to get seconds (96 bit scaled result)
	MOVW	$0x89705f41, R3		// 2**61 * 10**-9
	MULLU	R1,R3,(R6,R5)		// R7:R6:R5 = R2:R1 * R3
	MOVW	$0,R7
	MULALU	R2,R3,(R7,R6)

	// unscale by discarding low 32 bits, shifting the rest by 29
	MOVW	R6>>29,R6		// R7:R6 = (R7:R6:R5 >> 61)
	ORR	R7<<3,R6
	MOVW	R7>>29,R7

	// subtract (10**9 * sec) from nsec to get nanosecond remainder
	MOVW	$1000000000, R5	// 10**9
	MULLU	R6,R5,(R9,R8)   // R9:R8 = R7:R6 * R5
	MULA	R7,R5,R9,R9
	SUB.S	R8,R1		// R2:R1 -= R9:R8
	SBC	R9,R2

	// because reciprocal was a truncated repeating fraction, quotient
	// may be slightly too small -- adjust to make remainder < 10**9
	CMP	R5,R1	// if remainder > 10**9
	SUB.HS	R5,R1   //    remainder -= 10**9
	ADD.HS	$1,R6	//    sec += 1

	MOVW	R6,sec_lo+0(FP)
	MOVW	R7,sec_hi+4(FP)
	MOVW	R1,nsec+8(FP)
	RET
useQPC:
	B	runtime·nowQPC(SB)		// tail call

