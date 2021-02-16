// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build mips mipsle

#include "textflag.h"

#ifdef GOARCH_mips
#define MOVWHI  MOVWL
#define MOVWLO  MOVWR
#else
#define MOVWHI  MOVWR
#define MOVWLO  MOVWL
#endif

// See memclrNoHeapPointers Go doc for important implementation constraints.

// func memclrNoHeapPointers(ptr unsafe.Pointer, n uintptr)
TEXT runtime·memclrNoHeapPointers(SB),NOSPLIT,$0-8
	MOVW	n+4(FP), R2
	MOVW	ptr+0(FP), R1

	SGTU	$4, R2, R3
	ADDU	R2, R1, R4
	BNE	R3, small_zero

ptr_align:
	AND	$3, R1, R3
	BEQ	R3, setup
	SUBU	R1, R0, R3
	AND	$3, R3		// R3 contains number of bytes needed to align ptr
	MOVWHI	R0, 0(R1)	// MOVWHI will write zeros up to next word boundary
	SUBU	R3, R2
	ADDU	R3, R1

setup:
	AND	$31, R2, R6
	AND	$3, R2, R5
	SUBU	R6, R4, R6	// end pointer for 32-byte chunks
	SUBU	R5, R4, R5	// end pointer for 4-byte chunks

large:
	BEQ	R1, R6, words
	MOVW	R0, 0(R1)
	MOVW	R0, 4(R1)
	MOVW	R0, 8(R1)
	MOVW	R0, 12(R1)
	MOVW	R0, 16(R1)
	MOVW	R0, 20(R1)
	MOVW	R0, 24(R1)
	MOVW	R0, 28(R1)
	ADDU	$32, R1
	JMP	large

words:
	BEQ	R1, R5, tail
	MOVW	R0, 0(R1)
	ADDU	$4, R1
	JMP	words

tail:
	BEQ	R1, R4, ret
	MOVWLO	R0, -1(R4)

ret:
	RET

small_zero:
	BEQ	R1, R4, ret
	MOVB	R0, 0(R1)
	ADDU	$1, R1
	JMP	small_zero
