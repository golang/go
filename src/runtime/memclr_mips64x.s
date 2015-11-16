// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build mips64 mips64le

#include "textflag.h"

// void runtime·memclr(void*, uintptr)
TEXT runtime·memclr(SB),NOSPLIT,$0-16
	MOVV	ptr+0(FP), R1
	MOVV	n+8(FP), R2
	ADDV	R1, R2, R4

	// if less than 8 bytes, do one byte at a time
	SGTU	$8, R2, R3
	BNE	R3, out

	// do one byte at a time until 8-aligned
	AND	$7, R1, R3
	BEQ	R3, words
	MOVB	R0, (R1)
	ADDV	$1, R1
	JMP	-4(PC)

words:
	// do 8 bytes at a time if there is room
	ADDV	$-7, R4, R2

	SGTU	R2, R1, R3
	BEQ	R3, out
	MOVV	R0, (R1)
	ADDV	$8, R1
	JMP	-4(PC)

out:
	BEQ	R1, R4, done
	MOVB	R0, (R1)
	ADDV	$1, R1
	JMP	-3(PC)
done:
	RET
