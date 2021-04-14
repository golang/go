// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build mips64 mips64le

#include "textflag.h"

// See memmove Go doc for important implementation constraints.

// func memmove(to, from unsafe.Pointer, n uintptr)
TEXT runtime·memmove(SB), NOSPLIT|NOFRAME, $0-24
	MOVV	to+0(FP), R1
	MOVV	from+8(FP), R2
	MOVV	n+16(FP), R3
	BNE	R3, check
	RET

check:
	SGTU	R1, R2, R4
	BNE	R4, backward

	ADDV	R1, R3, R6 // end pointer

	// if the two pointers are not of same alignments, do byte copying
	SUBVU	R2, R1, R4
	AND	$7, R4
	BNE	R4, out

	// if less than 8 bytes, do byte copying
	SGTU	$8, R3, R4
	BNE	R4, out

	// do one byte at a time until 8-aligned
	AND	$7, R1, R5
	BEQ	R5, words
	MOVB	(R2), R4
	ADDV	$1, R2
	MOVB	R4, (R1)
	ADDV	$1, R1
	JMP	-6(PC)

words:
	// do 8 bytes at a time if there is room
	ADDV	$-7, R6, R3 // R3 is end pointer-7

	SGTU	R3, R1, R5
	BEQ	R5, out
	MOVV	(R2), R4
	ADDV	$8, R2
	MOVV	R4, (R1)
	ADDV	$8, R1
	JMP	-6(PC)

out:
	BEQ	R1, R6, done
	MOVB	(R2), R4
	ADDV	$1, R2
	MOVB	R4, (R1)
	ADDV	$1, R1
	JMP	-5(PC)
done:
	RET

backward:
	ADDV	R3, R2 // from-end pointer
	ADDV	R1, R3, R6 // to-end pointer

	// if the two pointers are not of same alignments, do byte copying
	SUBVU	R6, R2, R4
	AND	$7, R4
	BNE	R4, out1

	// if less than 8 bytes, do byte copying
	SGTU	$8, R3, R4
	BNE	R4, out1

	// do one byte at a time until 8-aligned
	AND	$7, R6, R5
	BEQ	R5, words1
	ADDV	$-1, R2
	MOVB	(R2), R4
	ADDV	$-1, R6
	MOVB	R4, (R6)
	JMP	-6(PC)

words1:
	// do 8 bytes at a time if there is room
	ADDV	$7, R1, R3 // R3 is start pointer+7

	SGTU	R6, R3, R5
	BEQ	R5, out1
	ADDV	$-8, R2
	MOVV	(R2), R4
	ADDV	$-8, R6
	MOVV	R4, (R6)
	JMP	-6(PC)

out1:
	BEQ	R1, R6, done1
	ADDV	$-1, R2
	MOVB	(R2), R4
	ADDV	$-1, R6
	MOVB	R4, (R6)
	JMP	-5(PC)
done1:
	RET
