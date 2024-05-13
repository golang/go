// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// See memmove Go doc for important implementation constraints.

// func memmove(to, from unsafe.Pointer, n uintptr)
TEXT runtime·memmove<ABIInternal>(SB), NOSPLIT|NOFRAME, $0-24
	BNE	R6, check
	RET

check:
	SGTU	R4, R5, R7
	BNE	R7, backward

	ADDV	R4, R6, R9 // end pointer

	// if the two pointers are not of same alignments, do byte copying
	SUBVU	R5, R4, R7
	AND	$7, R7
	BNE	R7, out

	// if less than 8 bytes, do byte copying
	SGTU	$8, R6, R7
	BNE	R7, out

	// do one byte at a time until 8-aligned
	AND	$7, R4, R8
	BEQ	R8, words
	MOVB	(R5), R7
	ADDV	$1, R5
	MOVB	R7, (R4)
	ADDV	$1, R4
	JMP	-6(PC)

words:
	// do 8 bytes at a time if there is room
	ADDV	$-7, R9, R6 // R6 is end pointer-7

	PCALIGN	$16
	SGTU	R6, R4, R8
	BEQ	R8, out
	MOVV	(R5), R7
	ADDV	$8, R5
	MOVV	R7, (R4)
	ADDV	$8, R4
	JMP	-6(PC)

out:
	BEQ	R4, R9, done
	MOVB	(R5), R7
	ADDV	$1, R5
	MOVB	R7, (R4)
	ADDV	$1, R4
	JMP	-5(PC)
done:
	RET

backward:
	ADDV	R6, R5 // from-end pointer
	ADDV	R4, R6, R9 // to-end pointer

	// if the two pointers are not of same alignments, do byte copying
	SUBVU	R9, R5, R7
	AND	$7, R7
	BNE	R7, out1

	// if less than 8 bytes, do byte copying
	SGTU	$8, R6, R7
	BNE	R7, out1

	// do one byte at a time until 8-aligned
	AND	$7, R9, R8
	BEQ	R8, words1
	ADDV	$-1, R5
	MOVB	(R5), R7
	ADDV	$-1, R9
	MOVB	R7, (R9)
	JMP	-6(PC)

words1:
	// do 8 bytes at a time if there is room
	ADDV	$7, R4, R6 // R6 is start pointer+7

	PCALIGN	$16
	SGTU	R9, R6, R8
	BEQ	R8, out1
	ADDV	$-8, R5
	MOVV	(R5), R7
	ADDV	$-8, R9
	MOVV	R7, (R9)
	JMP	-6(PC)

out1:
	BEQ	R4, R9, done1
	ADDV	$-1, R5
	MOVB	(R5), R7
	ADDV	$-1, R9
	MOVB	R7, (R9)
	JMP	-5(PC)
done1:
	RET
