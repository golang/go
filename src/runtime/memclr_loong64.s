// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

// func memclrNoHeapPointers(ptr unsafe.Pointer, n uintptr)
TEXT runtime·memclrNoHeapPointers(SB),NOSPLIT,$0-16
	MOVV	ptr+0(FP), R6
	MOVV	n+8(FP), R7
	ADDV	R6, R7, R4

	// if less than 8 bytes, do one byte at a time
	SGTU	$8, R7, R8
	BNE	R8, out

	// do one byte at a time until 8-aligned
	AND	$7, R6, R8
	BEQ	R8, words
	MOVB	R0, (R6)
	ADDV	$1, R6
	JMP	-4(PC)

words:
	// do 8 bytes at a time if there is room
	ADDV	$-7, R4, R7

	PCALIGN	$16
	SGTU	R7, R6, R8
	BEQ	R8, out
	MOVV	R0, (R6)
	ADDV	$8, R6
	JMP	-4(PC)

out:
	BEQ	R6, R4, done
	MOVB	R0, (R6)
	ADDV	$1, R6
	JMP	-3(PC)
done:
	RET
