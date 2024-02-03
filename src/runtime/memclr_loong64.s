// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

// func memclrNoHeapPointers(ptr unsafe.Pointer, n uintptr)
TEXT runtime·memclrNoHeapPointers<ABIInternal>(SB),NOSPLIT,$0-16
#ifndef GOEXPERIMENT_regabiargs
	MOVV	ptr+0(FP), R4
	MOVV	n+8(FP), R5
#endif
	ADDV	R4, R5, R6

	// if less than 8 bytes, do one byte at a time
	SGTU	$8, R5, R8
	BNE	R8, out

	// do one byte at a time until 8-aligned
	AND	$7, R4, R8
	BEQ	R8, words
	MOVB	R0, (R4)
	ADDV	$1, R4
	JMP	-4(PC)

words:
	// do 8 bytes at a time if there is room
	ADDV	$-7, R6, R5

	PCALIGN	$16
	SGTU	R5, R4, R8
	BEQ	R8, out
	MOVV	R0, (R4)
	ADDV	$8, R4
	JMP	-4(PC)

out:
	BEQ	R4, R6, done
	MOVB	R0, (R4)
	ADDV	$1, R4
	JMP	-3(PC)
done:
	RET
