// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// See memclrNoHeapPointers Go doc for important implementation constraints.

// void runtime·memclrNoHeapPointers(void*, uintptr)
TEXT runtime·memclrNoHeapPointers(SB),NOSPLIT,$0-16
	MOV	ptr+0(FP), T1
	MOV	n+8(FP), T2
	ADD	T1, T2, T4

	// If less than eight bytes, do one byte at a time.
	SLTU	$8, T2, T3
	BNE	T3, ZERO, outcheck

	// Do one byte at a time until eight-aligned.
	JMP	aligncheck
align:
	MOVB	ZERO, (T1)
	ADD	$1, T1
aligncheck:
	AND	$7, T1, T3
	BNE	T3, ZERO, align

	// Do eight bytes at a time as long as there is room.
	ADD	$-7, T4, T5
	JMP	wordscheck
words:
	MOV	ZERO, (T1)
	ADD	$8, T1
wordscheck:
	SLTU	T5, T1, T3
	BNE	T3, ZERO, words

	JMP	outcheck
out:
	MOVB	ZERO, (T1)
	ADD	$1, T1
outcheck:
	BNE	T1, T4, out

done:
	RET
