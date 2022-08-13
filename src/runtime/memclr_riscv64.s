// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// See memclrNoHeapPointers Go doc for important implementation constraints.

// void runtime·memclrNoHeapPointers(void*, uintptr)
TEXT runtime·memclrNoHeapPointers<ABIInternal>(SB),NOSPLIT,$0-16
#ifndef GOEXPERIMENT_regabiargs
	MOV	ptr+0(FP), A0
	MOV	n+8(FP), A1
#endif
	ADD	A0, A1, T4

	// If less than eight bytes, do one byte at a time.
	SLTU	$8, A1, T3
	BNE	T3, ZERO, outcheck

	// Do one byte at a time until eight-aligned.
	JMP	aligncheck
align:
	MOVB	ZERO, (A0)
	ADD	$1, A0
aligncheck:
	AND	$7, A0, T3
	BNE	T3, ZERO, align

	// Do eight bytes at a time as long as there is room.
	ADD	$-7, T4, T5
	JMP	wordscheck
words:
	MOV	ZERO, (A0)
	ADD	$8, A0
wordscheck:
	SLTU	T5, A0, T3
	BNE	T3, ZERO, words

	JMP	outcheck
out:
	MOVB	ZERO, (A0)
	ADD	$1, A0
outcheck:
	BNE	A0, T4, out

done:
	RET
