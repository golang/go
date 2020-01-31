// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// See memmove Go doc for important implementation constraints.

// void runtime·memmove(void*, void*, uintptr)
TEXT runtime·memmove(SB),NOSPLIT,$-0-24
	MOV	to+0(FP), T0
	MOV	from+8(FP), T1
	MOV	n+16(FP), T2
	ADD	T1, T2, T5

	// If the destination is ahead of the source, start at the end of the
	// buffer and go backward.
	BLTU	T1, T0, b

	// If less than eight bytes, do one byte at a time.
	SLTU	$8, T2, T3
	BNE	T3, ZERO, f_outcheck

	// Do one byte at a time until from is eight-aligned.
	JMP	f_aligncheck
f_align:
	MOVB	(T1), T3
	MOVB	T3, (T0)
	ADD	$1, T0
	ADD	$1, T1
f_aligncheck:
	AND	$7, T1, T3
	BNE	T3, ZERO, f_align

	// Do eight bytes at a time as long as there is room.
	ADD	$-7, T5, T6
	JMP	f_wordscheck
f_words:
	MOV	(T1), T3
	MOV	T3, (T0)
	ADD	$8, T0
	ADD	$8, T1
f_wordscheck:
	SLTU	T6, T1, T3
	BNE	T3, ZERO, f_words

	// Finish off the remaining partial word.
	JMP 	f_outcheck
f_out:
	MOVB	(T1), T3
	MOVB	T3, (T0)
	ADD	$1, T0
	ADD	$1, T1
f_outcheck:
	BNE	T1, T5, f_out

	RET

b:
	ADD	T0, T2, T4
	// If less than eight bytes, do one byte at a time.
	SLTU	$8, T2, T3
	BNE	T3, ZERO, b_outcheck

	// Do one byte at a time until from+n is eight-aligned.
	JMP	b_aligncheck
b_align:
	ADD	$-1, T4
	ADD	$-1, T5
	MOVB	(T5), T3
	MOVB	T3, (T4)
b_aligncheck:
	AND	$7, T5, T3
	BNE	T3, ZERO, b_align

	// Do eight bytes at a time as long as there is room.
	ADD	$7, T1, T6
	JMP	b_wordscheck
b_words:
	ADD	$-8, T4
	ADD	$-8, T5
	MOV	(T5), T3
	MOV	T3, (T4)
b_wordscheck:
	SLTU	T5, T6, T3
	BNE	T3, ZERO, b_words

	// Finish off the remaining partial word.
	JMP	b_outcheck
b_out:
	ADD	$-1, T4
	ADD	$-1, T5
	MOVB	(T5), T3
	MOVB	T3, (T4)
b_outcheck:
	BNE	T5, T1, b_out

	RET
