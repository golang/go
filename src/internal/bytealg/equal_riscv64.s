// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

#define	CTXT	S10

// func memequal(a, b unsafe.Pointer, size uintptr) bool
TEXT runtime·memequal(SB),NOSPLIT|NOFRAME,$0-25
	MOV	a+0(FP), X5
	MOV	b+8(FP), X6
	MOV	size+16(FP), X7
	MOV	$ret+24(FP), X19
	JMP	memequal<>(SB)

// func memequal_varlen(a, b unsafe.Pointer) bool
TEXT runtime·memequal_varlen(SB),NOSPLIT|NOFRAME,$0-17
	MOV	a+0(FP), X5
	MOV	b+8(FP), X6
	MOV	8(CTXT), X7    // compiler stores size at offset 8 in the closure
	MOV	$ret+16(FP), X19
	JMP	memequal<>(SB)

// On entry X5 and X6 contain pointers, X7 contains length.
// X19 contains address for return value.
TEXT memequal<>(SB),NOSPLIT|NOFRAME,$0
	BEQ	X5, X6, eq

	MOV	$32, X8
	BLT	X7, X8, loop4_check

	// Check alignment - if alignment differs we have to do one byte at a time.
	AND	$3, X5, X9
	AND	$3, X6, X10
	BNE	X9, X10, loop4_check
	BEQZ	X9, loop32_check

	// Check one byte at a time until we reach 8 byte alignment.
	SUB	X9, X7, X7
align:
	ADD	$-1, X9
	MOVBU	0(X5), X10
	MOVBU	0(X6), X11
	BNE	X10, X11, not_eq
	ADD	$1, X5
	ADD	$1, X6
	BNEZ	X9, align

loop32_check:
	MOV	$32, X9
	BLT	X7, X9, loop16_check
loop32:
	MOV	0(X5), X10
	MOV	0(X6), X11
	MOV	8(X5), X12
	MOV	8(X6), X13
	BNE	X10, X11, not_eq
	BNE	X12, X13, not_eq
	MOV	16(X5), X14
	MOV	16(X6), X15
	MOV	24(X5), X16
	MOV	24(X6), X17
	BNE	X14, X15, not_eq
	BNE	X16, X17, not_eq
	ADD	$32, X5
	ADD	$32, X6
	ADD	$-32, X7
	BGE	X7, X9, loop32
	BEQZ	X7, eq

loop16_check:
	MOV	$16, X8
	BLT	X7, X8, loop4_check
loop16:
	MOV	0(X5), X10
	MOV	0(X6), X11
	MOV	8(X5), X12
	MOV	8(X6), X13
	BNE	X10, X11, not_eq
	BNE	X12, X13, not_eq
	ADD	$16, X5
	ADD	$16, X6
	ADD	$-16, X7
	BGE	X7, X8, loop16
	BEQZ	X7, eq

loop4_check:
	MOV	$4, X8
	BLT	X7, X8, loop1
loop4:
	MOVBU	0(X5), X10
	MOVBU	0(X6), X11
	MOVBU	1(X5), X12
	MOVBU	1(X6), X13
	BNE	X10, X11, not_eq
	BNE	X12, X13, not_eq
	MOVBU	2(X5), X14
	MOVBU	2(X6), X15
	MOVBU	3(X5), X16
	MOVBU	3(X6), X17
	BNE	X14, X15, not_eq
	BNE	X16, X17, not_eq
	ADD	$4, X5
	ADD	$4, X6
	ADD	$-4, X7
	BGE	X7, X8, loop4

loop1:
	BEQZ	X7, eq
	MOVBU	0(X5), X10
	MOVBU	0(X6), X11
	BNE	X10, X11, not_eq
	ADD	$1, X5
	ADD	$1, X6
	ADD	$-1, X7
	JMP	loop1

not_eq:
	MOV	$0, X5
	MOVB	X5, (X19)
	RET
eq:
	MOV	$1, X5
	MOVB	X5, (X19)
	RET
