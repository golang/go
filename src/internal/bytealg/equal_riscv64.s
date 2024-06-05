// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

#define	CTXT	S10

// func memequal(a, b unsafe.Pointer, size uintptr) bool
TEXT runtime·memequal<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-25
	// X10 = a_base
	// X11 = b_base
	// X12 = size
	JMP	memequal<>(SB)

// func memequal_varlen(a, b unsafe.Pointer) bool
TEXT runtime·memequal_varlen<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-17
	MOV	8(CTXT), X12    // compiler stores size at offset 8 in the closure
	// X10 = a_base
	// X11 = b_base
	JMP	memequal<>(SB)

// On entry X10 and X11 contain pointers, X12 contains length.
// For non-regabi X13 contains address for return value.
// For regabi return value in X10.
TEXT memequal<>(SB),NOSPLIT|NOFRAME,$0
	BEQ	X10, X11, eq

	MOV	$32, X23
	BLT	X12, X23, loop4_check

	// Check alignment - if alignment differs we have to do one byte at a time.
	AND	$7, X10, X9
	AND	$7, X11, X19
	BNE	X9, X19, loop4_check
	BEQZ	X9, loop32_check

	// Check one byte at a time until we reach 8 byte alignment.
	SUB	X9, X0, X9
	ADD	$8, X9, X9
	SUB	X9, X12, X12
align:
	SUB	$1, X9
	MOVBU	0(X10), X19
	MOVBU	0(X11), X20
	BNE	X19, X20, not_eq
	ADD	$1, X10
	ADD	$1, X11
	BNEZ	X9, align

loop32_check:
	MOV	$32, X9
	BLT	X12, X9, loop16_check
loop32:
	MOV	0(X10), X19
	MOV	0(X11), X20
	MOV	8(X10), X21
	MOV	8(X11), X22
	BNE	X19, X20, not_eq
	BNE	X21, X22, not_eq
	MOV	16(X10), X14
	MOV	16(X11), X15
	MOV	24(X10), X16
	MOV	24(X11), X17
	BNE	X14, X15, not_eq
	BNE	X16, X17, not_eq
	ADD	$32, X10
	ADD	$32, X11
	SUB	$32, X12
	BGE	X12, X9, loop32
	BEQZ	X12, eq

loop16_check:
	MOV	$16, X23
	BLT	X12, X23, loop4_check
loop16:
	MOV	0(X10), X19
	MOV	0(X11), X20
	MOV	8(X10), X21
	MOV	8(X11), X22
	BNE	X19, X20, not_eq
	BNE	X21, X22, not_eq
	ADD	$16, X10
	ADD	$16, X11
	SUB	$16, X12
	BGE	X12, X23, loop16
	BEQZ	X12, eq

loop4_check:
	MOV	$4, X23
	BLT	X12, X23, loop1
loop4:
	MOVBU	0(X10), X19
	MOVBU	0(X11), X20
	MOVBU	1(X10), X21
	MOVBU	1(X11), X22
	BNE	X19, X20, not_eq
	BNE	X21, X22, not_eq
	MOVBU	2(X10), X14
	MOVBU	2(X11), X15
	MOVBU	3(X10), X16
	MOVBU	3(X11), X17
	BNE	X14, X15, not_eq
	BNE	X16, X17, not_eq
	ADD	$4, X10
	ADD	$4, X11
	SUB	$4, X12
	BGE	X12, X23, loop4

loop1:
	BEQZ	X12, eq
	MOVBU	0(X10), X19
	MOVBU	0(X11), X20
	BNE	X19, X20, not_eq
	ADD	$1, X10
	ADD	$1, X11
	SUB	$1, X12
	JMP	loop1

not_eq:
	MOVB	ZERO, X10
	RET
eq:
	MOV	$1, X10
	RET
