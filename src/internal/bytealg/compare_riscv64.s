// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

TEXT ·Compare<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-56
	// X10 = a_base
	// X11 = a_len
	// X12 = a_cap (unused)
	// X13 = b_base (want in X12)
	// X14 = b_len (want in X13)
	// X15 = b_cap (unused)
	MOV	X13, X12
	MOV	X14, X13
	JMP	compare<>(SB)

TEXT runtime·cmpstring<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-40
	// X10 = a_base
	// X11 = a_len
	// X12 = b_base
	// X13 = b_len
	JMP	compare<>(SB)

// On entry:
// X10 points to start of a
// X11 length of a
// X12 points to start of b
// X13 length of b
// for non-regabi X14 points to the address to store the return value (-1/0/1)
// for regabi the return value in X10
TEXT compare<>(SB),NOSPLIT|NOFRAME,$0
	BEQ	X10, X12, cmp_len

	MOV	X11, X5
	BGE	X13, X5, use_a_len // X5 = min(len(a), len(b))
	MOV	X13, X5
use_a_len:
	BEQZ	X5, cmp_len

	MOV	$32, X6
	BLT	X5, X6, loop4_check

	// Check alignment - if alignment differs we have to do one byte at a time.
	AND	$7, X10, X7
	AND	$7, X12, X8
	BNE	X7, X8, loop4_check
	BEQZ	X7, loop32_check

	// Check one byte at a time until we reach 8 byte alignment.
	SUB	X7, X5, X5
align:
	ADD	$-1, X7
	MOVBU	0(X10), X8
	MOVBU	0(X12), X9
	BNE	X8, X9, cmp
	ADD	$1, X10
	ADD	$1, X12
	BNEZ	X7, align

loop32_check:
	MOV	$32, X7
	BLT	X5, X7, loop16_check
loop32:
	MOV	0(X10), X15
	MOV	0(X12), X16
	MOV	8(X10), X17
	MOV	8(X12), X18
	BEQ	X15, X16, loop32a
	JMP	cmp8a
loop32a:
	BEQ	X17, X18, loop32b
	JMP	cmp8b
loop32b:
	MOV	16(X10), X15
	MOV	16(X12), X16
	MOV	24(X10), X17
	MOV	24(X12), X18
	BEQ	X15, X16, loop32c
	JMP	cmp8a
loop32c:
	BEQ	X17, X18, loop32d
	JMP	cmp8b
loop32d:
	ADD	$32, X10
	ADD	$32, X12
	ADD	$-32, X5
	BGE	X5, X7, loop32
	BEQZ	X5, cmp_len

loop16_check:
	MOV	$16, X6
	BLT	X5, X6, loop4_check
loop16:
	MOV	0(X10), X15
	MOV	0(X12), X16
	MOV	8(X10), X17
	MOV	8(X12), X18
	BEQ	X15, X16, loop16a
	JMP	cmp8a
loop16a:
	BEQ	X17, X18, loop16b
	JMP	cmp8b
loop16b:
	ADD	$16, X10
	ADD	$16, X12
	ADD	$-16, X5
	BGE	X5, X6, loop16
	BEQZ	X5, cmp_len

loop4_check:
	MOV	$4, X6
	BLT	X5, X6, loop1
loop4:
	MOVBU	0(X10), X8
	MOVBU	0(X12), X9
	MOVBU	1(X10), X15
	MOVBU	1(X12), X16
	BEQ	X8, X9, loop4a
	SLTU	X9, X8, X5
	SLTU	X8, X9, X6
	JMP	cmp_ret
loop4a:
	BEQ	X15, X16, loop4b
	SLTU	X16, X15, X5
	SLTU	X15, X16, X6
	JMP	cmp_ret
loop4b:
	MOVBU	2(X10), X21
	MOVBU	2(X12), X22
	MOVBU	3(X10), X23
	MOVBU	3(X12), X24
	BEQ	X21, X22, loop4c
	SLTU	X22, X21, X5
	SLTU	X21, X22, X6
	JMP	cmp_ret
loop4c:
	BEQ	X23, X24, loop4d
	SLTU	X24, X23, X5
	SLTU	X23, X24, X6
	JMP	cmp_ret
loop4d:
	ADD	$4, X10
	ADD	$4, X12
	ADD	$-4, X5
	BGE	X5, X6, loop4

loop1:
	BEQZ	X5, cmp_len
	MOVBU	0(X10), X8
	MOVBU	0(X12), X9
	BNE	X8, X9, cmp
	ADD	$1, X10
	ADD	$1, X12
	ADD	$-1, X5
	JMP	loop1

	// Compare 8 bytes of memory in X15/X16 that are known to differ.
cmp8a:
	MOV	$0xff, X19
cmp8a_loop:
	AND	X15, X19, X8
	AND	X16, X19, X9
	BNE	X8, X9, cmp
	SLLI	$8, X19
	JMP	cmp8a_loop

	// Compare 8 bytes of memory in X17/X18 that are known to differ.
cmp8b:
	MOV	$0xff, X19
cmp8b_loop:
	AND	X17, X19, X8
	AND	X18, X19, X9
	BNE	X8, X9, cmp
	SLLI	$8, X19
	JMP	cmp8b_loop

cmp_len:
	MOV	X11, X8
	MOV	X13, X9
cmp:
	SLTU	X9, X8, X5
	SLTU	X8, X9, X6
cmp_ret:
	SUB	X5, X6, X10
	RET
