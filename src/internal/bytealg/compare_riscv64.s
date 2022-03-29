// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

TEXT ·Compare(SB),NOSPLIT|NOFRAME,$0-56
	MOV	a_base+0(FP), X5
	MOV	a_len+8(FP), X6
	MOV	b_base+24(FP), X7
	MOV	b_len+32(FP), X8
	MOV	$ret+48(FP), X9
	JMP	compare<>(SB)

TEXT runtime·cmpstring(SB),NOSPLIT|NOFRAME,$0-40
	MOV	a_base+0(FP), X5
	MOV	a_len+8(FP), X6
	MOV	b_base+16(FP), X7
	MOV	b_len+24(FP), X8
	MOV	$ret+32(FP), X9
	JMP	compare<>(SB)

// On entry:
// X5 points to start of a
// X6 length of a
// X7 points to start of b
// X8 length of b
// X9 points to the address to store the return value (-1/0/1)
TEXT compare<>(SB),NOSPLIT|NOFRAME,$0
	BEQ	X5, X7, cmp_len

	MOV	X6, X10
	BGE	X8, X10, use_a_len // X10 = min(len(a), len(b))
	MOV	X8, X10
use_a_len:
	BEQZ	X10, cmp_len

	MOV	$32, X11
	BLT	X10, X11, loop4_check

	// Check alignment - if alignment differs we have to do one byte at a time.
	AND	$3, X5, X12
	AND	$3, X7, X13
	BNE	X12, X13, loop4_check
	BEQZ	X12, loop32_check

	// Check one byte at a time until we reach 8 byte alignment.
	SUB	X12, X10, X10
align:
	ADD	$-1, X12
	MOVBU	0(X5), X13
	MOVBU	0(X7), X14
	BNE	X13, X14, cmp
	ADD	$1, X5
	ADD	$1, X7
	BNEZ	X12, align

loop32_check:
	MOV	$32, X12
	BLT	X10, X12, loop16_check
loop32:
	MOV	0(X5), X15
	MOV	0(X7), X16
	MOV	8(X5), X17
	MOV	8(X7), X18
	BEQ	X15, X16, loop32a
	JMP	cmp8a
loop32a:
	BEQ	X17, X18, loop32b
	JMP	cmp8b
loop32b:
	MOV	16(X5), X15
	MOV	16(X7), X16
	MOV	24(X5), X17
	MOV	24(X7), X18
	BEQ	X15, X16, loop32c
	JMP	cmp8a
loop32c:
	BEQ	X17, X18, loop32d
	JMP	cmp8b
loop32d:
	ADD	$32, X5
	ADD	$32, X7
	ADD	$-32, X10
	BGE	X10, X12, loop32
	BEQZ	X10, cmp_len

loop16_check:
	MOV	$16, X11
	BLT	X10, X11, loop4_check
loop16:
	MOV	0(X5), X15
	MOV	0(X7), X16
	MOV	8(X5), X17
	MOV	8(X7), X18
	BEQ	X15, X16, loop16a
	JMP	cmp8a
loop16a:
	BEQ	X17, X18, loop16b
	JMP	cmp8b
loop16b:
	ADD	$16, X5
	ADD	$16, X7
	ADD	$-16, X10
	BGE	X10, X11, loop16
	BEQZ	X10, cmp_len

loop4_check:
	MOV	$4, X11
	BLT	X10, X11, loop1
loop4:
	MOVBU	0(X5), X13
	MOVBU	0(X7), X14
	MOVBU	1(X5), X15
	MOVBU	1(X7), X16
	BEQ	X13, X14, loop4a
	SLTU	X14, X13, X10
	SLTU	X13, X14, X11
	JMP	cmp_ret
loop4a:
	BEQ	X15, X16, loop4b
	SLTU	X16, X15, X10
	SLTU	X15, X16, X11
	JMP	cmp_ret
loop4b:
	MOVBU	2(X5), X21
	MOVBU	2(X7), X22
	MOVBU	3(X5), X23
	MOVBU	3(X7), X24
	BEQ	X21, X22, loop4c
	SLTU	X22, X21, X10
	SLTU	X21, X22, X11
	JMP	cmp_ret
loop4c:
	BEQ	X23, X24, loop4d
	SLTU	X24, X23, X10
	SLTU	X23, X24, X11
	JMP	cmp_ret
loop4d:
	ADD	$4, X5
	ADD	$4, X7
	ADD	$-4, X10
	BGE	X10, X11, loop4

loop1:
	BEQZ	X10, cmp_len
	MOVBU	0(X5), X13
	MOVBU	0(X7), X14
	BNE	X13, X14, cmp
	ADD	$1, X5
	ADD	$1, X7
	ADD	$-1, X10
	JMP	loop1

	// Compare 8 bytes of memory in X15/X16 that are known to differ.
cmp8a:
	MOV	$0xff, X19
cmp8a_loop:
	AND	X15, X19, X13
	AND	X16, X19, X14
	BNE	X13, X14, cmp
	SLLI	$8, X19
	JMP	cmp8a_loop

	// Compare 8 bytes of memory in X17/X18 that are known to differ.
cmp8b:
	MOV	$0xff, X19
cmp8b_loop:
	AND	X17, X19, X13
	AND	X18, X19, X14
	BNE	X13, X14, cmp
	SLLI	$8, X19
	JMP	cmp8b_loop

cmp_len:
	MOV	X6, X13
	MOV	X8, X14
cmp:
	SLTU	X14, X13, X10
	SLTU	X13, X14, X11
cmp_ret:
	SUB	X10, X11, X12
	MOV	X12, (X9)
	RET
