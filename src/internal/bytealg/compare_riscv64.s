// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "asm_riscv64.h"
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
// return value in X10 (-1/0/1)
TEXT compare<>(SB),NOSPLIT|NOFRAME,$0
	BEQ	X10, X12, cmp_len

	MIN	X11, X13, X5
	BEQZ	X5, cmp_len

	MOV	$16, X6
	BLT	X5, X6, check8_unaligned

#ifndef hasV
	MOVB	internal∕cpu·RISCV64+const_offsetRISCV64HasV(SB), X6
	BEQZ	X6, compare_scalar
#endif

	// Use vector if not 8 byte aligned.
	OR	X10, X12, X6
	AND	$7, X6
	BNEZ	X6, vector_loop

	// Use scalar if 8 byte aligned and <= 128 bytes.
	SUB	$128, X5, X6
	BLEZ	X6, compare_scalar_aligned

	PCALIGN	$16
vector_loop:
	VSETVLI	X5, E8, M8, TA, MA, X6
	VLE8V	(X10), V8
	VLE8V	(X12), V16
	VMSNEVV	V8, V16, V0
	VFIRSTM	V0, X7
	BGEZ	X7, vector_not_eq
	ADD	X6, X10
	ADD	X6, X12
	SUB	X6, X5
	BNEZ	X5, vector_loop
	JMP	cmp_len

vector_not_eq:
	// Load first differing bytes in X8/X9.
	ADD	X7, X10
	ADD	X7, X12
	MOVBU	(X10), X8
	MOVBU	(X12), X9
	JMP	cmp

compare_scalar:
	MOV	$32, X6
	BLT	X5, X6, check8_unaligned

	// Check alignment - if alignment differs we have to do one byte at a time.
	AND	$7, X10, X7
	AND	$7, X12, X8
	BNE	X7, X8, check8_unaligned
	BEQZ	X7, compare32

	// Check one byte at a time until we reach 8 byte alignment.
	SUB	X7, X0, X7
	ADD	$8, X7, X7
	SUB	X7, X5, X5
align:
	SUB	$1, X7
	MOVBU	0(X10), X8
	MOVBU	0(X12), X9
	BNE	X8, X9, cmp
	ADD	$1, X10
	ADD	$1, X12
	BNEZ	X7, align

compare_scalar_aligned:
	MOV	$32, X6
	BLT	X5, X6, check16
compare32:
	MOV	0(X10), X15
	MOV	0(X12), X16
	MOV	8(X10), X17
	MOV	8(X12), X18
	BNE	X15, X16, cmp8a
	BNE	X17, X18, cmp8b
	MOV	16(X10), X15
	MOV	16(X12), X16
	MOV	24(X10), X17
	MOV	24(X12), X18
	BNE	X15, X16, cmp8a
	BNE	X17, X18, cmp8b
	ADD	$32, X10
	ADD	$32, X12
	SUB	$32, X5
	BGE	X5, X6, compare32
	BEQZ	X5, cmp_len

check16:
	MOV	$16, X6
	BLT	X5, X6, check8_unaligned
compare16:
	MOV	0(X10), X15
	MOV	0(X12), X16
	MOV	8(X10), X17
	MOV	8(X12), X18
	BNE	X15, X16, cmp8a
	BNE	X17, X18, cmp8b
	ADD	$16, X10
	ADD	$16, X12
	SUB	$16, X5
	BEQZ	X5, cmp_len

check8_unaligned:
	MOV	$8, X6
	BLT	X5, X6, check4_unaligned
compare8_unaligned:
	MOVBU	0(X10), X8
	MOVBU	1(X10), X15
	MOVBU	2(X10), X17
	MOVBU	3(X10), X19
	MOVBU	4(X10), X21
	MOVBU	5(X10), X23
	MOVBU	6(X10), X25
	MOVBU	7(X10), X29
	MOVBU	0(X12), X9
	MOVBU	1(X12), X16
	MOVBU	2(X12), X18
	MOVBU	3(X12), X20
	MOVBU	4(X12), X22
	MOVBU	5(X12), X24
	MOVBU	6(X12), X28
	MOVBU	7(X12), X30
	BNE	X8, X9, cmp1a
	BNE	X15, X16, cmp1b
	BNE	X17, X18, cmp1c
	BNE	X19, X20, cmp1d
	BNE	X21, X22, cmp1e
	BNE	X23, X24, cmp1f
	BNE	X25, X28, cmp1g
	BNE	X29, X30, cmp1h
	ADD	$8, X10
	ADD	$8, X12
	SUB	$8, X5
	BGE	X5, X6, compare8_unaligned
	BEQZ	X5, cmp_len

check4_unaligned:
	MOV	$4, X6
	BLT	X5, X6, compare1
compare4_unaligned:
	MOVBU	0(X10), X8
	MOVBU	1(X10), X15
	MOVBU	2(X10), X17
	MOVBU	3(X10), X19
	MOVBU	0(X12), X9
	MOVBU	1(X12), X16
	MOVBU	2(X12), X18
	MOVBU	3(X12), X20
	BNE	X8, X9, cmp1a
	BNE	X15, X16, cmp1b
	BNE	X17, X18, cmp1c
	BNE	X19, X20, cmp1d
	ADD	$4, X10
	ADD	$4, X12
	SUB	$4, X5
	BGE	X5, X6, compare4_unaligned

compare1:
	BEQZ	X5, cmp_len
	MOVBU	0(X10), X8
	MOVBU	0(X12), X9
	BNE	X8, X9, cmp
	ADD	$1, X10
	ADD	$1, X12
	SUB	$1, X5
	JMP	compare1

	// Compare 8 bytes of memory in X15/X16 that are known to differ.
cmp8a:
	MOV	X15, X17
	MOV	X16, X18

	// Compare 8 bytes of memory in X17/X18 that are known to differ.
cmp8b:
	MOV	$0xff, X19
cmp8_loop:
	AND	X17, X19, X8
	AND	X18, X19, X9
	BNE	X8, X9, cmp
	SLLI	$8, X19
	JMP	cmp8_loop

cmp1a:
	SLTU	X9, X8, X5
	SLTU	X8, X9, X6
	JMP	cmp_ret
cmp1b:
	SLTU	X16, X15, X5
	SLTU	X15, X16, X6
	JMP	cmp_ret
cmp1c:
	SLTU	X18, X17, X5
	SLTU	X17, X18, X6
	JMP	cmp_ret
cmp1d:
	SLTU	X20, X19, X5
	SLTU	X19, X20, X6
	JMP	cmp_ret
cmp1e:
	SLTU	X22, X21, X5
	SLTU	X21, X22, X6
	JMP	cmp_ret
cmp1f:
	SLTU	X24, X23, X5
	SLTU	X23, X24, X6
	JMP	cmp_ret
cmp1g:
	SLTU	X28, X25, X5
	SLTU	X25, X28, X6
	JMP	cmp_ret
cmp1h:
	SLTU	X30, X29, X5
	SLTU	X29, X30, X6
	JMP	cmp_ret

cmp_len:
	MOV	X11, X8
	MOV	X13, X9
cmp:
	SLTU	X9, X8, X5
	SLTU	X8, X9, X6
cmp_ret:
	SUB	X5, X6, X10
	RET
