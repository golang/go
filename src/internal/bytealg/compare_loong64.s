// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

TEXT ·Compare<ABIInternal>(SB),NOSPLIT,$0-56
	// R4 = a_base
	// R5 = a_len
	// R6 = a_cap (unused)
	// R7 = b_base (want in R6)
	// R8 = b_len (want in R7)
	// R9 = b_cap (unused)
	MOVV	R7, R6
	MOVV	R8, R7
	JMP	cmpbody<>(SB)

TEXT runtime·cmpstring<ABIInternal>(SB),NOSPLIT,$0-40
	// R4 = a_base
	// R5 = a_len
	// R6 = b_base
	// R7 = b_len
	JMP	cmpbody<>(SB)

// input:
//    R4: points to the start of a
//    R5: length of a
//    R6: points to the start of b
//    R7: length of b
// for regabi the return value (-1/0/1) in R4
TEXT cmpbody<>(SB),NOSPLIT|NOFRAME,$0
	BEQ	R4, R6, cmp_len	// same start of a and b, then compare lengths

	SGTU	R5, R7, R9
	BNE	R9, b_lt_a
	MOVV	R5, R14
	JMP	entry

b_lt_a:
	MOVV	R7, R14

entry:
	BEQ	R14, cmp_len	// minlength is 0

	MOVV	$32, R15
	BGE	R14, R15, lasx
tail:
	MOVV	$8, R15
	BLT	R14, R15, lt_8
generic8_loop:
	MOVV	(R4), R10
	MOVV	(R6), R11
	BEQ	R10, R11, generic8_equal

cmp8:
	AND	$0xff, R10, R16
	AND	$0xff, R11, R17
	BNE	R16, R17, cmp_byte

	BSTRPICKV	$15, R10, $8, R16
	BSTRPICKV	$15, R11, $8, R17
	BNE	R16, R17, cmp_byte

	BSTRPICKV	$23, R10, $16, R16
	BSTRPICKV	$23, R11, $16, R17
	BNE	R16, R17, cmp_byte

	BSTRPICKV	$31, R10, $24, R16
	BSTRPICKV	$31, R11, $24, R17
	BNE	R16, R17, cmp_byte

	BSTRPICKV	$39, R10, $32, R16
	BSTRPICKV	$39, R11, $32, R17
	BNE	R16, R17, cmp_byte

	BSTRPICKV	$47, R10, $40, R16
	BSTRPICKV	$47, R11, $40, R17
	BNE	R16, R17, cmp_byte

	BSTRPICKV	$55, R10, $48, R16
	BSTRPICKV	$55, R11, $48, R17
	BNE	R16, R17, cmp_byte

	BSTRPICKV	$63, R10, $56, R16
	BSTRPICKV	$63, R11, $56, R17
	BNE	R16, R17, cmp_byte

generic8_equal:
	ADDV	$-8, R14
	BEQ	R14, cmp_len
	ADDV	$8, R4
	ADDV	$8, R6
	BGE	R14, R15, generic8_loop

lt_8:
	MOVV	$4, R15
	BLT	R14, R15, lt_4

	MOVWU	(R4), R10
	MOVWU	(R6), R11
	BEQ	R10, R11, lt_8_equal

	AND	$0xff, R10, R16
	AND	$0xff, R11, R17
	BNE	R16, R17, cmp_byte

	BSTRPICKV	$15, R10, $8, R16
	BSTRPICKV	$15, R11, $8, R17
	BNE	R16, R17, cmp_byte

	BSTRPICKV	$23, R10, $16, R16
	BSTRPICKV	$23, R11, $16, R17
	BNE	R16, R17, cmp_byte

	BSTRPICKV	$31, R10, $24, R16
	BSTRPICKV	$31, R11, $24, R17
	BNE	R16, R17, cmp_byte

lt_8_equal:
	ADDV	$-4, R14
	BEQ	R14, cmp_len
	ADDV	$4, R4
	ADDV	$4, R6

lt_4:
	MOVV	$2, R15
	BLT	R14, R15, lt_2

	MOVHU	(R4), R10
	MOVHU	(R6), R11
	BEQ	R10, R11, lt_4_equal

	AND	$0xff, R10, R16
	AND	$0xff, R11, R17
	BNE	R16, R17, cmp_byte

	BSTRPICKV	$15, R10, $8, R16
	BSTRPICKV	$15, R11, $8, R17
	BNE	R16, R17, cmp_byte

lt_4_equal:
	ADDV	$-2, R14
	BEQ	R14, cmp_len
	ADDV	$2, R4
	ADDV	$2, R6

lt_2:
	MOVBU	(R4), R16
	MOVBU	(R6), R17
	BNE	R16, R17, cmp_byte
	JMP	cmp_len

	// Compare 1 byte taken from R16/R17 that are known to differ.
cmp_byte:
	SGTU	R16, R17, R4	// R4 = 1 if (R16 > R17)
	BNE	R0, R4, ret
	MOVV	$-1, R4
	RET

cmp_len:
	SGTU	R5, R7, R8
	SGTU	R7, R5, R9
	SUBV	R9, R8, R4

ret:
	RET

lasx:
	MOVV	$64, R20
	MOVBU	internal∕cpu·Loong64+const_offsetLOONG64HasLASX(SB), R9
	BEQ	R9, lsx

	MOVV	$128, R15
	BLT	R14, R15, lasx32_loop
lasx128_loop:
	XVMOVQ	(R4), X0
	XVMOVQ	(R6), X1
	XVSEQB	X0, X1, X0
	XVSETANYEQB	X0, FCC0
	BFPT	lasx_found_0

	XVMOVQ	32(R4), X0
	XVMOVQ	32(R6), X1
	XVSEQB	X0, X1, X0
	XVSETANYEQB	X0, FCC0
	BFPT	lasx_found_32

	XVMOVQ	64(R4), X0
	XVMOVQ	64(R6), X1
	XVSEQB	X0, X1, X0
	XVSETANYEQB	X0, FCC0
	BFPT	lasx_found_64

	XVMOVQ	96(R4), X0
	XVMOVQ	96(R6), X1
	XVSEQB	X0, X1, X0
	XVSETANYEQB	X0, FCC0
	BFPT	lasx_found_96

	ADDV	$-128, R14
	BEQ	R14, cmp_len
	ADDV	$128, R4
	ADDV	$128, R6
	BGE	R14, R15, lasx128_loop

	MOVV	$32, R15
	BLT	R14, R15, tail
lasx32_loop:
	XVMOVQ	(R4), X0
	XVMOVQ	(R6), X1
	XVSEQB	X0, X1, X0
	XVSETANYEQB	X0, FCC0
	BFPT	lasx_found_0

	ADDV	$-32, R14
	BEQ	R14, cmp_len
	ADDV	$32, R4
	ADDV	$32, R6
	BGE	R14, R15, lasx32_loop
	JMP	tail

lasx_found_0:
	MOVV	R0, R11
	JMP	lasx_find_byte

lasx_found_32:
	MOVV	$32, R11
	JMP	lasx_find_byte

lasx_found_64:
	MOVV	$64, R11
	JMP	lasx_find_byte

lasx_found_96:
	MOVV	$96, R11

lasx_find_byte:
	XVMOVQ	X0.V[0], R10
	CTOV	R10, R10
	BNE	R10, R20, find_byte
	ADDV	$8, R11

	XVMOVQ	X0.V[1], R10
	CTOV	R10, R10
	BNE	R10, R20, find_byte
	ADDV	$8, R11

	XVMOVQ	X0.V[2], R10
	CTOV	R10, R10
	BNE	R10, R20, find_byte
	ADDV	$8, R11

	XVMOVQ	X0.V[3], R10
	CTOV	R10, R10
	JMP	find_byte

lsx:
	MOVBU	internal∕cpu·Loong64+const_offsetLOONG64HasLSX(SB), R9
	BEQ	R9, generic32_loop

	MOVV	$64, R15
	BLT	R14, R15, lsx16_loop
lsx64_loop:
	VMOVQ	(R4), V0
	VMOVQ	(R6), V1
	VSEQB	V0, V1, V0
	VSETANYEQB	V0, FCC0
	BFPT	lsx_found_0

	VMOVQ	16(R4), V0
	VMOVQ	16(R6), V1
	VSEQB	V0, V1, V0
	VSETANYEQB	V0, FCC0
	BFPT	lsx_found_16

	VMOVQ	32(R4), V0
	VMOVQ	32(R6), V1
	VSEQB	V0, V1, V0
	VSETANYEQB	V0, FCC0
	BFPT	lsx_found_32

	VMOVQ	48(R4), V0
	VMOVQ	48(R6), V1
	VSEQB	V0, V1, V0
	VSETANYEQB	V0, FCC0
	BFPT	lsx_found_48

	ADDV	$-64, R14
	BEQ	R14, cmp_len
	ADDV	$64, R4
	ADDV	$64, R6
	BGE	R14, R15, lsx64_loop

	MOVV	$16, R15
	BLT	R14, R15, tail
lsx16_loop:
	VMOVQ	(R4), V0
	VMOVQ	(R6), V1
	VSEQB	V0, V1, V0
	VSETANYEQB	V0, FCC0
	BFPT	lsx_found_0

	ADDV	$-16, R14
	BEQ	R14, cmp_len
	ADDV	$16, R4
	ADDV	$16, R6
	BGE	R14, R15, lsx16_loop
	JMP	tail

lsx_found_0:
	MOVV	R0, R11
	JMP	lsx_find_byte

lsx_found_16:
	MOVV	$16, R11
	JMP	lsx_find_byte

lsx_found_32:
	MOVV	$32, R11
	JMP	lsx_find_byte

lsx_found_48:
	MOVV	$48, R11

lsx_find_byte:
	VMOVQ	V0.V[0], R10
	CTOV	R10, R10
	BNE	R10, R20, find_byte
	ADDV	$8, R11

	VMOVQ	V0.V[1], R10
	CTOV	R10, R10

find_byte:
	SRLV	$3, R10
	ADDV	R10, R11
	ADDV	R11, R4
	ADDV	R11, R6
	MOVB	(R4), R16
	MOVB	(R6), R17
	JMP	cmp_byte

generic32_loop:
	MOVV	(R4), R10
	MOVV	(R6), R11
	BNE	R10, R11, cmp8
	MOVV	8(R4), R10
	MOVV	8(R6), R11
	BNE	R10, R11, cmp8
	MOVV	16(R4), R10
	MOVV	16(R6), R11
	BNE	R10, R11, cmp8
	MOVV	24(R4), R10
	MOVV	24(R6), R11
	BNE	R10, R11, cmp8
	ADDV	$-32, R14
	BEQ	R14, cmp_len
	ADDV	$32, R4
	ADDV	$32, R6
	MOVV	$32, R15
	BGE	R14, R15, generic32_loop
	JMP	tail
