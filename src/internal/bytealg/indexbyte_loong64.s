// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

// input:
//   R4 = b_base
//   R5 = b_len
//   R6 = b_cap (unused)
//   R7 = byte to find
TEXT ·IndexByte<ABIInternal>(SB),NOSPLIT,$0-40
	AND	$0xff, R7
	JMP	indexbytebody<>(SB)

// input:
//   R4 = s_base
//   R5 = s_len
//   R6 = byte to find
TEXT ·IndexByteString<ABIInternal>(SB),NOSPLIT,$0-32
	AND	$0xff, R6, R7	// byte to find
	JMP	indexbytebody<>(SB)

// input:
//   R4: b_base
//   R5: len
//   R7: byte to find
TEXT indexbytebody<>(SB),NOSPLIT,$0
	BEQ	R5, notfound	// len == 0

	MOVV	R4, R6		// store base for later
	ADDV	R4, R5, R8	// end

	MOVV	$32, R9
	BGE	R5, R9, lasx
tail:
	// Constants for the SWAR zero-byte test in generic8_loop:
	// R12 = byte to find broadcast to all 8 bytes.
	MOVV	$0x0101010101010101, R13
	MULV	R7, R13, R12
	MOVV	$0x8080808080808080, R14

	MOVV	$8, R9
	BLT	R5, R9, lt_8
generic8_loop:
	MOVV	(R4), R10

	XOR	R12, R10, R11	// bytes equal to R7 become 0
	// haszero(t) = (t - 0x01..) & ~t & 0x80..
	SUBV	R13, R11, R15
	NOR	R11, R0, R16	// ~t
	AND	R15, R16, R15
	AND	R14, R15
	BNE	R15, generic8_found

	ADDV	$8, R4
	ADDV	$-8, R5
	BGE	R5, R9, generic8_loop

	// 0 <= R5 < 8: overlapping load of the last 8 bytes.
	// Safe: at least one full 8-byte chunk was checked, so
	// R4-8 stays inside the original buffer. Already-checked
	// bytes contain no match, so the lowest set bit of the
	// mask is the first match in the unchecked tail.
	ADDV	R4, R5, R15
	MOVV	-8(R15), R10
	XOR	R12, R10, R11
	SUBV	R13, R11, R15
	NOR	R11, R0, R16
	AND	R15, R16, R15
	AND	R14, R15
	BEQ	R15, notfound
	CTZV	R15, R15
	SRLV	$3, R15		// byte index within last 8 bytes
	ADDV	R4, R5, R4
	ADDV	$-8, R4
	ADDV	R15, R4
	JMP	found

generic8_found:
	CTZV	R15, R15
	SRLV	$3, R15		// byte index of first match
	ADDV	R15, R4
	JMP	found

lt_8:
	BEQ	R4, R8, notfound
	MOVBU	(R4), R10
	BEQ	R7, R10, found
	ADDV	$1, R4
	JMP	lt_8

found:
	SUBV	R6, R4
	RET

notfound:
	MOVV	$-1, R4
	RET

lasx:
	MOVBU   internal∕cpu·Loong64+const_offsetLOONG64HasLASX(SB), R9
	BEQ     R9, lsx
	XVMOVQ	R7, X0.B32

	MOVV	$128, R9
	BLT	R5, R9, lasx32_loop
lasx128_loop:
	XVMOVQ	0(R4), X1
	XVMOVQ	32(R4), X2
	XVMOVQ	64(R4), X3
	XVMOVQ	96(R4), X4

	XVSEQB	X1, X0, X1
	XVSETNEV	X1, FCC0
	BFPT	lasx_found_add_0

	XVSEQB	X2, X0, X1
	XVSETNEV	X1, FCC0
	BFPT	lasx_found_add_32

	XVSEQB	X3, X0, X1
	XVSETNEV	X1, FCC0
	BFPT	lasx_found_add_64

	XVSEQB	X4, X0, X1
	XVSETNEV	X1, FCC0
	BFPT	lasx_found_add_96

	ADDV	$128, R4
	ADDV	$-128, R5
	BGE	R5, R9, lasx128_loop

	BEQ	R5, notfound

	MOVV	$32, R9
	BLT	R5, R9, tail
lasx32_loop:
	XVMOVQ	0(R4), X1

	XVSEQB	X1, X0, X1
	XVSETNEV	X1, FCC0
	BFPT	lasx_found_add_0

	ADDV	$32, R4
	ADDV	$-32, R5
	BGE	R5, R9, lasx32_loop

	BEQ	R5, notfound

	JMP	tail

lasx_found_add_0:
	MOVV	R0, R11
	JMP	lasx_index_cal

lasx_found_add_32:
	MOVV	$32, R11
	JMP	lasx_index_cal

lasx_found_add_64:
	MOVV	$64, R11
	JMP	lasx_index_cal

lasx_found_add_96:
	MOVV	$96, R11
	JMP	lasx_index_cal

lasx_index_cal:
	MOVV	$64, R9
	XVMOVQ	X1.V[0], R10
	CTZV	R10, R10
	BNE	R10, R9, index_cal
	ADDV	$8, R11

	XVMOVQ	X1.V[1], R10
	CTZV	R10, R10
	BNE	R10, R9, index_cal
	ADDV	$8, R11

	XVMOVQ	X1.V[2], R10
	CTZV	R10, R10
	BNE	R10, R9, index_cal
	ADDV	$8, R11

	XVMOVQ	X1.V[3], R10
	CTZV	R10, R10
	JMP	index_cal

lsx:
	MOVBU   internal∕cpu·Loong64+const_offsetLOONG64HasLSX(SB), R9
	BEQ     R9, tail
	VMOVQ	R7, V0.B16

	MOVV	$64, R9
	BLT	R5, R9, lsx16_loop
lsx64_loop:
	VMOVQ	0(R4), V1
	VMOVQ	16(R4), V2
	VMOVQ	32(R4), V3
	VMOVQ	48(R4), V4

	VSEQB	V1, V0, V1
	VSETNEV	V1, FCC0
	BFPT	lsx_found_add_0

	VSEQB	V2, V0, V1
	VSETNEV	V1, FCC0
	BFPT	lsx_found_add_16

	VSEQB	V3, V0, V1
	VSETNEV	V1, FCC0
	BFPT	lsx_found_add_32

	VSEQB	V4, V0, V1
	VSETNEV	V1, FCC0
	BFPT	lsx_found_add_48

	ADDV	$64, R4
	ADDV	$-64, R5
	BGE	R5, R9, lsx64_loop

	BEQ	R5, notfound

	MOVV	$16, R9
	BLT	R5, R9, tail
lsx16_loop:
	VMOVQ	0(R4), V1

	VSEQB	V1, V0, V1
	VSETNEV	V1, FCC0
	BFPT	lsx_found_add_0

	ADDV	$16, R4
	ADDV	$-16, R5
	BGE	R5, R9, lsx16_loop

	BEQ	R5, notfound

	JMP	tail

lsx_found_add_0:
	MOVV	R0, R11
	JMP	lsx_index_cal

lsx_found_add_16:
	MOVV	$16, R11
	JMP	lsx_index_cal

lsx_found_add_32:
	MOVV	$32, R11
	JMP	lsx_index_cal

lsx_found_add_48:
	MOVV	$48, R11
	JMP	lsx_index_cal

lsx_index_cal:
	MOVV	$64, R9

	VMOVQ	V1.V[0], R10
	CTZV	R10, R10
	BNE	R10, R9, index_cal
	ADDV	$8, R11

	VMOVQ	V1.V[1], R10
	CTZV	R10, R10
	JMP	index_cal

index_cal:
	SRLV	$3, R10
	ADDV	R11, R10
	ADDV	R10, R4
	JMP	found
