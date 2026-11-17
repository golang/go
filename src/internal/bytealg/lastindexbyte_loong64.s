// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

// input:
//   R4 = b_base
//   R5 = b_len
//   R6 = b_cap (unused)
//   R7 = byte to find
TEXT ·LastIndexByte<ABIInternal>(SB),NOSPLIT,$0-40
	AND	$0xff, R7
	JMP	lastindexbytebody<>(SB)

// input:
//   R4 = s_base
//   R5 = s_len
//   R6 = byte to find
TEXT ·LastIndexByteString<ABIInternal>(SB),NOSPLIT,$0-32
	AND	$0xff, R6, R7	// byte to find
	JMP	lastindexbytebody<>(SB)

// input:
//   R4: b_base
//   R5: len
//   R7: byte to find
TEXT lastindexbytebody<>(SB),NOSPLIT,$0
	BEQ	R5, notfound	// len == 0

	MOVV	R4, R6		// store base for later
	ADDV	R4, R5, R8	// R8 = current right end (exclusive)

	MOVV	$32, R9
	BGE	R5, R9, lasx
tail:
	// Scalar backward scan, one 8-byte word at a time.
	// Bytes are extracted from the highest byte down so that
	// the last match in the word is found first.
	MOVV	$8, R9
	BLT	R5, R9, lt_8
generic8_loop:
	ADDV	$-8, R8
	MOVV	(R8), R10

	BSTRPICKV	$63, R10, $56, R11
	BEQ	R7, R11, found_add_7

	BSTRPICKV	$55, R10, $48, R11
	BEQ	R7, R11, found_add_6

	BSTRPICKV	$47, R10, $40, R11
	BEQ	R7, R11, found_add_5

	BSTRPICKV	$39, R10, $32, R11
	BEQ	R7, R11, found_add_4

	BSTRPICKV	$31, R10, $24, R11
	BEQ	R7, R11, found_add_3

	BSTRPICKV	$23, R10, $16, R11
	BEQ	R7, R11, found_add_2

	BSTRPICKV	$15, R10, $8, R11
	BEQ	R7, R11, found_add_1

	AND	$0xff, R10, R11
	BEQ	R7, R11, found_r8

	ADDV	$-8, R5
	BGE	R5, R9, generic8_loop

lt_8:
	// Fewer than 8 bytes left (possibly 0 after generic8_loop):
	// scan backward byte by byte.
	BEQ	R4, R8, notfound
	ADDV	$-1, R8
	MOVBU	(R8), R10
	BEQ	R7, R10, found_r8
	JMP	lt_8

found_add_7:
	ADDV	$7, R8
	JMP	found_r8
found_add_6:
	ADDV	$6, R8
	JMP	found_r8
found_add_5:
	ADDV	$5, R8
	JMP	found_r8
found_add_4:
	ADDV	$4, R8
	JMP	found_r8
found_add_3:
	ADDV	$3, R8
	JMP	found_r8
found_add_2:
	ADDV	$2, R8
	JMP	found_r8
found_add_1:
	ADDV	$1, R8
	JMP	found_r8

found_r8:
	MOVV	R8, R4		// R8 points at the matching byte
found:
	SUBV	R6, R4
	RET

notfound:
	MOVV	$-1, R4
	RET

lasx:
	MOVBU	internal∕cpu·Loong64+const_offsetLOONG64HasLASX(SB), R9
	BEQ	R9, lsx
	XVMOVQ	R7, X0.B32

	MOVV	$128, R9
	BLT	R5, R9, lasx32_entry
lasx128_loop:
	ADDV	$-128, R8
	XVMOVQ	0(R8), X1
	XVMOVQ	32(R8), X2
	XVMOVQ	64(R8), X3
	XVMOVQ	96(R8), X4

	// Check higher offsets first: the last match wins.
	XVSEQB	X4, X0, X5
	XVSETNEV	X5, FCC0
	BFPT	lasx_found_add_96

	XVSEQB	X3, X0, X5
	XVSETNEV	X5, FCC0
	BFPT	lasx_found_add_64

	XVSEQB	X2, X0, X5
	XVSETNEV	X5, FCC0
	BFPT	lasx_found_add_32

	XVSEQB	X1, X0, X5
	XVSETNEV	X5, FCC0
	BFPT	lasx_found_add_0

	ADDV	$-128, R5
	BGE	R5, R9, lasx128_loop

	BEQ	R5, notfound	// all bytes checked

	// 0 < R5 < 128
lasx32_entry:
	MOVV	$32, R9
	BLT	R5, R9, tail
lasx32_loop:
	ADDV	$-32, R8
	XVMOVQ	0(R8), X1

	XVSEQB	X1, X0, X5
	XVSETNEV	X5, FCC0
	BFPT	lasx_found_add_0

	ADDV	$-32, R5
	BGE	R5, R9, lasx32_loop

	BEQ	R5, notfound

	JMP	tail		// 0 < R5 < 32: finish with the scalar tail

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
	// X5 holds the compare result of the matching block.
	// Find the last set byte: scan 64-bit lanes from high to
	// low; a lane is non-zero iff its clz != 64. R11 = block
	// offset; add the lane base (24/16/8/0) for the first
	// non-zero lane. Byte index in lane = 7 - (clz>>3).
	MOVV	$64, R9
	XVMOVQ	X5.V[3], R10
	CLZV	R10, R10
	BNE	R10, R9, lasx_last_add_24

	XVMOVQ	X5.V[2], R10
	CLZV	R10, R10
	BNE	R10, R9, lasx_last_add_16

	XVMOVQ	X5.V[1], R10
	CLZV	R10, R10
	BNE	R10, R9, lasx_last_add_8

	XVMOVQ	X5.V[0], R10
	CLZV	R10, R10
	JMP	lasx_last_cal

lasx_last_add_24:
	ADDV	$24, R11
	JMP	lasx_last_cal
lasx_last_add_16:
	ADDV	$16, R11
	JMP	lasx_last_cal
lasx_last_add_8:
	ADDV	$8, R11

lasx_last_cal:
	SRLV	$3, R10
	MOVV	$7, R9
	SUBV	R10, R9		// byte index of last match in lane
	ADDV	R9, R11
	ADDV	R11, R8
	MOVV	R8, R4
	JMP	found

lsx:
	MOVBU	internal∕cpu·Loong64+const_offsetLOONG64HasLSX(SB), R9
	BEQ	R9, tail
	VMOVQ	R7, V0.B16

	MOVV	$64, R9
	BLT	R5, R9, lsx16_entry
lsx64_loop:
	ADDV	$-64, R8
	VMOVQ	0(R8), V1
	VMOVQ	16(R8), V2
	VMOVQ	32(R8), V3
	VMOVQ	48(R8), V4

	// Check higher offsets first: the last match wins.
	VSEQB	V4, V0, V5
	VSETNEV	V5, FCC0
	BFPT	lsx_found_add_48

	VSEQB	V3, V0, V5
	VSETNEV	V5, FCC0
	BFPT	lsx_found_add_32

	VSEQB	V2, V0, V5
	VSETNEV	V5, FCC0
	BFPT	lsx_found_add_16

	VSEQB	V1, V0, V5
	VSETNEV	V5, FCC0
	BFPT	lsx_found_add_0

	ADDV	$-64, R5
	BGE	R5, R9, lsx64_loop

	BEQ	R5, notfound

	// 0 < R5 < 64
lsx16_entry:
	MOVV	$16, R9
	BLT	R5, R9, tail
lsx16_loop:
	ADDV	$-16, R8
	VMOVQ	0(R8), V1

	VSEQB	V1, V0, V5
	VSETNEV	V5, FCC0
	BFPT	lsx_found_add_0

	ADDV	$-16, R5
	BGE	R5, R9, lsx16_loop

	BEQ	R5, notfound

	JMP	tail		// 0 < R5 < 16: finish with the scalar tail

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
	// V5 holds the compare result of the matching block.
	// Find the last set byte: scan 64-bit lanes from high to
	// low; a lane is non-zero iff its clz != 64. R11 = block
	// offset; add the lane base (8/0) for the first non-zero
	// lane. Byte index in lane = 7 - (clz>>3).
	MOVV	$64, R9

	VMOVQ	V5.V[1], R10
	CLZV	R10, R10
	BNE	R10, R9, lsx_last_add_8

	VMOVQ	V5.V[0], R10
	CLZV	R10, R10
	JMP	lsx_last_cal

lsx_last_add_8:
	ADDV	$8, R11

lsx_last_cal:
	SRLV	$3, R10
	MOVV	$7, R9
	SUBV	R10, R9		// byte index of last match in lane
	ADDV	R9, R11
	ADDV	R11, R8
	MOVV	R8, R4
	JMP	found
