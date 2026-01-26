// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

#define	REGCTXT	R29

// memequal(a, b unsafe.Pointer, size uintptr) bool
TEXT runtime·memequal<ABIInternal>(SB),NOSPLIT|NOFRAME,$0
	// R4 = a_base
	// R5 = b_base
	// R6 = size
	JMP	equalbody<>(SB)

// memequal_varlen(a, b unsafe.Pointer) bool
TEXT runtime·memequal_varlen<ABIInternal>(SB),NOSPLIT,$0
	// R4 = a_base
	// R5 = b_base
	MOVV	8(REGCTXT), R6    // compiler stores size at offset 8 in the closure
	JMP	equalbody<>(SB)

// input:
//   R4 = a_base
//   R5 = b_base
//   R6 = size
TEXT equalbody<>(SB),NOSPLIT|NOFRAME,$0
	// a_base == b_base
	BEQ	R4, R5, equal
	// 0 bytes
	BEQ	R6, equal

	MOVV	$64, R7
	BGE	R6, R7, lasx

	// size < 64 bytes
tail:
	MOVV	$16, R7
	BLT	R6, R7, lt_16
generic16_loop:
	ADDV	$-16, R6
	MOVV	0(R4), R8
	MOVV	8(R4), R9
	MOVV	0(R5), R10
	MOVV	8(R5), R11
	BNE	R8, R10, not_equal
	BNE	R9, R11, not_equal
	BEQ	R6, equal
	ADDV	$16, R4
	ADDV	$16, R5
	BGE	R6, R7, generic16_loop

	// size < 16 bytes
lt_16:
	MOVV	$8, R7
	BLT	R6, R7, lt_8
	ADDV	$-8, R6
	MOVV	0(R4), R8
	MOVV	0(R5), R9
	BNE	R8, R9, not_equal
	BEQ	R6, equal
	ADDV	$8, R4
	ADDV	$8, R5

	// size < 8 bytes
lt_8:
	MOVV	$4, R7
	BLT	R6, R7, lt_4
	ADDV	$-4, R6
	MOVW	0(R4), R8
	MOVW	0(R5), R9
	BNE	R8, R9, not_equal
	BEQ	R6, equal
	ADDV	$4, R4
	ADDV	$4, R5

	// size < 4 bytes
lt_4:
	MOVV	$2, R7
	BLT	R6, R7, lt_2
	ADDV	$-2, R6
	MOVH	0(R4), R8
	MOVH	0(R5), R9
	BNE	R8, R9, not_equal
	BEQ	R6, equal
	ADDV	$2, R4
	ADDV	$2, R5

	// size < 2 bytes
lt_2:
	MOVB	0(R4), R8
	MOVB	0(R5), R9
	BNE	R8, R9, not_equal

equal:
	MOVV	$1, R4
	RET

not_equal:
	MOVV	R0, R4
	RET

	// Implemented using 256-bit SIMD instructions
lasx:
	MOVBU   internal∕cpu·Loong64+const_offsetLOONG64HasLASX(SB), R7
	BEQ	R7, lsx

lasx256:
	MOVV	$256, R7
	BLT	R6, R7, lasx64
lasx256_loop:
	ADDV	$-256, R6
	XVMOVQ	0(R4), X0
	XVMOVQ	32(R4), X1
	XVMOVQ	64(R4), X2
	XVMOVQ	96(R4), X3
	XVMOVQ	128(R4), X4
	XVMOVQ	160(R4), X5
	XVMOVQ	192(R4), X6
	XVMOVQ	224(R4), X7
	XVMOVQ	0(R5), X8
	XVMOVQ	32(R5), X9
	XVMOVQ	64(R5), X10
	XVMOVQ	96(R5), X11
	XVMOVQ	128(R5), X12
	XVMOVQ	160(R5), X13
	XVMOVQ	192(R5), X14
	XVMOVQ	224(R5), X15
	XVSEQV	X0, X8, X0
	XVSEQV	X1, X9, X1
	XVSEQV	X2, X10, X2
	XVSEQV	X3, X11, X3
	XVSEQV	X4, X12, X4
	XVSEQV	X5, X13, X5
	XVSEQV	X6, X14, X6
	XVSEQV	X7, X15, X7
	XVANDV	X0, X1, X0
	XVANDV	X2, X3, X2
	XVANDV	X4, X5, X4
	XVANDV	X6, X7, X6
	XVANDV	X0, X2, X0
	XVANDV	X4, X6, X4
	XVANDV	X0, X4, X0
	XVSETALLNEV	X0, FCC0
	BFPF	not_equal
	BEQ	R6, equal
	ADDV	$256, R4
	ADDV	$256, R5
	BGE	R6, R7, lasx256_loop

lasx64:
	MOVV	$64, R7
	BLT	R6, R7, tail
lasx64_loop:
	ADDV	$-64, R6
	XVMOVQ	0(R4), X0
	XVMOVQ	32(R4), X1
	XVMOVQ	0(R5), X2
	XVMOVQ	32(R5), X3
	XVSEQV	X0, X2, X0
	XVSEQV	X1, X3, X1
	XVANDV	X0, X1, X0
	XVSETALLNEV	X0, FCC0
	BFPF	not_equal
	BEQ	R6, equal
	ADDV	$64, R4
	ADDV	$64, R5
	BGE	R6, R7, lasx64_loop
	JMP	tail

	// Implemented using 128-bit SIMD instructions
lsx:
	MOVBU	internal∕cpu·Loong64+const_offsetLOONG64HasLSX(SB), R7
	BEQ	R7, generic64_loop

lsx128:
	MOVV	$128, R7
	BLT	R6, R7, lsx32
lsx128_loop:
	ADDV	$-128, R6
	VMOVQ	0(R4), V0
	VMOVQ	16(R4), V1
	VMOVQ	32(R4), V2
	VMOVQ	48(R4), V3
	VMOVQ	64(R4), V4
	VMOVQ	80(R4), V5
	VMOVQ	96(R4), V6
	VMOVQ	112(R4), V7
	VMOVQ	0(R5), V8
	VMOVQ	16(R5), V9
	VMOVQ	32(R5), V10
	VMOVQ	48(R5), V11
	VMOVQ	64(R5), V12
	VMOVQ	80(R5), V13
	VMOVQ	96(R5), V14
	VMOVQ	112(R5), V15
	VSEQV	V0, V8, V0
	VSEQV	V1, V9, V1
	VSEQV	V2, V10, V2
	VSEQV	V3, V11, V3
	VSEQV	V4, V12, V4
	VSEQV	V5, V13, V5
	VSEQV	V6, V14, V6
	VSEQV	V7, V15, V7
	VANDV	V0, V1, V0
	VANDV	V2, V3, V2
	VANDV	V4, V5, V4
	VANDV	V6, V7, V6
	VANDV	V0, V2, V0
	VANDV	V4, V6, V4
	VANDV	V0, V4, V0
	VSETALLNEV	V0, FCC0
	BFPF	not_equal
	BEQ	R6, equal

	ADDV	$128, R4
	ADDV	$128, R5
	BGE	R6, R7, lsx128_loop

lsx32:
	MOVV	$32, R7
	BLT	R6, R7, tail
lsx32_loop:
	ADDV	$-32, R6
	VMOVQ	0(R4), V0
	VMOVQ	16(R4), V1
	VMOVQ	0(R5), V2
	VMOVQ	16(R5), V3
	VSEQV	V0, V2, V0
	VSEQV	V1, V3, V1
	VANDV	V0, V1, V0
	VSETALLNEV	V0, FCC0
	BFPF	not_equal
	BEQ	R6, equal
	ADDV	$32, R4
	ADDV	$32, R5
	BGE	R6, R7, lsx32_loop
	JMP tail

	// Implemented using general instructions
generic64_loop:
	ADDV	$-64, R6
	MOVV	0(R4), R7
	MOVV	8(R4), R8
	MOVV	16(R4), R9
	MOVV	24(R4), R10
	MOVV	0(R5), R15
	MOVV	8(R5), R16
	MOVV	16(R5), R17
	MOVV	24(R5), R18
	BNE	R7, R15, not_equal
	BNE	R8, R16, not_equal
	BNE	R9, R17, not_equal
	BNE	R10, R18, not_equal
	MOVV	32(R4), R11
	MOVV	40(R4), R12
	MOVV	48(R4), R13
	MOVV	56(R4), R14
	MOVV	32(R5), R19
	MOVV	40(R5), R20
	MOVV	48(R5), R21
	MOVV	56(R5), R23
	BNE	R11, R19, not_equal
	BNE	R12, R20, not_equal
	BNE	R13, R21, not_equal
	BNE	R14, R23, not_equal
	BEQ	R6, equal
	ADDV	$64, R4
	ADDV	$64, R5
	MOVV	$64, R7
	BGE	R6, R7, generic64_loop
	JMP tail
