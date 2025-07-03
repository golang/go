// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

#include "textflag.h"

#define SMALL_TAIL \
	SGTU	$2, R7, R8; \
	BNE	R8, xor_1; \
	SGTU	$4, R7, R8; \
	BNE	R8, xor_2; \
	SGTU	$8, R7, R8; \
	BNE	R8, xor_4; \
	SGTU	$16, R7, R8; \
	BNE	R8, xor_8; \

#define SMALL \
xor_8_check:; \
	SGTU	$8, R7, R8; \
	BNE	R8, xor_4_check; \
xor_8:; \
	SUBV	$8, R7; \
	MOVV	(R5), R10; \
	MOVV	(R6), R11; \
	XOR	R10, R11; \
	MOVV	R11, (R4); \
	ADDV	$8, R5; \
	ADDV	$8, R6; \
	ADDV	$8, R4; \
	BEQ	R7, R0, end; \
xor_4_check:; \
	SGTU	$4, R7, R8; \
	BNE	R8, xor_2_check; \
xor_4:; \
	SUBV	$4, R7; \
	MOVW	(R5), R10; \
	MOVW	(R6), R11; \
	XOR	R10, R11; \
	MOVW	R11, (R4); \
	ADDV	$4, R5; \
	ADDV	$4, R6; \
	ADDV	$4, R4; \
	BEQ	R7, R0, end; \
xor_2_check:; \
	SGTU	$2, R7, R8; \
	BNE	R8, xor_1; \
xor_2:; \
	SUBV	$2, R7; \
	MOVH	(R5), R10; \
	MOVH	(R6), R11; \
	XOR	R10, R11; \
	MOVH	R11, (R4); \
	ADDV	$2, R5; \
	ADDV	$2, R6; \
	ADDV	$2, R4; \
	BEQ	R7, R0, end; \
xor_1:; \
	MOVB	(R5), R10; \
	MOVB	(R6), R11; \
	XOR	R10, R11; \
	MOVB	R11, (R4); \

// func xorBytesBasic(dst, a, b *byte, n int)
TEXT ·xorBytesBasic(SB), NOSPLIT, $0
	MOVV	dst+0(FP), R4
	MOVV	a+8(FP), R5
	MOVV	b+16(FP), R6
	MOVV	n+24(FP), R7

	SMALL_TAIL

xor_64_check:
	SGTU	$64, R7, R8
	BNE	R8, xor_32_check
xor_64_loop:
	SUBV	$64, R7
	MOVV	(R5), R10
	MOVV	8(R5), R11
	MOVV	16(R5), R12
	MOVV	24(R5), R13
	MOVV	(R6), R14
	MOVV	8(R6), R15
	MOVV	16(R6), R16
	MOVV	24(R6), R17
	XOR	R10, R14
	XOR	R11, R15
	XOR	R12, R16
	XOR	R13, R17
	MOVV	R14, (R4)
	MOVV	R15, 8(R4)
	MOVV	R16, 16(R4)
	MOVV	R17, 24(R4)
	MOVV	32(R5), R10
	MOVV	40(R5), R11
	MOVV	48(R5), R12
	MOVV	56(R5), R13
	MOVV	32(R6), R14
	MOVV	40(R6), R15
	MOVV	48(R6), R16
	MOVV	56(R6), R17
	XOR	R10, R14
	XOR	R11, R15
	XOR	R12, R16
	XOR	R13, R17
	MOVV	R14, 32(R4)
	MOVV	R15, 40(R4)
	MOVV	R16, 48(R4)
	MOVV	R17, 56(R4)
	SGTU	$64, R7, R8
	ADDV	$64, R5
	ADDV	$64, R6
	ADDV	$64, R4
	BEQ	R8, xor_64_loop
	BEQ	R7, end

xor_32_check:
	SGTU	$32, R7, R8
	BNE	R8, xor_16_check
xor_32:
	SUBV	$32, R7
	MOVV	(R5), R10
	MOVV	8(R5), R11
	MOVV	16(R5), R12
	MOVV	24(R5), R13
	MOVV	(R6), R14
	MOVV	8(R6), R15
	MOVV	16(R6), R16
	MOVV	24(R6), R17
	XOR	R10, R14
	XOR	R11, R15
	XOR	R12, R16
	XOR	R13, R17
	MOVV	R14, (R4)
	MOVV	R15, 8(R4)
	MOVV	R16, 16(R4)
	MOVV	R17, 24(R4)
	ADDV	$32, R5
	ADDV	$32, R6
	ADDV	$32, R4
	BEQ	R7, R0, end

xor_16_check:
	SGTU	$16, R7, R8
	BNE	R8, xor_8_check
xor_16:
	SUBV	$16, R7
	MOVV	(R5), R10
	MOVV	8(R5), R11
	MOVV	(R6), R12
	MOVV	8(R6), R13
	XOR	R10, R12
	XOR	R11, R13
	MOVV	R12, (R4)
	MOVV	R13, 8(R4)
	ADDV	$16, R5
	ADDV	$16, R6
	ADDV	$16, R4
	BEQ	R7, R0, end

	SMALL
end:
	RET

// func xorBytesLSX(dst, a, b *byte, n int)
TEXT ·xorBytesLSX(SB), NOSPLIT, $0
	MOVV	dst+0(FP), R4
	MOVV	a+8(FP), R5
	MOVV	b+16(FP), R6
	MOVV	n+24(FP), R7

	SMALL_TAIL

xor_128_lsx_check:
	SGTU	$128, R7, R8
	BNE	R8, xor_64_lsx_check
xor_128_lsx_loop:
	SUBV	$128, R7
	VMOVQ	(R5), V0
	VMOVQ	16(R5), V1
	VMOVQ	32(R5), V2
	VMOVQ	48(R5), V3
	VMOVQ	64(R5), V4
	VMOVQ	80(R5), V5
	VMOVQ	96(R5), V6
	VMOVQ	112(R5), V7
	VMOVQ	(R6), V8
	VMOVQ	16(R6), V9
	VMOVQ	32(R6), V10
	VMOVQ	48(R6), V11
	VMOVQ	64(R6), V12
	VMOVQ	80(R6), V13
	VMOVQ	96(R6), V14
	VMOVQ	112(R6), V15
	VXORV	V0, V8, V8
	VXORV	V1, V9, V9
	VXORV	V2, V10, V10
	VXORV	V3, V11, V11
	VXORV	V4, V12, V12
	VXORV	V5, V13, V13
	VXORV	V6, V14, V14
	VXORV	V7, V15, V15
	VMOVQ	V8, (R4)
	VMOVQ	V9, 16(R4)
	VMOVQ	V10, 32(R4)
	VMOVQ	V11, 48(R4)
	VMOVQ	V12, 64(R4)
	VMOVQ	V13, 80(R4)
	VMOVQ	V14, 96(R4)
	VMOVQ	V15, 112(R4)
	SGTU	$128, R7, R8
	ADDV	$128, R5
	ADDV	$128, R6
	ADDV	$128, R4
	BEQ	R8, xor_128_lsx_loop
	BEQ	R7, end

xor_64_lsx_check:
	SGTU	$64, R7, R8
	BNE	R8, xor_32_lsx_check
xor_64_lsx:
	SUBV	$64, R7
	VMOVQ	(R5), V0
	VMOVQ	16(R5), V1
	VMOVQ	32(R5), V2
	VMOVQ	48(R5), V3
	VMOVQ	(R6), V4
	VMOVQ	16(R6), V5
	VMOVQ	32(R6), V6
	VMOVQ	48(R6), V7
	VXORV	V0, V4, V4
	VXORV	V1, V5, V5
	VXORV	V2, V6, V6
	VXORV	V3, V7, V7
	VMOVQ	V4, (R4)
	VMOVQ	V5, 16(R4)
	VMOVQ	V6, 32(R4)
	VMOVQ	V7, 48(R4)
	ADDV	$64, R5
	ADDV	$64, R6
	ADDV	$64, R4
	BEQ	R7, end

xor_32_lsx_check:
	SGTU	$32, R7, R8
	BNE	R8, xor_16_lsx_check
xor_32_lsx:
	SUBV	$32, R7
	VMOVQ	(R5), V0
	VMOVQ	16(R5), V1
	VMOVQ	(R6), V2
	VMOVQ	16(R6), V3
	VXORV	V0, V2, V2
	VXORV	V1, V3, V3
	VMOVQ	V2, (R4)
	VMOVQ	V3, 16(R4)
	ADDV	$32, R5
	ADDV	$32, R6
	ADDV	$32, R4
	BEQ	R7, end

xor_16_lsx_check:
	SGTU	$16, R7, R8
	BNE	R8, xor_8_check
xor_16_lsx:
	SUBV	$16, R7
	VMOVQ	(R5), V0
	VMOVQ	(R6), V1
	VXORV	V0, V1, V1
	VMOVQ	V1, (R4)
	ADDV	$16, R5
	ADDV	$16, R6
	ADDV	$16, R4
	BEQ	R7, end

	SMALL
end:
	RET

// func xorBytesLASX(dst, a, b *byte, n int)
TEXT ·xorBytesLASX(SB), NOSPLIT, $0
	MOVV	dst+0(FP), R4
	MOVV	a+8(FP), R5
	MOVV	b+16(FP), R6
	MOVV	n+24(FP), R7

	SMALL_TAIL

xor_256_lasx_check:
	SGTU	$256, R7, R8
	BNE	R8, xor_128_lasx_check
xor_256_lasx_loop:
	SUBV	$256, R7
	XVMOVQ	(R5), X0
	XVMOVQ	32(R5), X1
	XVMOVQ	64(R5), X2
	XVMOVQ	96(R5), X3
	XVMOVQ	128(R5), X4
	XVMOVQ	160(R5), X5
	XVMOVQ	192(R5), X6
	XVMOVQ	224(R5), X7
	XVMOVQ	(R6), X8
	XVMOVQ	32(R6), X9
	XVMOVQ	64(R6), X10
	XVMOVQ	96(R6), X11
	XVMOVQ	128(R6), X12
	XVMOVQ	160(R6), X13
	XVMOVQ	192(R6), X14
	XVMOVQ	224(R6), X15
	XVXORV	X0, X8, X8
	XVXORV	X1, X9, X9
	XVXORV	X2, X10, X10
	XVXORV	X3, X11, X11
	XVXORV	X4, X12, X12
	XVXORV	X5, X13, X13
	XVXORV	X6, X14, X14
	XVXORV	X7, X15, X15
	XVMOVQ	X8, (R4)
	XVMOVQ	X9, 32(R4)
	XVMOVQ	X10, 64(R4)
	XVMOVQ	X11, 96(R4)
	XVMOVQ	X12, 128(R4)
	XVMOVQ	X13, 160(R4)
	XVMOVQ	X14, 192(R4)
	XVMOVQ	X15, 224(R4)
	SGTU	$256, R7, R8
	ADDV	$256, R5
	ADDV	$256, R6
	ADDV	$256, R4
	BEQ	R8, xor_256_lasx_loop
	BEQ	R7, end

xor_128_lasx_check:
	SGTU	$128, R7, R8
	BNE	R8, xor_64_lasx_check
xor_128_lasx:
	SUBV	$128, R7
	XVMOVQ	(R5), X0
	XVMOVQ	32(R5), X1
	XVMOVQ	64(R5), X2
	XVMOVQ	96(R5), X3
	XVMOVQ	(R6), X4
	XVMOVQ	32(R6), X5
	XVMOVQ	64(R6), X6
	XVMOVQ	96(R6), X7
	XVXORV	X0, X4, X4
	XVXORV	X1, X5, X5
	XVXORV	X2, X6, X6
	XVXORV	X3, X7, X7
	XVMOVQ	X4, (R4)
	XVMOVQ	X5, 32(R4)
	XVMOVQ	X6, 64(R4)
	XVMOVQ	X7, 96(R4)
	ADDV	$128, R5
	ADDV	$128, R6
	ADDV	$128, R4
	BEQ	R7, end

xor_64_lasx_check:
	SGTU	$64, R7, R8
	BNE	R8, xor_32_lasx_check
xor_64_lasx:
	SUBV	$64, R7
	XVMOVQ	(R5), X0
	XVMOVQ	32(R5), X1
	XVMOVQ	(R6), X2
	XVMOVQ	32(R6), X3
	XVXORV	X0, X2, X2
	XVXORV	X1, X3, X3
	XVMOVQ	X2, (R4)
	XVMOVQ	X3, 32(R4)
	ADDV	$64, R5
	ADDV	$64, R6
	ADDV	$64, R4
	BEQ	R7, end

xor_32_lasx_check:
	SGTU	$32, R7, R8
	BNE	R8, xor_16_lasx_check
xor_32_lasx:
	SUBV	$32, R7
	XVMOVQ	(R5), X0
	XVMOVQ	(R6), X1
	XVXORV	X0, X1, X1
	XVMOVQ	X1, (R4)
	ADDV	$32, R5
	ADDV	$32, R6
	ADDV	$32, R4
	BEQ	R7, end

xor_16_lasx_check:
	SGTU	$16, R7, R8
	BNE	R8, xor_8_check
xor_16_lasx:
	SUBV	$16, R7
	VMOVQ	(R5), V0
	VMOVQ	(R6), V1
	VXORV	V0, V1, V1
	VMOVQ	V1, (R4)
	ADDV	$16, R5
	ADDV	$16, R6
	ADDV	$16, R4
	BEQ	R7, end

	SMALL
end:
	RET

