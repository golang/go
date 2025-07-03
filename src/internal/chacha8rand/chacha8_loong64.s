// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

DATA	·chachaConst+0x00(SB)/4, $0x61707865
DATA	·chachaConst+0x04(SB)/4, $0x3320646e
DATA	·chachaConst+0x08(SB)/4, $0x79622d32
DATA	·chachaConst+0x0c(SB)/4, $0x6b206574
GLOBL	·chachaConst(SB), NOPTR|RODATA, $32

DATA	·chachaIncRot+0x00(SB)/4, $0x00000000
DATA	·chachaIncRot+0x04(SB)/4, $0x00000001
DATA	·chachaIncRot+0x08(SB)/4, $0x00000002
DATA	·chachaIncRot+0x0c(SB)/4, $0x00000003
GLOBL	·chachaIncRot(SB), NOPTR|RODATA, $32

// QR is the ChaCha8 quarter-round on a, b, c, and d.
#define QR(a, b, c, d) \
	VADDW	a, b, a; \
	VXORV	d, a, d; \
	VROTRW	$16, d; \
	VADDW	c, d, c; \
	VXORV	b, c, b; \
	VROTRW	$20, b; \
	VADDW	a, b, a; \
	VXORV	d, a, d; \
	VROTRW	$24, d; \
	VADDW	c, d, c; \
	VXORV	b, c, b; \
	VROTRW	$25, b


// func block(seed *[8]uint32, blocks *[4][16]uint32, counter uint32)
TEXT ·block<ABIInternal>(SB), NOSPLIT, $0
	// seed in R4
	// blocks in R5
	// counter in R6

	MOVBU	internal∕cpu·Loong64+const_offsetLOONG64HasLSX(SB), R7
	BNE	R7, lsx_chacha8
	JMP	·block_generic<ABIInternal>(SB)
	RET

lsx_chacha8:
	MOVV	$·chachaConst(SB), R10
	MOVV	$·chachaIncRot(SB), R11

	// load contants
	// VLDREPL.W  $0, R10, V0
	WORD	$0x30200140
	// VLDREPL.W  $1, R10, V1
	WORD	$0x30200541
	// VLDREPL.W  $2, R10, V2
	WORD	$0x30200942
	// VLDREPL.W  $3, R10, V3
	WORD	$0x30200d43

	// load 4-32bit data from incRotMatrix added to counter
	VMOVQ	(R11), V30

	// load seed
	// VLDREPL.W  $0, R4, V4
	WORD	$0x30200084
	// VLDREPL.W  $1, R4, V5
	WORD	$0x30200485
	// VLDREPL.W  $2, R4, V6
	WORD	$0x30200886
	// VLDREPL.W  $3, R4, V7
	WORD	$0x30200c87
	// VLDREPL.W  $4, R4, V8
	WORD	$0x30201088
	// VLDREPL.W  $5, R4, V9
	WORD	$0x30201489
	// VLDREPL.W  $6, R4, V10
	WORD	$0x3020188a
	// VLDREPL.W  $7, R4, V11
	WORD	$0x30201c8b

	// load counter and update counter
	VMOVQ	R6, V12.W4
	VADDW	V12, V30, V12

	// zeros for remaining three matrix entries
	VXORV	V13, V13, V13
	VXORV	V14, V14, V14
	VXORV	V15, V15, V15

	// save seed state for adding back later
	VORV	V4, V13, V20
	VORV	V5, V13, V21
	VORV	V6, V13, V22
	VORV	V7, V13, V23
	VORV	V8, V13, V24
	VORV	V9, V13, V25
	VORV	V10, V13, V26
	VORV	V11, V13, V27

	// 4 iterations. Each iteration is 8 quarter-rounds.
	MOVV	$4, R7
loop:
	QR(V0, V4, V8, V12)
	QR(V1, V5, V9, V13)
	QR(V2, V6, V10, V14)
	QR(V3, V7, V11, V15)

	QR(V0, V5, V10, V15)
	QR(V1, V6, V11, V12)
	QR(V2, V7, V8, V13)
	QR(V3, V4, V9, V14)

	SUBV	$1, R7
	BNE	R7, R0, loop

	// add seed back
	VADDW	V4, V20, V4
	VADDW	V5, V21, V5
	VADDW	V6, V22, V6
	VADDW	V7, V23, V7
	VADDW	V8, V24, V8
	VADDW	V9, V25, V9
	VADDW	V10, V26, V10
	VADDW	V11, V27, V11

	// store blocks back to output buffer
	VMOVQ	V0, (R5)
	VMOVQ	V1, 16(R5)
	VMOVQ	V2, 32(R5)
	VMOVQ	V3, 48(R5)
	VMOVQ	V4, 64(R5)
	VMOVQ	V5, 80(R5)
	VMOVQ	V6, 96(R5)
	VMOVQ	V7, 112(R5)
	VMOVQ	V8, 128(R5)
	VMOVQ	V9, 144(R5)
	VMOVQ	V10, 160(R5)
	VMOVQ	V11, 176(R5)
	VMOVQ	V12, 192(R5)
	VMOVQ	V13, 208(R5)
	VMOVQ	V14, 224(R5)
	VMOVQ	V15, 240(R5)

	RET
