// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "asm_riscv64.h"
#include "go_asm.h"
#include "textflag.h"

// TODO(mzh): use Zvkb if possible

#define QR(A, B, C, D) \
	VADDVV	A, B, A \
	VXORVV	D, A, D \
	VSLLVI	$16, D, V28 \
	VSRLVI	$16, D, D \
	VXORVV	V28, D, D \
	VADDVV	D, C, C  \
	VXORVV	C, B, B \
	VSLLVI	$12, B, V29 \
	VSRLVI	$20, B, B \
	VXORVV	V29, B, B \
	VADDVV	B, A, A  \
	VXORVV	A, D, D \
	VSLLVI	$8, D, V30 \
	VSRLVI	$24, D, D \
	VXORVV	V30, D, D \
	VADDVV	D, C, C  \
	VXORVV	C, B, B \
	VSLLVI	$7, B, V31 \
	VSRLVI	$25, B, B \
	VXORVV	V31, B, B

// block runs four ChaCha8 block transformations using four elements in each V register.
// func block(seed *[8]uint32, blocks *[16][4]uint32, counter uint32)
TEXT ·block<ABIInternal>(SB), NOSPLIT, $0
	// seed in X10
	// blocks in X11
	// counter in X12

#ifndef hasV
	MOVB	internal∕cpu·RISCV64+const_offsetRISCV64HasV(SB), X13
	BNEZ	X13, vector_chacha8
	JMP	·block_generic<ABIInternal>(SB)
#endif

vector_chacha8:
	// At least VLEN >= 128
	VSETIVLI	$4, E32, M1, TA, MA, X0
	// Load initial constants into top row.
	MOV $·chachaConst(SB), X14
	VLSSEG4E32V	(X14), X0, V0 // V0, V1, V2, V3 = const row
	VLSSEG8E32V	(X10), X0, V4 // V4 ... V11, seed
	VIDV	V12
	VADDVX	X12, V12, V12		// counter

	// Clear all nonces.
	VXORVV	V13, V13, V13
	VXORVV	V14, V14, V14
	VXORVV	V15, V15, V15

	// Copy initial state.
	VMV4RV V4, V20
	VMV4RV V8, V24

	MOV	$4, X15
	PCALIGN	$16
loop:
	QR(V0, V4, V8, V12)
	QR(V1, V5, V9, V13)
	QR(V2, V6, V10, V14)
	QR(V3, V7, V11, V15)

	QR(V0, V5, V10, V15)
	QR(V1, V6, V11, V12)
	QR(V2, V7, V8, V13)
	QR(V3, V4, V9, V14)

	SUB	$1, X15
	BNEZ	X15, loop

	VADDVV	V20, V4, V4
	VADDVV	V21, V5, V5
	VADDVV	V22, V6, V6
	VADDVV	V23, V7, V7
	VADDVV	V24, V8, V8
	VADDVV	V25, V9, V9
	VADDVV	V26, V10, V10
	VADDVV	V27, V11, V11

	VSE32V	V0, (X11); ADD $16, X11;
	VSE32V	V1, (X11); ADD $16, X11;
	VSE32V	V2, (X11); ADD $16, X11;
	VSE32V	V3, (X11); ADD $16, X11;
	VSE32V	V4, (X11); ADD $16, X11;
	VSE32V	V5, (X11); ADD $16, X11;
	VSE32V	V6, (X11); ADD $16, X11;
	VSE32V	V7, (X11); ADD $16, X11;
	VSE32V	V8, (X11); ADD $16, X11;
	VSE32V	V9, (X11); ADD $16, X11;
	VSE32V	V10, (X11); ADD $16, X11;
	VSE32V	V11, (X11); ADD $16, X11;
	VSE32V	V12, (X11); ADD $16, X11;
	VSE32V	V13, (X11); ADD $16, X11;
	VSE32V	V14, (X11); ADD $16, X11;
	VSE32V	V15, (X11); ADD $16, X11;

	RET

GLOBL	·chachaConst(SB), NOPTR|RODATA, $32
DATA	·chachaConst+0x00(SB)/4, $0x61707865
DATA	·chachaConst+0x04(SB)/4, $0x3320646e
DATA	·chachaConst+0x08(SB)/4, $0x79622d32
DATA	·chachaConst+0x0c(SB)/4, $0x6b206574
