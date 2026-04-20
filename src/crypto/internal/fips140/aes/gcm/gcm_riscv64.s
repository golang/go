// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

#include "textflag.h"

// LOAD_AES_ROUND_KEYS loads all AES encryption round keys from the key
// schedule at KS and Clobbers X24.
// KS_LEN is the length of the key schedule in uint32.
// KS_LEN vs Number of Round
// - 44/10
// - 52/12
// - 60/14
#define LOAD_AES_ROUND_KEYS(KS, KS_LEN) \
	VLE32V	(KS), V10; \
	ADD	$16, KS; \
	VLE32V	(KS), V11; \
	ADD	$16, KS; \
	VLE32V	(KS), V12; \
	ADD	$16, KS; \
	VLE32V	(KS), V13; \
	ADD	$16, KS; \
	VLE32V	(KS), V14; \
	ADD	$16, KS; \
	VLE32V	(KS), V15; \
	ADD	$16, KS; \
	VLE32V	(KS), V16; \
	ADD	$16, KS; \
	VLE32V	(KS), V17; \
	ADD	$16, KS; \
	VLE32V	(KS), V18; \
	ADD	$16, KS; \
	VLE32V	(KS), V19; \
	ADD	$16, KS; \
	SUB	$52, KS_LEN, X24; \
	BLTZ	X24, nr10; \
	BEQZ	X24, nr12; \
	VLE32V	(KS), V24; \
	ADD	$16, KS; \
	VLE32V	(KS), V23; \
	ADD	$16, KS; \
nr12: \
	VLE32V	(KS), V22; \
	ADD	$16, KS; \
	VLE32V	(KS), V21; \
	ADD	$16, KS; \
nr10: \
	VLE32V	(KS), V20

// ENCRYPT(BLOCK, KS_LEN) - AES encrypt block at BLOCK.
// KS_LEN is the length of the key schedule in uint32.
// Clobbers X24 and returns the result in V30.
#define ENCRYPT(BLOCK, KS_LEN) \
	VLE32V	(BLOCK), V30; \
	VAESZVS	V10, V30; \
	VAESEMVS	V11, V30; \
	VAESEMVS	V12, V30; \
	VAESEMVS	V13, V30; \
	VAESEMVS	V14, V30; \
	VAESEMVS	V15, V30; \
	VAESEMVS	V16, V30; \
	VAESEMVS	V17, V30; \
	VAESEMVS	V18, V30; \
	VAESEMVS	V19, V30; \
	SUB	$52, KS_LEN, X24; \
	BLTZ	X24, 6(PC); \
	BEQZ	X24, 3(PC); \
	VAESEMVS	V24, V30; \
	VAESEMVS	V23, V30; \
	VAESEMVS	V22, V30; \
	VAESEMVS	V21, V30; \
	VAESEFVS	V20, V30

// INCCTR increments the big-endian counter in the last 4 bytes at CTR.
#define INCCTR(CTR) \
	MOV	8(CTR), X28; \
	REV8	X28, X28; \
	ADD	$1, X28; \
	SLL	$32, X28; \
	REV8	X28, X28; \
	MOVW	X28, 12(CTR)

// func gcmAesInit(productTable *[256]byte, ks []uint32)
TEXT ·gcmAesInit(SB),NOSPLIT,$16
	MOV	productTable+0(FP), X10
	MOV	ks_base+8(FP), X11
	MOV	ks_len+16(FP), X12

	VSETIVLI	$4, E32, M1, TA, MA, ZERO
	ADD	$8, X2, X6
	VXORVV	V2, V2, V2
	VSE32V	V2, (X6)
	LOAD_AES_ROUND_KEYS(X11, X12)
	ENCRYPT(X6, X12)
	VSE32V	V30, (X10)
	RET

// func gcmAesData(productTable *[256]byte, data []byte, T *[16]byte)
TEXT ·gcmAesData(SB),NOSPLIT,$16
	MOV	productTable+0(FP), X10
	MOV	data_base+8(FP), X11
	MOV	data_len+16(FP), X12
	BEQZ	X12, empty
	MOV	T+32(FP), X13

	VSETIVLI	$4, E32, M1, TA, MA, ZERO
	VLE32V	(X10), V2
	VLE32V	(X13), V1
	SRL	$4, X12, X14
	BEQZ	X14, partial
loop:
	VLE32V	(X11), V3
	VGHSHVV	V3, V2, V1
	ADD	$16, X11
	SUB	$1, X14
	BNEZ	X14, loop
	AND	$15, X12
	BEQZ	X12, done
partial:
	ADD	$8, X2, X6
	VXORVV	V3, V3, V3
	VSE32V	V3, (X6)
	MOV	X6, X5
	MOV	X11, X14
	ADD	X12, X14
copy:
	MOVBU	(X11), X28
	MOVB	X28, (X5)
	ADD	$1, X11
	ADD	$1, X5
	BNE	X11, X14, copy

	VLE32V	(X6), V3
	VGHSHVV	V3, V2, V1
done:
	VSE32V	V1, (X13)
empty:
	RET

// func gcmAesEnc(productTable *[256]byte, dst, src []byte, ctr, T *[16]byte, ks []uint32)
TEXT ·gcmAesEnc(SB),NOSPLIT,$40
	MOV	productTable+0(FP), X10
	MOV	dst_base+8(FP), X11
	MOV	src_base+32(FP), X12
	MOV	src_len+40(FP), X13
	MOV	ctr+56(FP), X14
	MOV	T+64(FP), X15
	MOV	ks_base+72(FP), X16
	MOV	ks_len+80(FP), X17

	INCCTR(X14)
	VSETIVLI	$4, E32, M1, TA, MA, ZERO
	VLE32V	(X10), V2
	VLE32V	(X15), V1
	LOAD_AES_ROUND_KEYS(X16, X17)
	AND	$15, X13, X25
	SRL	$4, X13
	BEQZ	X13, tail
loop:
	ENCRYPT(X14, X17)
	VLE32V	(X12), V3
	VXORVV	V30, V3, V4
	VSE32V	V4, (X11)
	VGHSHVV	V4, V2, V1
	ADD	$16, X11
	ADD	$16, X12
	INCCTR(X14)
	SUB	$1, X13
	BNEZ	X13, loop

tail:
	BEQZ	X25, done
	ENCRYPT(X14, X17)
	ADD	$8, X2, X5
	VSE32V	V30, (X5)
	ADD	$24, X2, X7
	VXORVV	V3, V3, V3
	VSE32V	V3, (X7)
	MOV	X12, X13
	ADD	X25, X13
	MOV	X7, X26
tailLoop:
	MOVBU	(X12), X28
	MOVBU	(X5), X29
	XOR	X29, X28
	MOVB	X28, (X11)
	MOVB	X28, (X26)
	ADD	$1, X12
	ADD	$1, X5
	ADD	$1, X11
	ADD	$1, X26
	BNE	X12, X13, tailLoop
	VLE32V	(X7), V3
	VGHSHVV	V3, V2, V1
done:
	VSE32V	V1, (X15)
	RET

// func gcmAesDec(productTable *[256]byte, dst, src []byte, ctr, T *[16]byte, ks []uint32)
TEXT ·gcmAesDec(SB),NOSPLIT,$32
	MOV	productTable+0(FP), X10
	MOV	dst_base+8(FP), X11
	MOV	src_base+32(FP), X12
	MOV	src_len+40(FP), X13
	MOV	ctr+56(FP), X14
	MOV	T+64(FP), X15
	MOV	ks_base+72(FP), X16
	MOV	ks_len+80(FP), X17

	INCCTR(X14)
	VSETIVLI	$4, E32, M1, TA, MA, ZERO
	VLE32V	(X10), V2
	VLE32V	(X15), V1
	LOAD_AES_ROUND_KEYS(X16, X17)
	AND	$15, X13, X25
	SRL	$4, X13
	BEQZ	X13, tail
loop:
	ENCRYPT(X14, X17)
	VLE32V	(X12), V3
	VXORVV	V30, V3, V4
	VSE32V	V4, (X11)
	VGHSHVV	V3, V2, V1
	ADD	$16, X11
	ADD	$16, X12
	INCCTR(X14)
	SUB	$1, X13
	BNEZ	X13, loop

tail:
	BEQZ	X25, done
	ENCRYPT(X14, X17)
	ADD	$8, X2, X5
	VSE32V	V30, (X5)
	ADD	$24, X2, X7
	VXORVV	V3, V3, V3
	VSE32V	V3, (X7)
	MOV	X12, X13
	ADD	X25, X13
	MOV	X7, X26
tailLoop:
	MOVBU	(X12), X28
	MOVB	X28, (X26)
	MOVBU	(X5), X29
	XOR	X29, X28
	MOVB	X28, (X11)
	ADD	$1, X12
	ADD	$1, X5
	ADD	$1, X11
	ADD	$1, X26
	BNE	X12, X13, tailLoop
	VLE32V	(X7), V3
	VGHSHVV	V3, V2, V1
done:
	VSE32V	V1, (X15)
	RET

// func gcmAesFinish(productTable *[256]byte, tagMask, T *[16]byte, pLen, dLen uint64)
TEXT ·gcmAesFinish(SB),NOSPLIT,$16
	MOV	productTable+0(FP), X10
	MOV	tagMask+8(FP), X11
	MOV	T+16(FP), X12
	MOV	pLen+24(FP), X13
	MOV	dLen+32(FP), X14

	VSETIVLI	$4, E32, M1, TA, MA, ZERO
	VLE32V	(X10), V2
	VLE32V	(X12), V1
	SLL	$3, X14
	SLL	$3, X13
	REV8	X14, X14
	REV8	X13, X13
	ADD	$8, X2, X6
	MOV	X14, 0(X6)
	MOV	X13, 8(X6)
	VLE32V	(X6), V3
	VGHSHVV	V3, V2, V1
	VLE32V	(X11), V3
	VXORVV	V1, V3, V1
	VSE32V	V1, (X12)
	RET
