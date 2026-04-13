// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

#include "textflag.h"

// SAVE_CTR writes (ivhi||ivlo) as a 16-byte big-endian block at (X6).
#define SAVE_CTR \
	REV8	X16, X25; \
	REV8	X15, X24; \
	MOV	X25, (X6); \
	MOV	X24, 8(X6)

// SAVE_INC_CTR advances the counter by one
#define SAVE_INC_CTR \
	ADD	$1, X15; \
	SEQZ	X15, X24; \
	ADD	X24, X16; \
	SAVE_CTR

// LOAD_AES_ROUND_KEYS loads round keys once per ctrN path.
// Common prefix: V10..V19 hold RK0..RK9.
// Tail layout keeps the final round key fixed in V20:
// AES-128: V20=RK10
// AES-192: V22=RK10, V21=RK11, V20=RK12
// AES-256: V24=RK10, V23=RK11, V22=RK12, V21=RK13, V20=RK14
#define LOAD_AES_ROUND_KEYS \
	VLE32V	(X11), V10; \
	ADD	$16, X11; \
	VLE32V	(X11), V11; \
	ADD	$16, X11; \
	VLE32V	(X11), V12; \
	ADD	$16, X11; \
	VLE32V	(X11), V13; \
	ADD	$16, X11; \
	VLE32V	(X11), V14; \
	ADD	$16, X11; \
	VLE32V	(X11), V15; \
	ADD	$16, X11; \
	VLE32V	(X11), V16; \
	ADD	$16, X11; \
	VLE32V	(X11), V17; \
	ADD	$16, X11; \
	VLE32V	(X11), V18; \
	ADD	$16, X11; \
	VLE32V	(X11), V19; \
	ADD	$16, X11; \
	SUB	$12, X10, X20; \
	BLTZ	X20, nr10; \
	BEQZ	X20, nr12; \
	VLE32V	(X11), V24; \
	ADD	$16, X11; \
	VLE32V	(X11), V23; \
	ADD	$16, X11; \
nr12: ; \
	VLE32V	(X11), V22; \
	ADD	$16, X11; \
	VLE32V	(X11), V21; \
	ADD	$16, X11; \
nr10: ; \
	VLE32V	(X11), V20

// ENC_BLOCK encrypts counter block at (X6),
// XORs with VSRC, and stores to dst at (X12).
// Round keys are preloaded into V10..V24 by LOAD_AES_ROUND_KEYS.
#define ENC_BLOCK(VSRC) \
	VLE32V	(X6), V30; \
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
	SUB	$12, X10, X24; \
	BLTZ	X24, 6(PC); \
	BEQZ	X24, 3(PC); \
	VAESEMVS	V24, V30; \
	VAESEMVS	V23, V30; \
	VAESEMVS	V22, V30; \
	VAESEMVS	V21, V30; \
	VAESEFVS	V20, V30; \
	VXORVV	VSRC, V30, V30; \
	VSE32V	V30, (X12); \
	ADD	$16, X12

// func ctrBlocks1Asm(nr int, xk *[60]uint32, dst *[16]byte, src *[16]byte, ivlo uint64, ivhi uint64)
TEXT ·ctrBlocks1Asm(SB), NOSPLIT, $16
	MOV	nr+0(FP), X10
	MOV	xk+8(FP), X11
	MOV	dst+16(FP), X12
	MOV	src+24(FP), X13
	MOV	ivlo+32(FP), X15
	MOV	ivhi+40(FP), X16

	VSETIVLI	$4, E32, M1, TA, MA, X0

	// ctrBuf: [8, 24). 0..7 holds the saved LR.
	ADD	$8, X2, X6
	VLE32V	(X13), V2
	LOAD_AES_ROUND_KEYS
	SAVE_CTR
	ENC_BLOCK(V2)
	RET

// func ctrBlocks2Asm(nr int, xk *[60]uint32, dst *[2*16]byte, src *[2*16]byte, ivlo uint64, ivhi uint64)
TEXT ·ctrBlocks2Asm(SB), NOSPLIT, $16
	MOV	nr+0(FP), X10
	MOV	xk+8(FP), X11
	MOV	dst+16(FP), X12
	MOV	src+24(FP), X13
	MOV	ivlo+32(FP), X15
	MOV	ivhi+40(FP), X16

	VSETIVLI	$4, E32, M1, TA, MA, X0

	ADD	$8, X2, X6
	VLE32V	(X13), V2
	ADD	$16, X13
	VLE32V	(X13), V3
	LOAD_AES_ROUND_KEYS

	SAVE_CTR
	ENC_BLOCK(V2)
	SAVE_INC_CTR
	ENC_BLOCK(V3)
	RET

// func ctrBlocks4Asm(nr int, xk *[60]uint32, dst *[4*16]byte, src *[4*16]byte, ivlo uint64, ivhi uint64)
TEXT ·ctrBlocks4Asm(SB), NOSPLIT, $16
	MOV	nr+0(FP), X10
	MOV	xk+8(FP), X11
	MOV	dst+16(FP), X12
	MOV	src+24(FP), X13
	MOV	ivlo+32(FP), X15
	MOV	ivhi+40(FP), X16

	VSETIVLI	$4, E32, M1, TA, MA, X0

	ADD	$8, X2, X6
	VLE32V	(X13), V2
	ADD	$16, X13
	VLE32V	(X13), V3
	ADD	$16, X13
	VLE32V	(X13), V4
	ADD	$16, X13
	VLE32V	(X13), V5
	LOAD_AES_ROUND_KEYS

	SAVE_CTR
	ENC_BLOCK(V2)

	SAVE_INC_CTR
	ENC_BLOCK(V3)
	SAVE_INC_CTR
	ENC_BLOCK(V4)
	SAVE_INC_CTR
	ENC_BLOCK(V5)
	RET

// func ctrBlocks8Asm(nr int, xk *[60]uint32, dst *[8*16]byte, src *[8*16]byte, ivlo uint64, ivhi uint64)
TEXT ·ctrBlocks8Asm(SB), NOSPLIT, $16
	MOV	nr+0(FP), X10
	MOV	xk+8(FP), X11
	MOV	dst+16(FP), X12
	MOV	src+24(FP), X13
	MOV	ivlo+32(FP), X15
	MOV	ivhi+40(FP), X16

	VSETIVLI	$4, E32, M1, TA, MA, X0

	ADD	$8, X2, X6
	VLE32V	(X13), V2
	ADD	$16, X13
	VLE32V	(X13), V3
	ADD	$16, X13
	VLE32V	(X13), V4
	ADD	$16, X13
	VLE32V	(X13), V5
	ADD	$16, X13
	VLE32V	(X13), V6
	ADD	$16, X13
	VLE32V	(X13), V7
	ADD	$16, X13
	VLE32V	(X13), V8
	ADD	$16, X13
	VLE32V	(X13), V9
	LOAD_AES_ROUND_KEYS

	SAVE_CTR
	ENC_BLOCK(V2)

	SAVE_INC_CTR
	ENC_BLOCK(V3)
	SAVE_INC_CTR
	ENC_BLOCK(V4)
	SAVE_INC_CTR
	ENC_BLOCK(V5)
	SAVE_INC_CTR
	ENC_BLOCK(V6)
	SAVE_INC_CTR
	ENC_BLOCK(V7)
	SAVE_INC_CTR
	ENC_BLOCK(V8)
	SAVE_INC_CTR
	ENC_BLOCK(V9)
	RET
