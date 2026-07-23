// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

TEXT ·Compare<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-56
	// R2 = a_base
	// R3 = a_len
	// R4 = a_cap (unused)
	// R5 = b_base (want in R4)
	// R6 = b_len (want in R5)
	// R7 = b_cap (unused)
	MOVD	R5, R4
	MOVD	R6, R5
	BR	cmpbody<>(SB)

TEXT runtime·cmpstring<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-40
	// R2 = a_base
	// R3 = a_len
	// R4 = b_base
	// R5 = b_len

	BR	cmpbody<>(SB)

// input:
//   R2 = a
//   R3 = alen
//   R4 = b
//   R5 = blen
//   For regabiargs output value( -1/0/1 ) stored in R2
TEXT cmpbody<>(SB),NOSPLIT|NOFRAME,$0-0
	CMPBEQ	R2, R4, cmplengths
	MOVD	R3, R7
	CMPBLE	R3, R5, minlen
	MOVD	R5, R7
minlen:
	CMPBEQ	R7, $0, cmplengths
	MOVD	$0, R6
	CMPBGT	R7, $16, cmp17to32

cmp1to16:
	ADD	$-1, R7
	VLL	R7, (R2), V16
	VLL	R7, (R4), V17
	VFENEB	V16, V17, V18
	VLGVB	$7, V18, R8
	CMPBLE	R8, R7, cmpReturnMismatch
cmplengths:
	CMP	R3, R5
	BEQ	cmpEqual
	BLT	cmpLess
cmpGreater:
	MOVD	$1, R2
	RET
cmpLess:
	MOVD	$-1, R2
	RET
cmpEqual:
	MOVD	$0, R2
	RET

cmpMismatch:
	VLGVB	$7, V18, R8
cmpReturnMismatch:
	ADD   R6, R8, R8
	MOVBZ  (R2)(R8*1), R1
	MOVBZ  (R4)(R8*1), R6
	CMPW  R1, R6
	MOVD  $1, R2
	MOVD  $-1,R1
	LOCGR  $4, R1, R2
	RET

cmp17to32:
	CMPBGT	R7,$32, cmp33to64
	VL	(R2), V16
	VL	(R4), V17
	VFENEBS	V16, V17, V18
	BRC	$6, cmpMismatch
	MOVD	R7, R8
	SUB	$16, R8
	ADD	R8, R6
	VL	(R6)(R2), V16
	VL	(R6)(R4), V17
	VFENEBS	V16, V17, V18
	BRC	$6, cmpMismatch
	BR	cmplengths
cmp33to64:
	CMPBGT	R7, $64, cmpCLC
	VL	(R2), V16
	VL	(R4), V17
	VL	16(R2), V19
	VL	16(R4), V20

	VFENEBS	V16, V17, V18
	BRC	$6, cmpMismatch

	ADD	$16, R6
	VFENEBS	V19, V20, V18
	BRC	$6, cmpMismatch

	MOVD	R7, R8              // R6 is already advanced by 16 bytes. Adjust it by (R7 - 48)
	SUB	$48, R8             // to get the final offset: R6 = 16 + (R7 - 48) = R7 - 32.
	ADD	R8, R6
	VL	(R6)(R2), V21
	VL	(R6)(R4), V22
	VL	16(R6)(R2), V23
	VL	16(R6)(R4), V24

	VFENEBS   V21, V22, V18
	BRC       $6, cmpMismatch

	ADD     $16, R6
	VFENEBS   V23, V24, V18
	BRC       $6, cmpMismatch
	BR	cmplengths
cmpCLC:
	CMP	R7, $256
	BLE	cmpCLCTail
	PFD	$1, (R2)
	PFD	$1, (R4)
	PFD	$1, 256(R2)
	PFD	$1, 256(R4)
cmpCLC_loop:
	PFD	$1, 512(R2)
	PFD	$1, 512(R4)
	CLC	$256, 0(R2), 0(R4)
	BGT	cmpGreater
	BLT	cmpLess
	SUB	$256, R7
	MOVD	$256(R2), R2
	MOVD	$256(R4), R4
	CMP	R7, $256
	BGT	cmpCLC_loop
cmpCLCTail:
	SUB	$1, R7
	EXRL	$cmpbodyclc<>(SB), R7
	BGT	cmpGreater
	BLT	cmpLess
	BR	cmplengths

TEXT cmpbodyclc<>(SB),NOSPLIT|NOFRAME,$0-0
	CLC	$1, 0(R2), 0(R4)
	RET
