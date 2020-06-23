// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build riscv64

#include "textflag.h"

// func testBEQZ(a int64) (r bool)
TEXT ·testBEQZ(SB),NOSPLIT,$0-0
	MOV	a+0(FP), X5
	MOV	$1, X6
	BEQZ	X5, b
	MOV	$0, X6
b:
	MOV	X6, r+8(FP)
	RET

// func testBGEZ(a int64) (r bool)
TEXT ·testBGEZ(SB),NOSPLIT,$0-0
	MOV	a+0(FP), X5
	MOV	$1, X6
	BGEZ	X5, b
	MOV	$0, X6
b:
	MOV	X6, r+8(FP)
	RET

// func testBGT(a, b int64) (r bool)
TEXT ·testBGT(SB),NOSPLIT,$0-0
	MOV	a+0(FP), X5
	MOV	b+8(FP), X6
	MOV	$1, X7
	BGT	X5, X6, b
	MOV	$0, X7
b:
	MOV	X7, r+16(FP)
	RET

// func testBGTU(a, b int64) (r bool)
TEXT ·testBGTU(SB),NOSPLIT,$0-0
	MOV	a+0(FP), X5
	MOV	b+8(FP), X6
	MOV	$1, X7
	BGTU	X5, X6, b
	MOV	$0, X7
b:
	MOV	X7, r+16(FP)
	RET

// func testBGTZ(a int64) (r bool)
TEXT ·testBGTZ(SB),NOSPLIT,$0-0
	MOV	a+0(FP), X5
	MOV	$1, X6
	BGTZ	X5, b
	MOV	$0, X6
b:
	MOV	X6, r+8(FP)
	RET

// func testBLE(a, b int64) (r bool)
TEXT ·testBLE(SB),NOSPLIT,$0-0
	MOV	a+0(FP), X5
	MOV	b+8(FP), X6
	MOV	$1, X7
	BLE	X5, X6, b
	MOV	$0, X7
b:
	MOV	X7, r+16(FP)
	RET

// func testBLEU(a, b int64) (r bool)
TEXT ·testBLEU(SB),NOSPLIT,$0-0
	MOV	a+0(FP), X5
	MOV	b+8(FP), X6
	MOV	$1, X7
	BLEU	X5, X6, b
	MOV	$0, X7
b:
	MOV	X7, r+16(FP)
	RET

// func testBLEZ(a int64) (r bool)
TEXT ·testBLEZ(SB),NOSPLIT,$0-0
	MOV	a+0(FP), X5
	MOV	$1, X6
	BLEZ	X5, b
	MOV	$0, X6
b:
	MOV	X6, r+8(FP)
	RET

// func testBLTZ(a int64) (r bool)
TEXT ·testBLTZ(SB),NOSPLIT,$0-0
	MOV	a+0(FP), X5
	MOV	$1, X6
	BLTZ	X5, b
	MOV	$0, X6
b:
	MOV	X6, r+8(FP)
	RET

// func testBNEZ(a int64) (r bool)
TEXT ·testBNEZ(SB),NOSPLIT,$0-0
	MOV	a+0(FP), X5
	MOV	$1, X6
	BNEZ	X5, b
	MOV	$0, X6
b:
	MOV	X6, r+8(FP)
	RET
