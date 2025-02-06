// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build riscv64

#include "textflag.h"

// func testMIN1(a int64) (r int64)
TEXT ·testMIN1(SB),NOSPLIT,$0-16
	MOV	a+0(FP), X5
	MIN	X5, X5, X6
	MOV	X6, r+8(FP)
	RET

// func testMIN2(a, b int64) (r int64)
TEXT ·testMIN2(SB),NOSPLIT,$0-24
	MOV	a+0(FP), X5
	MOV	b+8(FP), X6
	MIN	X5, X6, X6
	MOV	X6, r+16(FP)
	RET

// func testMIN3(a, b int64) (r int64)
TEXT ·testMIN3(SB),NOSPLIT,$0-24
	MOV	a+0(FP), X5
	MOV	b+8(FP), X6
	MIN	X6, X5, X5
	MOV	X5, r+16(FP)
	RET

// func testMIN4(a, b int64) (r int64)
TEXT ·testMIN4(SB),NOSPLIT,$0-24
	MOV	a+0(FP), X5
	MOV	b+8(FP), X6
	MIN	X5, X6, X7
	MOV	X7, r+16(FP)
	RET

// func testMAX1(a int64) (r int64)
TEXT ·testMAX1(SB),NOSPLIT,$0-16
	MOV	a+0(FP), X5
	MAX	X5, X5, X6
	MOV	X6, r+8(FP)
	RET

// func testMAX2(a, b int64) (r int64)
TEXT ·testMAX2(SB),NOSPLIT,$0-24
	MOV	a+0(FP), X5
	MOV	b+8(FP), X6
	MAX	X5, X6, X6
	MOV	X6, r+16(FP)
	RET

// func testMAX3(a, b int64) (r int64)
TEXT ·testMAX3(SB),NOSPLIT,$0-24
	MOV	a+0(FP), X5
	MOV	b+8(FP), X6
	MAX	X6, X5, X5
	MOV	X5, r+16(FP)
	RET

// func testMAX4(a, b int64) (r int64)
TEXT ·testMAX4(SB),NOSPLIT,$0-24
	MOV	a+0(FP), X5
	MOV	b+8(FP), X6
	MAX	X5, X6, X7
	MOV	X7, r+16(FP)
	RET

// func testMINU1(a int64) (r int64)
TEXT ·testMINU1(SB),NOSPLIT,$0-16
	MOV	a+0(FP), X5
	MINU	X5, X5, X6
	MOV	X6, r+8(FP)
	RET

// func testMINU2(a, b int64) (r int64)
TEXT ·testMINU2(SB),NOSPLIT,$0-24
	MOV	a+0(FP), X5
	MOV	b+8(FP), X6
	MINU	X5, X6, X6
	MOV	X6, r+16(FP)
	RET

// func testMINU3(a, b int64) (r int64)
TEXT ·testMINU3(SB),NOSPLIT,$0-24
	MOV	a+0(FP), X5
	MOV	b+8(FP), X6
	MINU	X6, X5, X5
	MOV	X5, r+16(FP)
	RET

// func testMINU4(a, b int64) (r int64)
TEXT ·testMINU4(SB),NOSPLIT,$0-24
	MOV	a+0(FP), X5
	MOV	b+8(FP), X6
	MINU	X5, X6, X7
	MOV	X7, r+16(FP)
	RET

// func testMAXU1(a int64) (r int64)
TEXT ·testMAXU1(SB),NOSPLIT,$0-16
	MOV	a+0(FP), X5
	MAXU	X5, X5, X6
	MOV	X6, r+8(FP)
	RET

// func testMAXU2(a, b int64) (r int64)
TEXT ·testMAXU2(SB),NOSPLIT,$0-24
	MOV	a+0(FP), X5
	MOV	b+8(FP), X6
	MAXU	X5, X6, X6
	MOV	X6, r+16(FP)
	RET

// func testMAXU3(a, b int64) (r int64)
TEXT ·testMAXU3(SB),NOSPLIT,$0-24
	MOV	a+0(FP), X5
	MOV	b+8(FP), X6
	MAXU	X6, X5, X5
	MOV	X5, r+16(FP)
	RET

// func testMAXU4(a, b int64) (r int64)
TEXT ·testMAXU4(SB),NOSPLIT,$0-24
	MOV	a+0(FP), X5
	MOV	b+8(FP), X6
	MAXU	X5, X6, X7
	MOV	X7, r+16(FP)
	RET
