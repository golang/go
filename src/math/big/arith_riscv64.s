// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !math_big_pure_go && riscv64
// +build !math_big_pure_go,riscv64

#include "textflag.h"

// This file provides fast assembly versions for the elementary
// arithmetic operations on vectors implemented in arith.go.

// func mulWW(x, y Word) (z1, z0 Word)
TEXT ·mulWW(SB),NOSPLIT,$0
	MOV	x+0(FP), X5
	MOV	y+8(FP), X6
	MULHU	X5, X6, X7
	MUL	X5, X6, X8
	MOV	X7, z1+16(FP)
	MOV	X8, z0+24(FP)
	RET


TEXT ·addVV(SB),NOSPLIT,$0
	JMP ·addVV_g(SB)

TEXT ·subVV(SB),NOSPLIT,$0
	JMP ·subVV_g(SB)

TEXT ·addVW(SB),NOSPLIT,$0
	JMP ·addVW_g(SB)

TEXT ·subVW(SB),NOSPLIT,$0
	JMP ·subVW_g(SB)

TEXT ·shlVU(SB),NOSPLIT,$0
	JMP ·shlVU_g(SB)

TEXT ·shrVU(SB),NOSPLIT,$0
	JMP ·shrVU_g(SB)

TEXT ·mulAddVWW(SB),NOSPLIT,$0
	JMP ·mulAddVWW_g(SB)

TEXT ·addMulVVW(SB),NOSPLIT,$0
	JMP ·addMulVVW_g(SB)

