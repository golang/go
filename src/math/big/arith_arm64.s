// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !math_big_pure_go

#include "textflag.h"

// This file provides fast assembly versions for the elementary
// arithmetic operations on vectors implemented in arith.go.

TEXT ·mulWW(SB),NOSPLIT,$0
	B ·mulWW_g(SB)

TEXT ·divWW(SB),NOSPLIT,$0
	B ·divWW_g(SB)

TEXT ·addVV(SB),NOSPLIT,$0
	B ·addVV_g(SB)

TEXT ·subVV(SB),NOSPLIT,$0
	B ·subVV_g(SB)

TEXT ·addVW(SB),NOSPLIT,$0
	B ·addVW_g(SB)

TEXT ·subVW(SB),NOSPLIT,$0
	B ·subVW_g(SB)

TEXT ·shlVU(SB),NOSPLIT,$0
	B ·shlVU_g(SB)

TEXT ·shrVU(SB),NOSPLIT,$0
	B ·shrVU_g(SB)

TEXT ·mulAddVWW(SB),NOSPLIT,$0
	B ·mulAddVWW_g(SB)

TEXT ·addMulVVW(SB),NOSPLIT,$0
	B ·addMulVVW_g(SB)

TEXT ·divWVW(SB),NOSPLIT,$0
	B ·divWVW_g(SB)

TEXT ·bitLen(SB),NOSPLIT,$0
	B ·bitLen_g(SB)
