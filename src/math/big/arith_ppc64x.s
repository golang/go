// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ppc64 ppc64le

#include "textflag.h"

// This file provides fast assembly versions for the elementary
// arithmetic operations on vectors implemented in arith.go.

TEXT ·mulWW(SB),NOSPLIT,$0
	BR ·mulWW_g(SB)

TEXT ·divWW(SB),NOSPLIT,$0
	BR ·divWW_g(SB)

TEXT ·addVV(SB),NOSPLIT,$0
	BR ·addVV_g(SB)

TEXT ·subVV(SB),NOSPLIT,$0
	BR ·subVV_g(SB)

TEXT ·addVW(SB),NOSPLIT,$0
	BR ·addVW_g(SB)

TEXT ·subVW(SB),NOSPLIT,$0
	BR ·subVW_g(SB)

TEXT ·shlVU(SB),NOSPLIT,$0
	BR ·shlVU_g(SB)

TEXT ·shrVU(SB),NOSPLIT,$0
	BR ·shrVU_g(SB)

TEXT ·mulAddVWW(SB),NOSPLIT,$0
	BR ·mulAddVWW_g(SB)

TEXT ·addMulVVW(SB),NOSPLIT,$0
	BR ·addMulVVW_g(SB)

TEXT ·divWVW(SB),NOSPLIT,$0
	BR ·divWVW_g(SB)

TEXT ·bitLen(SB),NOSPLIT,$0
	BR ·bitLen_g(SB)
