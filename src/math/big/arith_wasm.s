// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !math_big_pure_go

#include "textflag.h"

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

