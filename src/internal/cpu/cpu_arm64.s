// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func getisar0() uint64
TEXT ·getisar0(SB),NOSPLIT,$0
	// get Instruction Set Attributes 0 into R0
	MRS	ID_AA64ISAR0_EL1, R0
	MOVD	R0, ret+0(FP)
	RET

// func getMIDR() uint64
TEXT ·getMIDR(SB), NOSPLIT, $0-8
	MRS	MIDR_EL1, R0
	MOVD	R0, ret+0(FP)
	RET
