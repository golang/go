// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

#include "textflag.h"

// func sgemmKernelSVE(c, a, b *float32, k, bstride int64)
//
// The same instruction sequence in the sgemmSIMD
// innermost loop.
TEXT ·sgemmKernelSVE(SB), NOSPLIT, $0-40
	MOVD cioff+0(FP), R0
	MOVD ai+8(FP), R1
	MOVD boff+16(FP), R2
	MOVD k+24(FP), R3
	MOVD vlx4+32(FP), R4

	// Construct the all-true predicate
	MOVD $8, R5
	MOVD ZR, R6
	PWHILELT R5, R6, P0.S

	ZDUP $0, Z0.S     // the C strip accumulator

loop:
	FMOVS (R1), F1          // a[i][p]
	ZDUP  Z1.S[0], Z1.S
	ZLDR  (VL*0)(R2), Z2    // B[p][off:off+vl]
	ZFMLA Z2.S, Z1.S, P0.M, Z0.S
	ADD   $4, R1
	ADD   R4, R2
	SUBS  $1, R3
	BNE   loop

	ZSTR Z0, (VL*0)(R0)
	RET
