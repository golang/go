// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// func Fabs(x float64) float64
TEXT ·Fabs(SB),7,$0
	MOVQ   $(1<<63), BX
	MOVQ   BX, X0 // movsd $(-0.0), x0
	MOVSD  x+0(FP), X1
	ANDNPD X1, X0
	MOVSD  X0, r+8(FP)
	RET
