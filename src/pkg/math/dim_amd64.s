// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// func Dim(x, y float64) float64
TEXT ·Dim(SB),7,$0
	MOVSD x+0(FP), X0
	SUBSD y+8(FP), X0
	MOVSD $(0.0), X1
	MAXSD X1, X0
	MOVSD X0, r+16(FP)
	RET

// func ·Max(x, y float64) float64
TEXT ·Max(SB),7,$0
	MOVSD x+0(FP), X0
	MAXSD y+8(FP), X0
	MOVSD X0, r+16(FP)
	RET

// func Min(x, y float64) float64
TEXT ·Min(SB),7,$0
	MOVSD x+0(FP), X0
	MINSD y+8(FP), X0
	MOVSD X0, r+16(FP)
	RET
