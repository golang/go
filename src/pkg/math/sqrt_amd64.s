// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// func Sqrt(x float64) float64
TEXT ·Sqrt(SB),7,$0
	MOVSD x+0(FP), X0
	SQRTSD X0, X0
	MOVSD X0, r+8(FP)
	RET
