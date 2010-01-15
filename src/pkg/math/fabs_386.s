// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// func Fabs(x float64) float64
TEXT math·Fabs(SB),7,$0
	FMOVD   x+0(FP), F0  // F0=x
	FABS                 // F0=|x|
	FMOVDP  F0, r+8(FP)
	RET
