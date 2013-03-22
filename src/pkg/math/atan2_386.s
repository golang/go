// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// func Atan2(y, x float64) float64	// =atan(y/x)
TEXT ·Atan2(SB),7,$0
	FMOVD   y+0(FP), F0  // F0=y
	FMOVD   x+8(FP), F0  // F0=x, F1=y
	FPATAN               // F0=atan(F1/F0)
	FMOVDP  F0, ret+16(FP)
	RET
