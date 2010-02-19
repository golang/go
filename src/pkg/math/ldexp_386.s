// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// func Ldexp(f float64, e int) float64
TEXT ·Ldexp(SB),7,$0
	FMOVL   e+8(FP), F0   // F0=e
	FMOVD   x+0(FP), F0   // F0=x, F1=e
	FSCALE                // F0=x*2**e, F1=e
	FMOVDP  F0, F1        // F0=x*2**e
	FMOVDP  F0, r+12(FP)
	RET
