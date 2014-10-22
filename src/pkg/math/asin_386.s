// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func Asin(x float64) float64
TEXT ·Asin(SB),NOSPLIT,$0
	FMOVD   x+0(FP), F0  // F0=sin(x)
	FMOVD   F0, F1       // F0=sin(x), F1=sin(x)
	FMULD   F0, F0       // F0=sin(x)*sin(x), F1=sin(x)
	FLD1                 // F0=1, F1=sin(x)*sin(x), F2=sin(x)
	FSUBRDP F0, F1       // F0=1-sin(x)*sin(x) (=cos(x)*cos(x)), F1=sin(x)
	FSQRT                // F0=cos(x), F1=sin(x)
	FPATAN               // F0=arcsin(sin(x))=x
	FMOVDP  F0, ret+8(FP)
	RET

// func Acos(x float64) float64
TEXT ·Acos(SB),NOSPLIT,$0
	FMOVD   x+0(FP), F0  // F0=cos(x)
	FMOVD   F0, F1       // F0=cos(x), F1=cos(x)
	FMULD   F0, F0       // F0=cos(x)*cos(x), F1=cos(x)
	FLD1                 // F0=1, F1=cos(x)*cos(x), F2=cos(x)
	FSUBRDP F0, F1       // F0=1-cos(x)*cos(x) (=sin(x)*sin(x)), F1=cos(x)
	FSQRT                // F0=sin(x), F1=cos(x)
	FXCHD   F0, F1       // F0=cos(x), F1=sin(x)
	FPATAN               // F0=arccos(cos(x))=x
	FMOVDP	F0, ret+8(FP)
	RET
