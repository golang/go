// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ppc64 ppc64le

#include "textflag.h"

// func Modf(f float64) (int float64, frac float64)
TEXT ·Modf(SB),NOSPLIT,$0
	FMOVD	f+0(FP), F0
	FRIZ	F0, F1
	FMOVD	F1, int+8(FP)
	FSUB	F1, F0, F2
	FCPSGN	F2, F0, F2
	FMOVD	F2, frac+16(FP)
	RET
