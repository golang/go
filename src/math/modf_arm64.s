// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func archModf(f float64) (int float64, frac float64)
TEXT ·archModf(SB),NOSPLIT,$0
	MOVD	f+0(FP), R0
	FMOVD	R0, F0
	FRINTZD	F0, F1
	FMOVD	F1, int+8(FP)
	FSUBD	F1, F0
	FMOVD	F0, R1
	AND	$(1<<63), R0
	ORR	R0, R1 // must have same sign
	MOVD	R1, frac+16(FP)
	RET
