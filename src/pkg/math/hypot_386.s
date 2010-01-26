// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// func Hypot(x, y float64) float64
TEXT ·Hypot(SB),7,$0
// test bits for not-finite
	MOVL    xh+4(FP), AX   // high word x
	ANDL    $0x7ff00000, AX
	CMPL    AX, $0x7ff00000
	JEQ     not_finite
	MOVL    yh+12(FP), AX   // high word y
	ANDL    $0x7ff00000, AX
	CMPL    AX, $0x7ff00000
	JEQ     not_finite
	FMOVD   x+0(FP), F0  // F0=x
	FABS                 // F0=|x|
	FMOVD   y+8(FP), F0  // F0=y, F1=|x|
	FABS                 // F0=|y|, F1=|x|
	FUCOMI  F0, F1       // compare F0 to F1
	JCC     2(PC)        // jump if F0 < F1
	FXCHD   F0, F1       // F0=|x| (larger), F1=|y| (smaller)
	FTST                 // compare F0 to 0
	FSTSW	AX
	ANDW    $0x4000, AX
	JNE		10(PC)       // jump if F0 = 0
	FXCHD   F0, F1       // F0=y (smaller), F1=x (larger)
	FDIVD   F1, F0       // F0=y(=y/x), F1=x
	FMULD   F0, F0       // F0=y*y, F1=x
	FLD1                 // F0=1, F1=y*y, F2=x
	FADDDP  F0, F1       // F0=1+y*y, F1=x
	FSQRT                // F0=sqrt(1+y*y), F1=x
	FMULDP  F0, F1       // F0=x*sqrt(1+y*y)
	FMOVDP  F0, r+16(FP)
	RET
	FMOVDP  F0, F1       // F0=0
	FMOVDP  F0, r+16(FP)
	RET
not_finite:
// test bits for -Inf or +Inf
	MOVL    xh+4(FP), AX  // high word x
	ORL     xl+0(FP), AX  // low word x
	ANDL    $0x7fffffff, AX
	CMPL    AX, $0x7ff00000
	JEQ     is_inf
	MOVL    yh+12(FP), AX  // high word y
	ORL     yl+8(FP), AX   // low word y
	ANDL    $0x7fffffff, AX
	CMPL    AX, $0x7ff00000
	JEQ     is_inf
	MOVL    $0x7ff00000, rh+20(FP)  // return NaN = 0x7FF0000000000001
	MOVL    $0x00000001, rl+16(FP)
	RET
is_inf:
	MOVL    AX, rh+20(FP)  // return +Inf = 0x7FF0000000000000
	MOVL    $0x00000000, rl+16(FP)
	RET
