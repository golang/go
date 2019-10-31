// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build amd64

TEXT ·arg1(SB),0,$0-2
	MOVW	x+0(FP), AX // ERROR "\[amd64\] arg1: invalid MOVW of x\+0\(FP\); int8 is 1-byte value"

TEXT ·cpx(SB),0,$0-24
	// These are ok
	MOVSS	x_real+0(FP), X0
	MOVSS	x_imag+4(FP), X0
	MOVSD	y_real+8(FP), X0
	MOVSD	y_imag+16(FP), X0
	// Loading both parts of a complex is ok: see issue 35264.
	MOVSD	x+0(FP), X0
	MOVO	y+8(FP), X0
	MOVOU	y+8(FP), X0

	// These are not ok.
	MOVO	x+0(FP), X0 // ERROR "\[amd64\] cpx: invalid MOVO of x\+0\(FP\); complex64 is 8-byte value containing x_real\+0\(FP\) and x_imag\+4\(FP\)"
	MOVSD	y+8(FP), X0 // ERROR "\[amd64\] cpx: invalid MOVSD of y\+8\(FP\); complex128 is 16-byte value containing y_real\+8\(FP\) and y_imag\+16\(FP\)"
