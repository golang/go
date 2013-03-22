// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

TEXT ·Abs(SB),7,$0
	MOVW	x_lo+0(FP), R0
	MOVW	x_hi+4(FP), R1
	AND 	$((1<<31)-1), R1
	MOVW	R0, ret_lo+8(FP)
	MOVW	R1, ret_hi+12(FP)
	RET
