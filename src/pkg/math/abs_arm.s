// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

TEXT ·Abs(SB),7,$0
	MOVW	lo+0(FP), R0
	MOVW	hi+4(FP), R1
	AND 	$((1<<31)-1), R1
	MOVW	R0, resultlo+8(FP)
	MOVW	R1, resulthi+12(FP)
	RET
