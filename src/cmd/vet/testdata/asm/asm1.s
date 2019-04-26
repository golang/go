// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build amd64

TEXT ·arg1(SB),0,$0-2
	MOVW	x+0(FP), AX // ERROR "\[amd64\] arg1: invalid MOVW of x\+0\(FP\); int8 is 1-byte value"
