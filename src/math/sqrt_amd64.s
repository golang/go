// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func archSqrt(x float64) float64
TEXT ·archSqrt(SB), NOSPLIT, $0
	XORPS  X0, X0 // break dependency
	SQRTSD x+0(FP), X0
	MOVSD  X0, ret+8(FP)
	RET
