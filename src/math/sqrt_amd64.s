// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func Sqrt(x float64) float64
TEXT ·Sqrt(SB), NOSPLIT, $0
	MOVSD  x+0(FP), X0
	SQRTSD X0, X1
	MOVSD  X1, ret+8(FP)
	RET
