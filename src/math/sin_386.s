// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func Cos(x float64) float64
TEXT ·Cos(SB),NOSPLIT,$0
	JMP ·cos(SB)

// func Sin(x float64) float64
TEXT ·Sin(SB),NOSPLIT,$0
	JMP ·sin(SB)
