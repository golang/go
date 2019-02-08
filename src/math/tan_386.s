// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func Tan(x float64) float64
TEXT ·Tan(SB),NOSPLIT,$0
	JMP     ·tan(SB)
