// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func Exp(x float64) float64
TEXT ·Exp(SB),NOSPLIT,$0
	// Used to use 387 assembly (FLDL2E+F2XM1) here,
	// but it was both slower and less accurate than the portable Go code.
	JMP ·exp(SB)
