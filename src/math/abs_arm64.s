// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT ·Abs(SB),NOSPLIT,$0-16
	FMOVD	x+0(FP), F3
	FABSD	F3, F3
	FMOVD	F3, ret+8(FP)
	RET
