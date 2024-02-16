// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ppc64 || ppc64le

#include "textflag.h"

TEXT ·archFloor(SB),NOSPLIT,$0
	FMOVD   x+0(FP), F0
	FRIM	F0, F0
	FMOVD   F0, ret+8(FP)
	RET

TEXT ·archCeil(SB),NOSPLIT,$0
	FMOVD   x+0(FP), F0
	FRIP    F0, F0
	FMOVD	F0, ret+8(FP)
	RET

TEXT ·archTrunc(SB),NOSPLIT,$0
	FMOVD   x+0(FP), F0
	FRIZ    F0, F0
	FMOVD   F0, ret+8(FP)
	RET
