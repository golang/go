// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT ·Floor(SB),NOSPLIT,$0
	Get SP
	F64Load x+0(FP)
	F64Floor
	F64Store ret+8(FP)
	RET

TEXT ·Ceil(SB),NOSPLIT,$0
	Get SP
	F64Load x+0(FP)
	F64Ceil
	F64Store ret+8(FP)
	RET

TEXT ·Trunc(SB),NOSPLIT,$0
	Get SP
	F64Load x+0(FP)
	F64Trunc
	F64Store ret+8(FP)
	RET
