// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT ·asmAddFlags(SB),NOSPLIT,$0-24
	MOVQ	x+0(FP), AX
	ADDQ	y+8(FP), AX
	PUSHFQ
	POPQ	AX
	MOVQ	AX, ret+16(FP)
	RET

TEXT ·asmSubFlags(SB),NOSPLIT,$0-24
	MOVQ	x+0(FP), AX
	SUBQ	y+8(FP), AX
	PUSHFQ
	POPQ	AX
	MOVQ	AX, ret+16(FP)
	RET

TEXT ·asmAndFlags(SB),NOSPLIT,$0-24
	MOVQ	x+0(FP), AX
	ANDQ	y+8(FP), AX
	PUSHFQ
	POPQ	AX
	MOVQ	AX, ret+16(FP)
	RET
