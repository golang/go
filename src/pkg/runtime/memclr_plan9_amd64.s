// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// void runtime·memclr(void*, uintptr)
TEXT runtime·memclr(SB),NOSPLIT,$0-16
	MOVQ	ptr+0(FP), DI
	MOVQ	n+8(FP), CX
	MOVQ	CX, BX
	ANDQ	$7, BX
	SHRQ	$3, CX
	MOVQ	$0, AX
	CLD
	REP
	STOSQ
	MOVQ	BX, CX
	REP
	STOSB
	RET
