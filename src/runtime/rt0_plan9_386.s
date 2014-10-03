// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT _rt0_386_plan9(SB),NOSPLIT,$12
	MOVL	AX, _tos(SB)
	LEAL	8(SP), AX
	MOVL	AX, _privates(SB)
	MOVL	$1, _nprivates(SB)
	CALL	runtime·asminit(SB)
	MOVL	inargc-4(FP), AX
	MOVL	AX, 0(SP)
	LEAL	inargv+0(FP), AX
	MOVL	AX, 4(SP)
	CALL	runtime·rt0_go(SB)

DATA  runtime·isplan9(SB)/4, $1
GLOBL runtime·isplan9(SB), NOPTR, $4
GLOBL _tos(SB), NOPTR, $4
GLOBL _privates(SB), NOPTR, $4
GLOBL _nprivates(SB), NOPTR, $4
