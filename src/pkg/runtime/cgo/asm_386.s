// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

/*
 * void crosscall2(void (*fn)(void*, int32), void*, int32)
 * Save registers and call fn with two arguments.
 */
TEXT crosscall2(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	PUSHL	BX
	PUSHL	SI
	PUSHL	DI
	
	SUBL	$8, SP
	MOVL	16(BP), AX
	MOVL	AX, 4(SP)
	MOVL	12(BP), AX
	MOVL	AX, 0(SP)
	MOVL	8(BP), AX
	CALL	AX
	ADDL	$8, SP
	
	POPL	DI
	POPL	SI
	POPL	BX
	POPL	BP
	RET
