// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT runtime·memmove(SB), NOSPLIT, $0-12
	MOVL	to+0(FP), DI
	MOVL	from+4(FP), SI
	MOVL	n+8(FP), BX

	CMPL	SI, DI
	JLS back

forward:
	MOVL	BX, CX
	SHRL	$3, CX
	ANDL	$7, BX
	REP; MOVSQ
	MOVL	BX, CX
	REP; MOVSB
	RET

back:
	MOVL	SI, CX
	ADDL	BX, CX
	CMPL	CX, DI
	JLS forward

	ADDL	BX, DI
	ADDL	BX, SI
	STD
	
	MOVL	BX, CX
	SHRL	$3, CX
	ANDL	$7, BX
	SUBL	$8, DI
	SUBL	$8, SI
	REP; MOVSQ
	ADDL	$7, DI
	ADDL	$7, SI
	MOVL	BX, CX
	REP; MOVSB
	CLD

	RET
