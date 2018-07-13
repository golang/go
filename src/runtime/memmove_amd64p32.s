// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// This could use MOVSQ, but we use MOVSL so that if an object ends in
// a 4 byte pointer, we copy it as a unit instead of byte by byte.

// func memmove(to, from unsafe.Pointer, n uintptr)
TEXT runtime·memmove(SB), NOSPLIT, $0-12
	MOVL	to+0(FP), DI
	MOVL	from+4(FP), SI
	MOVL	n+8(FP), BX

	CMPL	SI, DI
	JLS back

forward:
	MOVL	BX, CX
	SHRL	$2, CX
	ANDL	$3, BX
	REP; MOVSL
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
	SHRL	$2, CX
	ANDL	$3, BX
	SUBL	$4, DI
	SUBL	$4, SI
	REP; MOVSL
	ADDL	$3, DI
	ADDL	$3, SI
	MOVL	BX, CX
	REP; MOVSB
	CLD

	// Note: we copy only 4 bytes at a time so that the tail is at most
	// 3 bytes. That guarantees that we aren't copying pointers with MOVSB.
	// See issue 13160.
	RET
