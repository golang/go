// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// void runtime·memmove(void*, void*, uintptr)
TEXT runtime·memmove(SB), NOSPLIT, $-8-24
	MOVD	to+0(FP), R3
	MOVD	from+8(FP), R4
	MOVD	n+16(FP), R5
	CMP	$0, R5
	BNE	check
	RET

check:
	CMP	R3, R4
	BLT	backward

	ADD	R3, R5
loop:
	MOVBU.P	1(R4), R6
	MOVBU.P	R6, 1(R3)
	CMP	R3, R5
	BNE	loop
	RET

backward:
	ADD	R5, R4
	ADD	R3, R5
loop1:
	MOVBU.W	-1(R4), R6
	MOVBU.W	R6, -1(R5)
	CMP	R3, R5
	BNE	loop1
	RET
