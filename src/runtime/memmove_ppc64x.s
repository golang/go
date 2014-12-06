// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ppc64 ppc64le

#include "textflag.h"

// void runtime·memmove(void*, void*, uintptr)
TEXT runtime·memmove(SB), NOSPLIT, $-8-24
	MOVD	to+0(FP), R3
	MOVD	from+8(FP), R4
	MOVD	n+16(FP), R5
	CMP	R5, $0
	BNE	check
	RETURN

check:
	CMP	R3, R4
	BGT	backward

	SUB	$1, R3
	ADD	R3, R5
	SUB	$1, R4
loop:
	MOVBU	1(R4), R6
	MOVBU	R6, 1(R3)
	CMP	R3, R5
	BNE	loop
	RETURN

backward:
	ADD	R5, R4
	ADD	R3, R5
loop1:
	MOVBU	-1(R4), R6
	MOVBU	R6, -1(R5)
	CMP	R3, R5
	BNE	loop1
	RETURN
