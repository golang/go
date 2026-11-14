// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

//in plan 9 argc is at top of stack followed by ptrs to arguments

TEXT _rt0_arm64_plan9(SB),NOSPLIT|NOFRAME,$0
	MOVD	R0, _tos(SB)
	MOVD	0(RSP), R0
	MOVD	$8(RSP), R1
	MOVD	$runtime·rt0_go(SB), R2
	BL	(R2)

GLOBL _tos(SB), NOPTR, $8
