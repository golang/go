// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ppc64 ppc64le

#include "textflag.h"

// void runtime·memclr(void*, uintptr)
TEXT runtime·memclr(SB),NOSPLIT,$0-16
	MOVD	ptr+0(FP), R3
	MOVD	n+8(FP), R4
	CMP	R4, $0
	BEQ	done
	SUB	$1, R3
	MOVD	R4, CTR
	MOVBU	R0, 1(R3)
	BC	25, 0, -1(PC) // bdnz+ $-4
done:
	RETURN
