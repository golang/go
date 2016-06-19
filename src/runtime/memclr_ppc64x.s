// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ppc64 ppc64le

#include "textflag.h"

// void runtime·memclr(void*, uintptr)
TEXT runtime·memclr(SB),NOSPLIT|NOFRAME,$0-16
	MOVD	ptr+0(FP), R3
	MOVD	n+8(FP), R4
	SRADCC	$3, R4, R6	// R6 is the number of words to zero
	BEQ	bytes

	SUB	$8, R3
	MOVD	R6, CTR
	MOVDU	R0, 8(R3)
	BC	25, 0, -1(PC)	// bdnz+ $-4
	ADD	$8, R3

bytes:
	ANDCC	$7, R4, R7	// R7 is the number of bytes to zero
	BEQ	done
	SUB	$1, R3
	MOVD	R7, CTR
	MOVBU	R0, 1(R3)
	BC	25, 0, -1(PC)	// bdnz+ $-4

done:
	RET
