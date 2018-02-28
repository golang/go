// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// void runtime·memclrNoHeapPointers(void*, uintptr)
TEXT runtime·memclrNoHeapPointers(SB),NOSPLIT,$0-16
	MOVD	ptr+0(FP), R0
	MOVD	n+8(FP), R1
	// If size is less than 16 bytes, use tail_zero to zero what remains
	CMP	$16, R1
	BLT	tail_zero
	// Get buffer offset into 16 byte aligned address for better performance
	ANDS	$15, R0, ZR
	BNE	unaligned_to_16
aligned_to_16:
	LSR	$4, R1, R2
zero_by_16:
	STP.P	(ZR, ZR), 16(R0)
	SUBS	$1, R2, R2
	BNE	zero_by_16

	ANDS	$15, R1, R1
	BEQ	ending

	// Zero buffer with size=R1 < 16
tail_zero:
	TBZ	$3, R1, tail_zero_4
	MOVD.P	ZR, 8(R0)

tail_zero_4:
	TBZ	$2, R1, tail_zero_2
	MOVW.P	ZR, 4(R0)

tail_zero_2:
	TBZ	$1, R1, tail_zero_1
	MOVH.P	ZR, 2(R0)

tail_zero_1:
	TBZ	$0, R1, ending
	MOVB	ZR, (R0)

ending:
	RET

unaligned_to_16:
	MOVD	R0, R2
head_loop:
	MOVBU.P	ZR, 1(R0)
	ANDS	$15, R0, ZR
	BNE	head_loop
	// Adjust length for what remains
	SUB	R2, R0, R3
	SUB	R3, R1
	// If size is less than 16 bytes, use tail_zero to zero what remains
	CMP	$16, R1
	BLT	tail_zero
	B	aligned_to_16
