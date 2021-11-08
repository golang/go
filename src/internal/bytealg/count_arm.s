// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

TEXT ·Count(SB),NOSPLIT,$0-20
	MOVW	b_base+0(FP), R0
	MOVW	b_len+4(FP), R1
	MOVBU	c+12(FP), R2
	MOVW	$ret+16(FP), R7
	B	countbytebody<>(SB)

TEXT ·CountString(SB),NOSPLIT,$0-16
	MOVW	s_base+0(FP), R0
	MOVW	s_len+4(FP), R1
	MOVBU	c+8(FP), R2
	MOVW	$ret+12(FP), R7
	B	countbytebody<>(SB)

// Input:
// R0: data
// R1: data length
// R2: byte to find
// R7: address to put result
//
// On exit:
// R4 and R8 are clobbered
TEXT countbytebody<>(SB),NOSPLIT,$0
	MOVW	$0, R8	// R8 = count of byte to search
	CMP	$0, R1
	B.EQ	done	// short path to handle 0-byte case
	ADD	R0, R1	// R1 is the end of the range
byte_loop:
	MOVBU.P	1(R0), R4
	CMP	R4, R2
	ADD.EQ	$1, R8
	CMP	R0, R1
	B.NE	byte_loop
done:
	MOVW	R8, (R7)
	RET
