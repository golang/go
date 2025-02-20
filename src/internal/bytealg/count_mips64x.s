// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build mips64 || mips64le

#include "go_asm.h"
#include "textflag.h"

TEXT ·Count(SB),NOSPLIT,$0-40
	// R1 = b_base
	// R2 = b_len
	// R3 = byte to count
	MOVV	b_base+0(FP), R1
	MOVV	b_len+8(FP), R2
	MOVBU	c+24(FP), R3
	MOVV	R0, R5	// count
	ADDV	R1, R2	// end

loop:
	BEQ	R1, R2, done
	MOVBU	(R1), R6
	ADDV	$1, R1
	BNE	R3, R6, loop
	ADDV	$1, R5
	JMP	loop

done:
	MOVV	R5, ret+32(FP)
	RET

TEXT ·CountString(SB),NOSPLIT,$0-32
	// R1 = s_base
	// R2 = s_len
	// R3 = byte to count
	MOVV	s_base+0(FP), R1
	MOVV	s_len+8(FP), R2
	MOVBU	c+16(FP), R3
	MOVV	R0, R5	// count
	ADDV	R1, R2	// end

loop:
	BEQ	R1, R2, done
	MOVBU	(R1), R6
	ADDV	$1, R1
	BNE	R3, R6, loop
	ADDV	$1, R5
	JMP	loop

done:
	MOVV	R5, ret+24(FP)
	RET
