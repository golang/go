// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

TEXT ·IndexByte(SB),NOSPLIT,$0-20
	MOVW	b_base+0(FP), R0
	MOVW	b_len+4(FP), R1
	MOVBU	c+12(FP), R2	// byte to find
	MOVW	R0, R4		// store base for later
	ADD	R0, R1		// end

_loop:
	CMP	R0, R1
	B.EQ	_notfound
	MOVBU.P	1(R0), R3
	CMP	R2, R3
	B.NE	_loop

	SUB	$1, R0		// R0 will be one beyond the position we want
	SUB	R4, R0		// remove base
	MOVW    R0, ret+16(FP)
	RET

_notfound:
	MOVW	$-1, R0
	MOVW	R0, ret+16(FP)
	RET

TEXT ·IndexByteString(SB),NOSPLIT,$0-16
	MOVW	s_base+0(FP), R0
	MOVW	s_len+4(FP), R1
	MOVBU	c+8(FP), R2	// byte to find
	MOVW	R0, R4		// store base for later
	ADD	R0, R1		// end

_sib_loop:
	CMP	R0, R1
	B.EQ	_sib_notfound
	MOVBU.P	1(R0), R3
	CMP	R2, R3
	B.NE	_sib_loop

	SUB	$1, R0		// R0 will be one beyond the position we want
	SUB	R4, R0		// remove base
	MOVW	R0, ret+12(FP)
	RET

_sib_notfound:
	MOVW	$-1, R0
	MOVW	R0, ret+12(FP)
	RET
