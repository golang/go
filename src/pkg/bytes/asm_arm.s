// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

TEXT ·IndexByte(SB),7,$0
	MOVW	s+0(FP), R0
	MOVW	s_len+4(FP), R1
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

TEXT ·Equal(SB),7,$0
	MOVW	a_len+4(FP), R1
	MOVW	b_len+16(FP), R3
	
	CMP	R1, R3		// unequal lengths are not equal
	B.NE	_notequal

	MOVW	a+0(FP), R0
	MOVW	b+12(FP), R2
	ADD	R0, R1		// end

_next:
	CMP	R0, R1
	B.EQ	_equal		// reached the end
	MOVBU.P	1(R0), R4
	MOVBU.P	1(R2), R5
	CMP	R4, R5
	B.EQ	_next

_notequal:
	MOVW	$0, R0
	MOVBU	R0, ret+24(FP)
	RET

_equal:
	MOVW	$1, R0
	MOVBU	R0, ret+24(FP)
	RET
