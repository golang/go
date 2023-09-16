// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

TEXT ·IndexByte(SB),NOSPLIT,$0-40
	MOVV	b_base+0(FP), R4
	MOVV	b_len+8(FP), R5
	MOVBU	c+24(FP), R6	// byte to find
	MOVV	R4, R7		// store base for later
	ADDV	R4, R5		// end
	ADDV	$-1, R4

	PCALIGN	$16
loop:
	ADDV	$1, R4
	BEQ	R4, R5, notfound
	MOVBU	(R4), R8
	BNE	R6, R8, loop

	SUBV	R7, R4		// remove base
	MOVV	R4, ret+32(FP)
	RET

notfound:
	MOVV	$-1, R4
	MOVV	R4, ret+32(FP)
	RET

TEXT ·IndexByteString(SB),NOSPLIT,$0-32
	MOVV	s_base+0(FP), R4
	MOVV	s_len+8(FP), R5
	MOVBU	c+16(FP), R6	// byte to find
	MOVV	R4, R7		// store base for later
	ADDV	R4, R5		// end
	ADDV	$-1, R4

	PCALIGN	$16
loop:
	ADDV	$1, R4
	BEQ	R4, R5, notfound
	MOVBU	(R4), R8
	BNE	R6, R8, loop

	SUBV	R7, R4		// remove base
	MOVV	R4, ret+24(FP)
	RET

notfound:
	MOVV	$-1, R4
	MOVV	R4, ret+24(FP)
	RET
