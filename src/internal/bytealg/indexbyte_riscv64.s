// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

TEXT ·IndexByte(SB),NOSPLIT,$0-40
	MOV	b_base+0(FP), A1
	MOV	b_len+8(FP), A2
	MOVBU	c+24(FP), A3	// byte to find
	MOV	A1, A4		// store base for later
	ADD	A1, A2		// end
	ADD	$-1, A1

loop:
	ADD	$1, A1
	BEQ	A1, A2, notfound
	MOVBU	(A1), A5
	BNE	A3, A5, loop

	SUB	A4, A1		// remove base
	MOV	A1, ret+32(FP)
	RET

notfound:
	MOV	$-1, A1
	MOV	A1, ret+32(FP)
	RET

TEXT ·IndexByteString(SB),NOSPLIT,$0-32
	MOV	s_base+0(FP), A1
	MOV	s_len+8(FP), A2
	MOVBU	c+16(FP), A3	// byte to find
	MOV	A1, A4		// store base for later
	ADD	A1, A2		// end
	ADD	$-1, A1

loop:
	ADD	$1, A1
	BEQ	A1, A2, notfound
	MOVBU	(A1), A5
	BNE	A3, A5, loop

	SUB	A4, A1		// remove base
	MOV	A1, ret+24(FP)
	RET

notfound:
	MOV	$-1, A1
	MOV	A1, ret+24(FP)
	RET
