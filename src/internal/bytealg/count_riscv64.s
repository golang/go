// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

TEXT ·Count(SB),NOSPLIT,$0-40
	MOV	b_base+0(FP), A1
	MOV	b_len+8(FP), A2
	MOVBU	c+24(FP), A3	// byte to count
	MOV	ZERO, A4	// count
	ADD	A1, A2		// end

loop:
	BEQ	A1, A2, done
	MOVBU	(A1), A5
	ADD	$1, A1
	BNE	A3, A5, loop
	ADD	$1, A4
	JMP	loop

done:
	MOV	A4, ret+32(FP)
	RET

TEXT ·CountString(SB),NOSPLIT,$0-32
	MOV	s_base+0(FP), A1
	MOV	s_len+8(FP), A2
	MOVBU	c+16(FP), A3	// byte to count
	MOV	ZERO, A4	// count
	ADD	A1, A2		// end

loop:
	BEQ	A1, A2, done
	MOVBU	(A1), A5
	ADD	$1, A1
	BNE	A3, A5, loop
	ADD	$1, A4
	JMP	loop

done:
	MOV	A4, ret+24(FP)
	RET
