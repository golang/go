// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

TEXT ·IndexByte(SB),7,$0
	MOVL	s+0(FP), SI
	MOVL	s_len+4(FP), CX
	MOVB	c+12(FP), AL
	MOVL	SI, DI
	CLD; REPN; SCASB
	JZ 3(PC)
	MOVL	$-1, ret+16(FP)
	RET
	SUBL	SI, DI
	SUBL	$1, DI
	MOVL	DI, ret+16(FP)
	RET

TEXT ·Equal(SB),7,$0
	MOVL	a_len+4(FP), BX
	MOVL	b_len+16(FP), CX
	MOVL	$0, AX
	CMPL	BX, CX
	JNE	eqret
	MOVL	a+0(FP), SI
	MOVL	b+12(FP), DI
	CLD
	REP; CMPSB
	JNE eqret
	MOVL	$1, AX
eqret:
	MOVB	AX, ret+24(FP)
	RET
