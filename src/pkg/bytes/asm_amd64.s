// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

TEXT ·IndexByte(SB),7,$0
	MOVQ	p+0(FP), SI
	MOVL	len+8(FP), CX
	MOVB	b+16(FP), AL
	MOVQ	SI, DI
	REPN; SCASB
	JZ 3(PC)
	MOVL	$-1, ret+24(FP)
	RET
	SUBQ	SI, DI
	SUBL	$1, DI
	MOVL	DI, ret+24(FP)
	RET
