// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

TEXT ·IndexByte(SB),NOSPLIT,$0-20
	MOVL	b_base+0(FP), SI
	MOVL	b_len+4(FP), CX
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

TEXT ·IndexByteString(SB),NOSPLIT,$0-16
	MOVL	s_base+0(FP), SI
	MOVL	s_len+4(FP), CX
	MOVB	c+8(FP), AL
	MOVL	SI, DI
	CLD; REPN; SCASB
	JZ 3(PC)
	MOVL	$-1, ret+12(FP)
	RET
	SUBL	SI, DI
	SUBL	$1, DI
	MOVL	DI, ret+12(FP)
	RET

TEXT bytes·IndexByte(SB),NOSPLIT,$0-20
	FUNCDATA $0, ·IndexByte·args_stackmap(SB)
	JMP ·IndexByte(SB)

TEXT strings·IndexByte(SB),NOSPLIT,$0-16
	FUNCDATA $0, ·IndexByteString·args_stackmap(SB)
	JMP ·IndexByteString(SB)
