// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build mips mipsle

#include "go_asm.h"
#include "textflag.h"

TEXT ·IndexByte(SB),NOSPLIT,$0-20
	MOVW	b_base+0(FP), R1
	MOVW	b_len+4(FP), R2
	MOVBU	c+12(FP), R3	// byte to find
	ADDU	$1, R1, R4	// store base+1 for later
	ADDU	R1, R2	// end

loop:
	BEQ	R1, R2, notfound
	MOVBU	(R1), R5
	ADDU	$1, R1
	BNE	R3, R5, loop

	SUBU	R4, R1	// R1 will be one beyond the position we want so remove (base+1)
	MOVW	R1, ret+16(FP)
	RET

notfound:
	MOVW	$-1, R1
	MOVW	R1, ret+16(FP)
	RET

TEXT ·IndexByteString(SB),NOSPLIT,$0-16
	MOVW	s_base+0(FP), R1
	MOVW	s_len+4(FP), R2
	MOVBU	c+8(FP), R3	// byte to find
	ADDU	$1, R1, R4	// store base+1 for later
	ADDU	R1, R2	// end

loop:
	BEQ	R1, R2, notfound
	MOVBU	(R1), R5
	ADDU	$1, R1
	BNE	R3, R5, loop

	SUBU	R4, R1	// remove (base+1)
	MOVW	R1, ret+12(FP)
	RET

notfound:
	MOVW	$-1, R1
	MOVW	R1, ret+12(FP)
	RET

TEXT bytes·IndexByte(SB),NOSPLIT,$0-20
	FUNCDATA $0, ·IndexByte·args_stackmap(SB)
	JMP ·IndexByte(SB)

TEXT strings·IndexByte(SB),NOSPLIT,$0-16
	FUNCDATA $0, ·IndexByteString·args_stackmap(SB)
	JMP ·IndexByteString(SB)
