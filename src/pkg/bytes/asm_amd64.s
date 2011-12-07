// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

TEXT ·IndexByte(SB),7,$0
	MOVQ p+0(FP), SI
	MOVL len+8(FP), BX
	MOVB b+16(FP), AL
	MOVQ SI, DI

	CMPL BX, $16
	JLT small

	// round up to first 16-byte boundary
	TESTQ $15, SI
	JZ aligned
	MOVQ SI, CX
	ANDQ $~15, CX
	ADDQ $16, CX

	// search the beginning
	SUBQ SI, CX
	REPN; SCASB
	JZ success

// DI is 16-byte aligned; get ready to search using SSE instructions
aligned:
	// round down to last 16-byte boundary
	MOVQ BX, R11
	ADDQ SI, R11
	ANDQ $~15, R11

	// shuffle X0 around so that each byte contains c
	MOVD AX, X0
	PUNPCKLBW X0, X0
	PUNPCKLBW X0, X0
	PSHUFL $0, X0, X0
	JMP condition

sse:
	// move the next 16-byte chunk of the buffer into X1
	MOVO (DI), X1
	// compare bytes in X0 to X1
	PCMPEQB X0, X1
	// take the top bit of each byte in X1 and put the result in DX
	PMOVMSKB X1, DX
	TESTL DX, DX
	JNZ ssesuccess
	ADDQ $16, DI

condition:
	CMPQ DI, R11
	JLT sse

	// search the end
	MOVQ SI, CX
	ADDQ BX, CX
	SUBQ R11, CX
	// if CX == 0, the zero flag will be set and we'll end up
	// returning a false success
	JZ failure
	REPN; SCASB
	JZ success

failure:
	MOVL $-1, ret+24(FP)
	RET

// handle for lengths < 16
small:
	MOVL BX, CX
	REPN; SCASB
	JZ success
	MOVL $-1, ret+24(FP)
	RET

// we've found the chunk containing the byte
// now just figure out which specific byte it is
ssesuccess:
	// get the index of the least significant set bit
	BSFW DX, DX
	SUBQ SI, DI
	ADDQ DI, DX
	MOVL DX, ret+24(FP)
	RET

success:
	SUBQ SI, DI
	SUBL $1, DI
	MOVL DI, ret+24(FP)
	RET

TEXT ·Equal(SB),7,$0
	MOVL	len+8(FP), BX
	MOVL	len1+24(FP), CX
	MOVL	$0, AX
	MOVL	$1, DX
	CMPL	BX, CX
	JNE	eqret
	MOVQ	p+0(FP), SI
	MOVQ	q+16(FP), DI
	CLD
	REP; CMPSB
	CMOVLEQ	DX, AX
eqret:
	MOVB	AX, ret+32(FP)
	RET

