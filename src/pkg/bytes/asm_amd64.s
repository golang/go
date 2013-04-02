// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

TEXT ·IndexByte(SB),7,$0
	MOVQ s+0(FP), SI
	MOVQ s_len+8(FP), BX
	MOVB c+24(FP), AL
	MOVQ SI, DI

	CMPQ BX, $16
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
	MOVQ $-1, ret+32(FP)
	RET

// handle for lengths < 16
small:
	MOVQ BX, CX
	REPN; SCASB
	JZ success
	MOVQ $-1, ret+32(FP)
	RET

// we've found the chunk containing the byte
// now just figure out which specific byte it is
ssesuccess:
	// get the index of the least significant set bit
	BSFW DX, DX
	SUBQ SI, DI
	ADDQ DI, DX
	MOVQ DX, ret+32(FP)
	RET

success:
	SUBQ SI, DI
	SUBL $1, DI
	MOVQ DI, ret+32(FP)
	RET
