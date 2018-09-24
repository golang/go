// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

TEXT ·IndexByte(SB),NOSPLIT,$0-20
	MOVL b_base+0(FP), SI
	MOVL b_len+4(FP), BX
	MOVB c+12(FP), AL
	CALL indexbytebody<>(SB)
	MOVL AX, ret+16(FP)
	RET

TEXT ·IndexByteString(SB),NOSPLIT,$0-20
	MOVL s_base+0(FP), SI
	MOVL s_len+4(FP), BX
	MOVB c+8(FP), AL
	CALL indexbytebody<>(SB)
	MOVL AX, ret+16(FP)
	RET

TEXT bytes·IndexByte(SB),NOSPLIT,$0-20
	FUNCDATA $0, ·IndexByte·args_stackmap(SB)
	MOVL b_base+0(FP), SI
	MOVL b_len+4(FP), BX
	MOVB c+12(FP), AL
	CALL indexbytebody<>(SB)
	MOVL AX, ret+16(FP)
	RET

TEXT strings·IndexByte(SB),NOSPLIT,$0-20
	FUNCDATA $0, ·IndexByteString·args_stackmap(SB)
	MOVL s_base+0(FP), SI
	MOVL s_len+4(FP), BX
	MOVB c+8(FP), AL
	CALL indexbytebody<>(SB)
	MOVL AX, ret+16(FP)
	RET

// input:
//   SI: data
//   BX: data len
//   AL: byte sought
// output:
//   AX
TEXT indexbytebody<>(SB),NOSPLIT,$0
	MOVL SI, DI

	CMPL BX, $16
	JLT small

	// round up to first 16-byte boundary
	TESTL $15, SI
	JZ aligned
	MOVL SI, CX
	ANDL $~15, CX
	ADDL $16, CX

	// search the beginning
	SUBL SI, CX
	REPN; SCASB
	JZ success

// DI is 16-byte aligned; get ready to search using SSE instructions
aligned:
	// round down to last 16-byte boundary
	MOVL BX, R11
	ADDL SI, R11
	ANDL $~15, R11

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
	ADDL $16, DI

condition:
	CMPL DI, R11
	JNE sse

	// search the end
	MOVL SI, CX
	ADDL BX, CX
	SUBL R11, CX
	// if CX == 0, the zero flag will be set and we'll end up
	// returning a false success
	JZ failure
	REPN; SCASB
	JZ success

failure:
	MOVL $-1, AX
	RET

// handle for lengths < 16
small:
	MOVL BX, CX
	REPN; SCASB
	JZ success
	MOVL $-1, AX
	RET

// we've found the chunk containing the byte
// now just figure out which specific byte it is
ssesuccess:
	// get the index of the least significant set bit
	BSFW DX, DX
	SUBL SI, DI
	ADDL DI, DX
	MOVL DX, AX
	RET

success:
	SUBL SI, DI
	SUBL $1, DI
	MOVL DI, AX
	RET
