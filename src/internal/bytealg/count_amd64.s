// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "asm_amd64.h"
#include "textflag.h"

TEXT ·Count(SB),NOSPLIT,$0-40
#ifndef hasPOPCNT
	CMPB	internal∕cpu·X86+const_offsetX86HasPOPCNT(SB), $1
	JEQ	2(PC)
	JMP	·countGeneric(SB)
#endif
	MOVQ	b_base+0(FP), SI
	MOVQ	b_len+8(FP), BX
	MOVB	c+24(FP), AL
	LEAQ	ret+32(FP), R8
	JMP	countbody<>(SB)

TEXT ·CountString(SB),NOSPLIT,$0-32
#ifndef hasPOPCNT
	CMPB	internal∕cpu·X86+const_offsetX86HasPOPCNT(SB), $1
	JEQ	2(PC)
	JMP	·countGenericString(SB)
#endif
	MOVQ	s_base+0(FP), SI
	MOVQ	s_len+8(FP), BX
	MOVB	c+16(FP), AL
	LEAQ	ret+24(FP), R8
	JMP	countbody<>(SB)

// input:
//   SI: data
//   BX: data len
//   AL: byte sought
//   R8: address to put result
// This function requires the POPCNT instruction.
TEXT countbody<>(SB),NOSPLIT,$0
	// Shuffle X0 around so that each byte contains
	// the character we're looking for.
	MOVD AX, X0
	PUNPCKLBW X0, X0
	PUNPCKLBW X0, X0
	PSHUFL $0, X0, X0

	CMPQ BX, $16
	JLT small

	MOVQ $0, R12 // Accumulator

	MOVQ SI, DI

	CMPQ BX, $64
	JAE avx2
sse:
	LEAQ	-16(SI)(BX*1), AX	// AX = address of last 16 bytes
	JMP	sseloopentry

	PCALIGN $16
sseloop:
	// Move the next 16-byte chunk of the data into X1.
	MOVOU	(DI), X1
	// Compare bytes in X0 to X1.
	PCMPEQB	X0, X1
	// Take the top bit of each byte in X1 and put the result in DX.
	PMOVMSKB X1, DX
	// Count number of matching bytes
	POPCNTL DX, DX
	// Accumulate into R12
	ADDQ DX, R12
	// Advance to next block.
	ADDQ	$16, DI
sseloopentry:
	CMPQ	DI, AX
	JBE	sseloop

	// Get the number of bytes to consider in the last 16 bytes
	ANDQ $15, BX
	JZ end

	// Create mask to ignore overlap between previous 16 byte block
	// and the next.
	MOVQ $16,CX
	SUBQ BX, CX
	MOVQ $0xFFFF, R10
	SARQ CL, R10
	SALQ CL, R10

	// Process the last 16-byte chunk. This chunk may overlap with the
	// chunks we've already searched so we need to mask part of it.
	MOVOU	(AX), X1
	PCMPEQB	X0, X1
	PMOVMSKB X1, DX
	// Apply mask
	ANDQ R10, DX
	POPCNTL DX, DX
	ADDQ DX, R12
end:
	MOVQ R12, (R8)
	RET

// handle for lengths < 16
small:
	TESTQ	BX, BX
	JEQ	endzero

	// Check if we'll load across a page boundary.
	LEAQ	16(SI), AX
	TESTW	$0xff0, AX
	JEQ	endofpage

	// We must ignore high bytes as they aren't part of our slice.
	// Create mask.
	MOVB BX, CX
	MOVQ $1, R10
	SALQ CL, R10
	SUBQ $1, R10

	// Load data
	MOVOU	(SI), X1
	// Compare target byte with each byte in data.
	PCMPEQB	X0, X1
	// Move result bits to integer register.
	PMOVMSKB X1, DX
	// Apply mask
	ANDQ R10, DX
	POPCNTL DX, DX
	// Directly return DX, we don't need to accumulate
	// since we have <16 bytes.
	MOVQ	DX, (R8)
	RET
endzero:
	MOVQ $0, (R8)
	RET

endofpage:
	// We must ignore low bytes as they aren't part of our slice.
	MOVQ $16,CX
	SUBQ BX, CX
	MOVQ $0xFFFF, R10
	SARQ CL, R10
	SALQ CL, R10

	// Load data into the high end of X1.
	MOVOU	-16(SI)(BX*1), X1
	// Compare target byte with each byte in data.
	PCMPEQB	X0, X1
	// Move result bits to integer register.
	PMOVMSKB X1, DX
	// Apply mask
	ANDQ R10, DX
	// Directly return DX, we don't need to accumulate
	// since we have <16 bytes.
	POPCNTL DX, DX
	MOVQ	DX, (R8)
	RET

avx2:
#ifndef hasAVX2
	CMPB   internal∕cpu·X86+const_offsetX86HasAVX2(SB), $1
	JNE sse
#endif
	MOVD AX, X0
	LEAQ -64(SI)(BX*1), R11
	LEAQ (SI)(BX*1), R13
	VPBROADCASTB  X0, Y1
	PCALIGN $32
avx2_loop:
	VMOVDQU (DI), Y2
	VMOVDQU 32(DI), Y4
	VPCMPEQB Y1, Y2, Y3
	VPCMPEQB Y1, Y4, Y5
	VPMOVMSKB Y3, DX
	VPMOVMSKB Y5, CX
	POPCNTL DX, DX
	POPCNTL CX, CX
	ADDQ DX, R12
	ADDQ CX, R12
	ADDQ $64, DI
	CMPQ DI, R11
	JLE avx2_loop

	// If last block is already processed,
	// skip to the end.
	//
	// This check is NOT an optimization; if the input length is a
	// multiple of 64, we must not go through the last leg of the
	// function because the bit shift count passed to SALQ below would
	// be 64, which is outside of the 0-63 range supported by those
	// instructions.
	//
	// Tests in the bytes and strings packages with input lengths that
	// are multiples of 64 will break if this condition were removed.
	CMPQ DI, R13
	JEQ endavx

	// Load address of the last 64 bytes.
	// There is an overlap with the previous block.
	MOVQ R11, DI
	VMOVDQU (DI), Y2
	VMOVDQU 32(DI), Y4
	VPCMPEQB Y1, Y2, Y3
	VPCMPEQB Y1, Y4, Y5
	VPMOVMSKB Y3, DX
	VPMOVMSKB Y5, CX
	// Exit AVX mode.
	VZEROUPPER
	SALQ $32, CX
	ORQ CX, DX

	// Create mask to ignore overlap between previous 64 byte block
	// and the next.
	ANDQ $63, BX
	MOVQ $64, CX
	SUBQ BX, CX
	MOVQ $0xFFFFFFFFFFFFFFFF, R10
	SALQ CL, R10
	// Apply mask
	ANDQ R10, DX
	POPCNTQ DX, DX
	ADDQ DX, R12
	MOVQ R12, (R8)
	RET
endavx:
	// Exit AVX mode.
	VZEROUPPER
	MOVQ R12, (R8)
	RET
