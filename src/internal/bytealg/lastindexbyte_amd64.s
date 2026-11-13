// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !plan9

#include "go_asm.h"
#include "textflag.h"

TEXT	·LastIndexByte(SB), NOSPLIT, $0-40
	MOVQ b_base+0(FP), SI
	MOVQ b_len+8(FP), BX
	MOVB c+24(FP), AL
	LEAQ ret+32(FP), R8
	JMP  lastindexbytebody<>(SB)

TEXT	·LastIndexByteString(SB), NOSPLIT, $0-32
	MOVQ s_base+0(FP), SI
	MOVQ s_len+8(FP), BX
	MOVB c+16(FP), AL
	LEAQ ret+24(FP), R8
	JMP  lastindexbytebody<>(SB)

// input:
//   SI: data
//   BX: data len
//   AL: byte sought
//   R8: address to put result
TEXT	lastindexbytebody<>(SB), NOSPLIT, $0
	// Shuffle X0 around so that each byte contains
	// the character we're looking for.
	MOVD AX, X0
	PUNPCKLBW X0, X0
	PUNPCKLBW X0, X0
	PSHUFL $0, X0, X0

	CMPQ BX, $16
	JLT small

	CMPQ BX, $32
	JA avx2
sse:
	LEAQ	-16(SI)(BX*1), DI	// DI = address of last (rightmost) 16-byte chunk
	PCALIGN $16
sseloop:
	// Move the 16-byte chunk of data at DI into X1.
	MOVOU	(DI), X1
	// Compare bytes in X0 to X1.
	PCMPEQB	X0, X1
	// Take the top bit of each byte in X1 and put the result in DX.
	PMOVMSKB X1, DX
	// Find last set bit, if any.
	BSRL	DX, DX
	JNZ	ssesuccess
	// Move to the chunk to the left.
	SUBQ	$16, DI
	CMPQ	DI, SI
	JA	sseloop

	// Search the leftmost 16-byte chunk, starting exactly at SI. This
	// chunk may overlap with the chunks we've already searched, but
	// that's ok.
	MOVQ	SI, DI
	MOVOU	(SI), X1
	PCMPEQB	X0, X1
	PMOVMSKB X1, DX
	BSRL	DX, DX
	JZ	failure

// We've found a chunk containing the byte.
// The chunk was loaded from DI.
// The index of the matching byte in the chunk is DX.
// The start of the data is SI.
ssesuccess:
	SUBQ SI, DI	// Compute offset of chunk within data.
	ADDQ DX, DI	// Add offset of byte within chunk.
	MOVQ DI, (R8)
	RET

failure:
	MOVQ $-1, (R8)
	RET

// handle for lengths < 16
small:
	TESTQ	BX, BX
	JEQ	failure

	// Check if we'll load across a page boundary.
	LEAQ	16(SI), AX
	TESTW	$0xff0, AX
	JEQ	endofpage

	MOVOU	(SI), X1 // Load data
	PCMPEQB	X0, X1	// Compare target byte with each byte in data.
	PMOVMSKB X1, DX	// Move result bits to integer register.
	// Bits past the end of the slice refer to memory outside the slice that
	// happens to be on the same page. Discard them by shifting them out of
	// the register and undo the shift once the highest valid set bit is
	// found.
	MOVL	$32, CX
	SUBL	BX, CX
	SHLL	CX, DX
	BSRL	DX, DX	// Find last set bit.
	JZ	failure	// No set bit, failure.
	SUBL	CX, DX
	MOVQ	DX, (R8)
	RET

endofpage:
	MOVOU	-16(SI)(BX*1), X1	// Load data into the high end of X1.
	PCMPEQB	X0, X1	// Compare target byte with each byte in data.
	PMOVMSKB X1, DX	// Move result bits to integer register.
	MOVL	BX, CX
	SHLL	CX, DX
	SHRL	$16, DX	// Shift desired bits down to bottom of register.
	BSRL	DX, DX	// Find last set bit.
	JZ	failure	// No set bit, failure.
	MOVQ	DX, (R8)
	RET

avx2:
#ifndef hasAVX2
	CMPB   internal∕cpu·X86+const_offsetX86HasAVX2(SB), $1
	JNE sse
#endif
	LEAQ -32(SI)(BX*1), DI	// DI = address of last (rightmost) 32-byte chunk
	VPBROADCASTB  X0, Y1

	PCALIGN $32
avx2_loop:
	VMOVDQU (DI), Y2
	VPCMPEQB Y1, Y2, Y3
	VPTEST Y3, Y3
	JNZ avx2success
	SUBQ $32, DI
	CMPQ DI, SI
	JA avx2_loop
	MOVQ SI, DI
	VMOVDQU (SI), Y2
	VPCMPEQB Y1, Y2, Y3
	VPTEST Y3, Y3
	JZ avx2failure
	JMP avx2success

avx2failure:
	VZEROUPPER
	MOVQ $-1, (R8)
	RET

avx2success:
	VPMOVMSKB Y3, DX
	BSRL DX, DX
	SUBQ SI, DI
	ADDQ DI, DX
	MOVQ DX, (R8)
	VZEROUPPER
	RET
