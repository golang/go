// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// Register use in every kernel:
//   SI: p, start of the input
//   BX: n, input length; n >= 16 for SSE2 and n >= 32 for AVX2
//   DI: start of the current block
//   AX: start of the last block, p+n-width
//   DX: hit mask of the current block, bit i for byte DI+i
//   X0/Y0: current block; X1/Y1 to X3/Y3: scratch
//   X4/Y4: 0x20; X5/Y5: '"'; X6/Y6: '\\'; X7/Y7: '<'; X8/Y8: '>'; X9/Y9: '&'
// Loop invariant: no byte in [SI, DI) is special and DI < AX, so the block
// load stays inside [SI, SI+BX). The last block is loaded from AX, which may
// overlap bytes already found clean but ends exactly at SI+BX.
// A signed compare 0x20 > c holds for c < 0x20 and for c >= 0x80.

#define SPLAT_SSE2(imm, x) \
	MOVQ	$imm, CX; \
	MOVQ	CX, x; \
	PUNPCKLQDQ	x, x

#define SPLAT_AVX2(imm, x, y) \
	MOVL	$imm, CX; \
	MOVQ	CX, x; \
	VPBROADCASTB	x, y

#define SPECIAL_SSE2() \
	MOVO	X4, X1; \
	PCMPGTB	X0, X1; \
	MOVO	X0, X2; \
	PCMPEQB	X5, X2; \
	POR	X2, X1; \
	PCMPEQB	X6, X0; \
	POR	X0, X1; \
	PMOVMSKB	X1, DX

#define HTML_SSE2() \
	MOVO	X4, X1; \
	PCMPGTB	X0, X1; \
	MOVO	X0, X2; \
	PCMPEQB	X5, X2; \
	POR	X2, X1; \
	MOVO	X0, X2; \
	PCMPEQB	X6, X2; \
	POR	X2, X1; \
	MOVO	X0, X2; \
	PCMPEQB	X7, X2; \
	POR	X2, X1; \
	MOVO	X0, X2; \
	PCMPEQB	X8, X2; \
	POR	X2, X1; \
	PCMPEQB	X9, X0; \
	POR	X0, X1; \
	PMOVMSKB	X1, DX

#define SPECIAL_AVX2() \
	VPCMPGTB	Y0, Y4, Y1; \
	VPCMPEQB	Y0, Y5, Y2; \
	VPCMPEQB	Y0, Y6, Y3; \
	VPOR	Y2, Y1, Y1; \
	VPOR	Y3, Y1, Y1; \
	VPMOVMSKB	Y1, DX

#define HTML_AVX2() \
	VPCMPGTB	Y0, Y4, Y1; \
	VPCMPEQB	Y0, Y5, Y2; \
	VPCMPEQB	Y0, Y6, Y3; \
	VPOR	Y2, Y1, Y1; \
	VPOR	Y3, Y1, Y1; \
	VPCMPEQB	Y0, Y7, Y2; \
	VPCMPEQB	Y0, Y8, Y3; \
	VPOR	Y2, Y1, Y1; \
	VPOR	Y3, Y1, Y1; \
	VPCMPEQB	Y0, Y9, Y2; \
	VPOR	Y2, Y1, Y1; \
	VPMOVMSKB	Y1, DX

// func indexStringSpecialSSE2(p *byte, n int) int
TEXT ·indexStringSpecialSSE2(SB), NOSPLIT, $0-24
	MOVQ	p+0(FP), SI
	MOVQ	n+8(FP), BX
	SPLAT_SSE2(0x2020202020202020, X4)
	SPLAT_SSE2(0x2222222222222222, X5)
	SPLAT_SSE2(0x5c5c5c5c5c5c5c5c, X6)
	MOVQ	SI, DI
	LEAQ	-16(SI)(BX*1), AX
	JMP	special_sse2_next

	PCALIGN	$16
special_sse2_loop:
	MOVOU	(DI), X0
	SPECIAL_SSE2()
	TESTL	DX, DX
	JNZ	special_sse2_hit
	ADDQ	$16, DI

special_sse2_next:
	CMPQ	DI, AX
	JB	special_sse2_loop
	MOVQ	AX, DI
	MOVOU	(DI), X0
	SPECIAL_SSE2()
	TESTL	DX, DX
	JNZ	special_sse2_hit
	MOVQ	BX, ret+16(FP)
	RET

special_sse2_hit:
	BSFL	DX, DX
	SUBQ	SI, DI
	ADDQ	DX, DI
	MOVQ	DI, ret+16(FP)
	RET

// func indexStringSpecialAVX2(p *byte, n int) int
TEXT ·indexStringSpecialAVX2(SB), NOSPLIT, $0-24
	MOVQ	p+0(FP), SI
	MOVQ	n+8(FP), BX
	SPLAT_AVX2(0x20, X4, Y4)
	SPLAT_AVX2(0x22, X5, Y5)
	SPLAT_AVX2(0x5c, X6, Y6)
	MOVQ	SI, DI
	LEAQ	-32(SI)(BX*1), AX
	JMP	special_avx2_next

	PCALIGN	$16
special_avx2_loop:
	VMOVDQU	(DI), Y0
	SPECIAL_AVX2()
	TESTL	DX, DX
	JNZ	special_avx2_hit
	ADDQ	$32, DI

special_avx2_next:
	CMPQ	DI, AX
	JB	special_avx2_loop
	MOVQ	AX, DI
	VMOVDQU	(DI), Y0
	SPECIAL_AVX2()
	TESTL	DX, DX
	JNZ	special_avx2_hit
	VZEROUPPER
	MOVQ	BX, ret+16(FP)
	RET

special_avx2_hit:
	VZEROUPPER
	BSFL	DX, DX
	SUBQ	SI, DI
	ADDQ	DX, DI
	MOVQ	DI, ret+16(FP)
	RET

// func indexEscapeHTMLSSE2(p *byte, n int) int
TEXT ·indexEscapeHTMLSSE2(SB), NOSPLIT, $0-24
	MOVQ	p+0(FP), SI
	MOVQ	n+8(FP), BX
	SPLAT_SSE2(0x2020202020202020, X4)
	SPLAT_SSE2(0x2222222222222222, X5)
	SPLAT_SSE2(0x5c5c5c5c5c5c5c5c, X6)
	SPLAT_SSE2(0x3c3c3c3c3c3c3c3c, X7)
	SPLAT_SSE2(0x3e3e3e3e3e3e3e3e, X8)
	SPLAT_SSE2(0x2626262626262626, X9)
	MOVQ	SI, DI
	LEAQ	-16(SI)(BX*1), AX
	JMP	html_sse2_next

	PCALIGN	$16
html_sse2_loop:
	MOVOU	(DI), X0
	HTML_SSE2()
	TESTL	DX, DX
	JNZ	html_sse2_hit
	ADDQ	$16, DI

html_sse2_next:
	CMPQ	DI, AX
	JB	html_sse2_loop
	MOVQ	AX, DI
	MOVOU	(DI), X0
	HTML_SSE2()
	TESTL	DX, DX
	JNZ	html_sse2_hit
	MOVQ	BX, ret+16(FP)
	RET

html_sse2_hit:
	BSFL	DX, DX
	SUBQ	SI, DI
	ADDQ	DX, DI
	MOVQ	DI, ret+16(FP)
	RET

// func indexEscapeHTMLAVX2(p *byte, n int) int
TEXT ·indexEscapeHTMLAVX2(SB), NOSPLIT, $0-24
	MOVQ	p+0(FP), SI
	MOVQ	n+8(FP), BX
	SPLAT_AVX2(0x20, X4, Y4)
	SPLAT_AVX2(0x22, X5, Y5)
	SPLAT_AVX2(0x5c, X6, Y6)
	SPLAT_AVX2(0x3c, X7, Y7)
	SPLAT_AVX2(0x3e, X8, Y8)
	SPLAT_AVX2(0x26, X9, Y9)
	MOVQ	SI, DI
	LEAQ	-32(SI)(BX*1), AX
	JMP	html_avx2_next

	PCALIGN	$16
html_avx2_loop:
	VMOVDQU	(DI), Y0
	HTML_AVX2()
	TESTL	DX, DX
	JNZ	html_avx2_hit
	ADDQ	$32, DI

html_avx2_next:
	CMPQ	DI, AX
	JB	html_avx2_loop
	MOVQ	AX, DI
	VMOVDQU	(DI), Y0
	HTML_AVX2()
	TESTL	DX, DX
	JNZ	html_avx2_hit
	VZEROUPPER
	MOVQ	BX, ret+16(FP)
	RET

html_avx2_hit:
	VZEROUPPER
	BSFL	DX, DX
	SUBQ	SI, DI
	ADDQ	DX, DI
	MOVQ	DI, ret+16(FP)
	RET
