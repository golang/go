// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !plan9

#include "go_asm.h"
#include "textflag.h"

// NOTE: Windows externalthreadhandler expects memclr to preserve DX.

// func memclrNoHeapPointers(ptr unsafe.Pointer, n uintptr)
TEXT runtime·memclrNoHeapPointers(SB), NOSPLIT, $0-16
	MOVQ	ptr+0(FP), DI
	MOVQ	n+8(FP), BX
	XORQ	AX, AX

	// MOVOU seems always faster than REP STOSQ.
tail:
	// BSR+branch table make almost all memmove/memclr benchmarks worse. Not worth doing.
	TESTQ	BX, BX
	JEQ	_0
	CMPQ	BX, $2
	JBE	_1or2
	CMPQ	BX, $4
	JBE	_3or4
	CMPQ	BX, $8
	JB	_5through7
	JE	_8
	CMPQ	BX, $16
	JBE	_9through16
	PXOR	X0, X0
	CMPQ	BX, $32
	JBE	_17through32
	CMPQ	BX, $64
	JBE	_33through64
	CMPQ	BX, $128
	JBE	_65through128
	CMPQ	BX, $256
	JBE	_129through256
	CMPB	internal∕cpu·X86+const_offsetX86HasAVX2(SB), $1
	JE loop_preheader_avx2
	// TODO: for really big clears, use MOVNTDQ, even without AVX2.

loop:
	MOVOU	X0, 0(DI)
	MOVOU	X0, 16(DI)
	MOVOU	X0, 32(DI)
	MOVOU	X0, 48(DI)
	MOVOU	X0, 64(DI)
	MOVOU	X0, 80(DI)
	MOVOU	X0, 96(DI)
	MOVOU	X0, 112(DI)
	MOVOU	X0, 128(DI)
	MOVOU	X0, 144(DI)
	MOVOU	X0, 160(DI)
	MOVOU	X0, 176(DI)
	MOVOU	X0, 192(DI)
	MOVOU	X0, 208(DI)
	MOVOU	X0, 224(DI)
	MOVOU	X0, 240(DI)
	SUBQ	$256, BX
	ADDQ	$256, DI
	CMPQ	BX, $256
	JAE	loop
	JMP	tail

loop_preheader_avx2:
	VPXOR Y0, Y0, Y0
	// For smaller sizes MOVNTDQ may be faster or slower depending on hardware.
	// For larger sizes it is always faster, even on dual Xeons with 30M cache.
	// TODO take into account actual LLC size. E. g. glibc uses LLC size/2.
	CMPQ    BX, $0x2000000
	JAE     loop_preheader_avx2_huge
loop_avx2:
	VMOVDQU	Y0, 0(DI)
	VMOVDQU	Y0, 32(DI)
	VMOVDQU	Y0, 64(DI)
	VMOVDQU	Y0, 96(DI)
	SUBQ	$128, BX
	ADDQ	$128, DI
	CMPQ	BX, $128
	JAE	loop_avx2
	VMOVDQU  Y0, -32(DI)(BX*1)
	VMOVDQU  Y0, -64(DI)(BX*1)
	VMOVDQU  Y0, -96(DI)(BX*1)
	VMOVDQU  Y0, -128(DI)(BX*1)
	VZEROUPPER
	RET
loop_preheader_avx2_huge:
	// Align to 32 byte boundary
	VMOVDQU  Y0, 0(DI)
	MOVQ	DI, SI
	ADDQ	$32, DI
	ANDQ	$~31, DI
	SUBQ	DI, SI
	ADDQ	SI, BX
loop_avx2_huge:
	VMOVNTDQ	Y0, 0(DI)
	VMOVNTDQ	Y0, 32(DI)
	VMOVNTDQ	Y0, 64(DI)
	VMOVNTDQ	Y0, 96(DI)
	SUBQ	$128, BX
	ADDQ	$128, DI
	CMPQ	BX, $128
	JAE	loop_avx2_huge
	// In the description of MOVNTDQ in [1]
	// "... fencing operation implemented with the SFENCE or MFENCE instruction
	// should be used in conjunction with MOVNTDQ instructions..."
	// [1] 64-ia-32-architectures-software-developer-manual-325462.pdf
	SFENCE
	VMOVDQU  Y0, -32(DI)(BX*1)
	VMOVDQU  Y0, -64(DI)(BX*1)
	VMOVDQU  Y0, -96(DI)(BX*1)
	VMOVDQU  Y0, -128(DI)(BX*1)
	VZEROUPPER
	RET

_1or2:
	MOVB	AX, (DI)
	MOVB	AX, -1(DI)(BX*1)
	RET
_0:
	RET
_3or4:
	MOVW	AX, (DI)
	MOVW	AX, -2(DI)(BX*1)
	RET
_5through7:
	MOVL	AX, (DI)
	MOVL	AX, -4(DI)(BX*1)
	RET
_8:
	// We need a separate case for 8 to make sure we clear pointers atomically.
	MOVQ	AX, (DI)
	RET
_9through16:
	MOVQ	AX, (DI)
	MOVQ	AX, -8(DI)(BX*1)
	RET
_17through32:
	MOVOU	X0, (DI)
	MOVOU	X0, -16(DI)(BX*1)
	RET
_33through64:
	MOVOU	X0, (DI)
	MOVOU	X0, 16(DI)
	MOVOU	X0, -32(DI)(BX*1)
	MOVOU	X0, -16(DI)(BX*1)
	RET
_65through128:
	MOVOU	X0, (DI)
	MOVOU	X0, 16(DI)
	MOVOU	X0, 32(DI)
	MOVOU	X0, 48(DI)
	MOVOU	X0, -64(DI)(BX*1)
	MOVOU	X0, -48(DI)(BX*1)
	MOVOU	X0, -32(DI)(BX*1)
	MOVOU	X0, -16(DI)(BX*1)
	RET
_129through256:
	MOVOU	X0, (DI)
	MOVOU	X0, 16(DI)
	MOVOU	X0, 32(DI)
	MOVOU	X0, 48(DI)
	MOVOU	X0, 64(DI)
	MOVOU	X0, 80(DI)
	MOVOU	X0, 96(DI)
	MOVOU	X0, 112(DI)
	MOVOU	X0, -128(DI)(BX*1)
	MOVOU	X0, -112(DI)(BX*1)
	MOVOU	X0, -96(DI)(BX*1)
	MOVOU	X0, -80(DI)(BX*1)
	MOVOU	X0, -64(DI)(BX*1)
	MOVOU	X0, -48(DI)(BX*1)
	MOVOU	X0, -32(DI)(BX*1)
	MOVOU	X0, -16(DI)(BX*1)
	RET
