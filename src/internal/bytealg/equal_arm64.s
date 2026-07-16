// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

// memequal_varlen(a, b unsafe.Pointer) bool
TEXT runtime·memequal_varlen<ABIInternal>(SB),NOSPLIT,$0-17
	MOVD	8(R26), R2    // compiler stores size at offset 8 in the closure
	CBZ	R2, eq
	B	runtime·memequal<ABIInternal>(SB)
eq:
	MOVD	$1, R0
	RET

// input:
// R0: pointer a
// R1: pointer b
// R2: data len
// at return: result in R0
// memequal(a, b unsafe.Pointer, size uintptr) bool
TEXT runtime·memequal<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-25
	// short path to handle 0-byte case
	CBZ     R2, equal
	// short path to handle equal pointers
	CMP     R0, R1
	BEQ     equal
	CMP	$1, R2
	// handle 1-byte special case for better performance
	BEQ	one
	CMP	$16, R2
	// handle specially if length < 16
	BLO	tail
	CMP	$33, R2
	BHS	large
pairwise_16_32:
	// use pairwise loads for 16 <= len <= 32
	LDP	(R0), (R16, R17)
	LDP	(R1), (R24, R26)
	CMP	R16, R24
	CCMP	EQ, R17, R26, $0
	BNE	not_equal
	SUB	$16, R2, R16
	CBZ	R16, equal
	ADD	R0, R16, R24
	ADD	R1, R16, R25
	LDP	(R24), (R16, R17)
	LDP	(R25), (R24, R26)
	CMP	R16, R24
	CCMP	EQ, R17, R26, $0
	CSET	EQ, R0
	RET
	PCALIGN	$16
tail:
	// special compare of tail with length < 16
	TBZ	$3, R2, lt_8
	MOVD	(R0), R16
	MOVD	(R1), R17
	CMP	R16, R17
	BNE	not_equal
	SUB	$8, R2, R26	// offset of the last 8 bytes
	MOVD	(R0)(R26), R16
	MOVD	(R1)(R26), R17
	CMP	R16, R17
	CSET	EQ, R0
	RET
	PCALIGN	$16
lt_8:
	TBZ	$2, R2, lt_4
	MOVWU	(R0), R16
	MOVWU	(R1), R17
	CMP	R16, R17
	BNE	not_equal
	SUB	$4, R2, R26	// offset of the last 4 bytes
	MOVWU	(R0)(R26), R16
	MOVWU	(R1)(R26), R17
	CMP	R16, R17
	CSET	EQ, R0
	RET
	PCALIGN	$16
lt_4:
	TBZ	$1, R2, lt_2
	MOVHU.P	2(R0), R16
	MOVHU.P	2(R1), R17
	CMP	R16, R17
	BNE	not_equal
lt_2:
	TBZ	$0, R2, equal
one:
	MOVBU	(R0), R16
	MOVBU	(R1), R17
	CMP	R16, R17
	BNE	not_equal
equal:
	MOVD	$1, R0
	RET
not_equal:
	MOVB	ZR, R0
	RET
large:
	BIC	$0x3f, R2, R26
	CBZ	R26, remainder_33_64
	// work with 64-byte chunks
	ADD	R0, R26		// end of chunks
chunk64_loop:
	VLD1.P	(R0), [V21.D2, V22.D2, V23.D2, V24.D2]
	VLD1.P	(R1), [V25.D2, V26.D2, V27.D2, V28.D2]
	VCMEQ	V21.D2, V25.D2, V21.D2
	VCMEQ	V22.D2, V26.D2, V22.D2
	VCMEQ	V23.D2, V27.D2, V23.D2
	VCMEQ	V24.D2, V28.D2, V24.D2
	VAND	V21.B16, V22.B16, V21.B16
	VAND	V23.B16, V24.B16, V23.B16
	VAND	V21.B16, V23.B16, V21.B16
	CMP	R0, R26
	VMOV	V21.D[0], R16
	VMOV	V21.D[1], R17
	CBZ	R16, not_equal
	CBZ	R17, not_equal
	BNE	chunk64_loop
	AND	$0x3f, R2, R2
	CBZ	R2, equal
	CMP	$16, R2
	BLO	tail
	CMP	$33, R2
	BLO	pairwise_16_32
remainder_33_64:
	// 33 <= len < 64
	VLD1	(R0), [V21.D2, V22.D2]
	VLD1	(R1), [V23.D2, V24.D2]
	SUB	$32, R2, R26
	ADD	R0, R26, R24
	ADD	R1, R26, R25
	VEOR	V23.B16, V21.B16, V21.B16
	VEOR	V24.B16, V22.B16, V22.B16
	VORR	V22.B16, V21.B16, V21.B16
	VMOV	V21.D[0], R16
	VMOV	V21.D[1], R17
	ORR	R17, R16, R16
	CBNZ	R16, not_equal
	VLD1	(R24), [V21.D2, V22.D2]
	VLD1	(R25), [V23.D2, V24.D2]
	VEOR	V23.B16, V21.B16, V21.B16
	VEOR	V24.B16, V22.B16, V22.B16
	VORR	V22.B16, V21.B16, V21.B16
	VMOV	V21.D[0], R16
	VMOV	V21.D[1], R17
	ORR	R17, R16, R16
	CBNZ	R16, not_equal
	B	equal
