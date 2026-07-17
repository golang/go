// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

TEXT ·Compare<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-56
	// R0 = a_base (want in R0)
	// R1 = a_len  (want in R1)
	// R2 = a_cap  (unused)
	// R3 = b_base (want in R2)
	// R4 = b_len  (want in R3)
	// R5 = b_cap  (unused)
	MOVD	R3, R2
	MOVD	R4, R3
	B	cmpbody<>(SB)

TEXT runtime·cmpstring<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-40
	// R0 = a_base
	// R1 = a_len
	// R2 = b_base
	// R3 = b_len
	B	cmpbody<>(SB)

// On entry:
// R0 points to the start of a
// R1 is the length of a
// R2 points to the start of b
// R3 is the length of b
//
// On exit:
// R0 is the result
// R4, R5, R6, R8, R9, R10 and V0-V11 are clobbered
TEXT cmpbody<>(SB),NOSPLIT|NOFRAME,$0-0
	CMP	R0, R2
	BEQ	samebytes         // same starting pointers; compare lengths
	CMP	R1, R3
	CSEL	LT, R3, R1, R6    // R6 is min(R1, R3)

	CBZ	R6, samebytes
	BIC	$0x3f, R6, R10
	CBZ	R10, less_than_64 // length < 64, use scalar path
	ADD	R0, R10           // end of 64-byte chunks

	// Process 64 bytes per iteration using NEON.
	// Compare 4x16-byte chunks and reduce the result to a single byte.
	// On mismatch, rewind and fall back to the scalar path.
	PCALIGN	$16
chunk64_loop:
	VLD1.P	(R0), [V0.D2, V1.D2, V2.D2, V3.D2]
	VLD1.P	(R2), [V4.D2, V5.D2, V6.D2, V7.D2]
	VCMEQ	V0.B16, V4.B16, V8.B16
	VCMEQ	V1.B16, V5.B16, V9.B16
	VCMEQ	V2.B16, V6.B16, V10.B16
	VCMEQ	V3.B16, V7.B16, V11.B16
	VAND	V8.B16, V9.B16, V8.B16
	VAND	V10.B16, V11.B16, V10.B16
	VAND	V8.B16, V10.B16, V8.B16
	VUMINV	V8.B16, V8
	VMOV	V8.B[0], R4
	CBZ	R4, neon_mismatch
	CMP	R10, R0
	BNE	chunk64_loop

	AND	$0x3f, R6, R6     // remaining 0-63 bytes
	CBZ	R6, samebytes

less_than_64:
	// Scalar 16-byte loop for up to 63 bytes, then byte-level tail.
	BIC	$0xf, R6, R10
	CBZ	R10, small        // length < 16
	ADD	R0, R10
	PCALIGN	$16
chunk16_loop:
	LDP.P	16(R0), (R4, R8)
	LDP.P	16(R2), (R5, R9)
	CMP	R4, R5
	BNE	cmp
	CMP	R8, R9
	BNE	cmpnext
	CMP	R10, R0
	BNE	chunk16_loop
	AND	$0xf, R6, R6
	CBZ	R6, samebytes
	SUBS	$8, R6
	BLT	tail
	// the length of tail > 8 bytes
	MOVD.P	8(R0), R4
	MOVD.P	8(R2), R5
	CMP	R4, R5
	BNE	cmp
	SUB	$8, R6
	// compare last 8 bytes
	// tail always reads the final 8-byte window at R0+R6. R6 may be
	// negative here, overlapping already-verified bytes, which is harmless.
tail:
	MOVD	(R0)(R6), R4
	MOVD	(R2)(R6), R5
	CMP	R4, R5
	BEQ	samebytes
cmp:
	REV	R4, R4
	REV	R5, R5
	CMP	R4, R5
ret:
	MOVD	$1, R0
	CNEG	HI, R0, R0
	RET

neon_mismatch:
	// A mismatch was found in the last 64-byte NEON chunk. Rewind both
	// pointers and let the scalar path locate the first differing byte.
	SUB	$64, R0
	SUB	$64, R2
	MOVD	$64, R6
	B	less_than_64

small:
	TBZ	$3, R6, lt_8
	MOVD	(R0), R4
	MOVD	(R2), R5
	CMP	R4, R5
	BNE	cmp
	SUBS	$8, R6
	BEQ	samebytes
	B	tail
lt_8:
	TBZ	$2, R6, lt_4
	MOVWU	(R0), R4
	MOVWU	(R2), R5
	CMPW	R4, R5
	BNE	cmp
	SUBS	$4, R6
	BEQ	samebytes
	ADD	$4, R0
	ADD	$4, R2
lt_4:
	TBZ	$1, R6, lt_2
	MOVHU	(R0), R4
	MOVHU	(R2), R5
	CMPW	R4, R5
	BNE	cmp
	ADD	$2, R0
	ADD	$2, R2
lt_2:
	TBZ	$0, R6, samebytes
one:
	MOVBU	(R0), R4
	MOVBU	(R2), R5
	CMPW	R4, R5
	BNE	ret
samebytes:
	CMP	R3, R1
	CSET	NE, R0
	CNEG	LO, R0, R0
	RET
cmpnext:
	REV	R8, R4
	REV	R9, R5
	CMP	R4, R5
	B	ret
