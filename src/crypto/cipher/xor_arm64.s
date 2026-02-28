// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func xorBytesARM64(dst, a, b *byte, n int)
TEXT ·xorBytesARM64(SB), NOSPLIT|NOFRAME, $0
	MOVD	dst+0(FP), R0
	MOVD	a+8(FP), R1
	MOVD	b+16(FP), R2
	MOVD	n+24(FP), R3
	CMP	$64, R3
	BLT	tail
loop_64:
	VLD1.P	64(R1), [V0.B16, V1.B16, V2.B16, V3.B16]
	VLD1.P	64(R2), [V4.B16, V5.B16, V6.B16, V7.B16]
	VEOR	V0.B16, V4.B16, V4.B16
	VEOR	V1.B16, V5.B16, V5.B16
	VEOR	V2.B16, V6.B16, V6.B16
	VEOR	V3.B16, V7.B16, V7.B16
	VST1.P	[V4.B16, V5.B16, V6.B16, V7.B16], 64(R0)
	SUBS	$64, R3
	CMP	$64, R3
	BGE	loop_64
tail:
	// quick end
	CBZ	R3, end
	TBZ	$5, R3, less_than32
	VLD1.P	32(R1), [V0.B16, V1.B16]
	VLD1.P	32(R2), [V2.B16, V3.B16]
	VEOR	V0.B16, V2.B16, V2.B16
	VEOR	V1.B16, V3.B16, V3.B16
	VST1.P	[V2.B16, V3.B16], 32(R0)
less_than32:
	TBZ	$4, R3, less_than16
	LDP.P	16(R1), (R11, R12)
	LDP.P	16(R2), (R13, R14)
	EOR	R11, R13, R13
	EOR	R12, R14, R14
	STP.P	(R13, R14), 16(R0)
less_than16:
	TBZ	$3, R3, less_than8
	MOVD.P	8(R1), R11
	MOVD.P	8(R2), R12
	EOR	R11, R12, R12
	MOVD.P	R12, 8(R0)
less_than8:
	TBZ	$2, R3, less_than4
	MOVWU.P	4(R1), R13
	MOVWU.P	4(R2), R14
	EORW	R13, R14, R14
	MOVWU.P	R14, 4(R0)
less_than4:
	TBZ	$1, R3, less_than2
	MOVHU.P	2(R1), R15
	MOVHU.P	2(R2), R16
	EORW	R15, R16, R16
	MOVHU.P	R16, 2(R0)
less_than2:
	TBZ	$0, R3, end
	MOVBU	(R1), R17
	MOVBU	(R2), R19
	EORW	R17, R19, R19
	MOVBU	R19, (R0)
end:
	RET
