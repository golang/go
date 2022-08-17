// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func xorBytesARM32(dst, a, b *byte, n int)
TEXT ·xorBytesARM32(SB), NOSPLIT|NOFRAME, $0
	MOVW	dst+0(FP), R0
	MOVW	a+4(FP), R1
	MOVW	b+8(FP), R2
	MOVW	n+12(FP), R3
	CMP	$4, R3
	BLT	less_than4

loop_4:
	MOVW.P	4(R1), R4
	MOVW.P	4(R2), R5
	EOR	R4, R5, R5
	MOVW.P	R5, 4(R0)

	SUB	$4, R3
	CMP	$4, R3
	BGE	loop_4

less_than4:
	CMP	$2, R3
	BLT	less_than2
	MOVH.P	2(R1), R4
	MOVH.P	2(R2), R5
	EOR	R4, R5, R5
	MOVH.P	R5, 2(R0)

	SUB	$2, R3

less_than2:
	CMP	$0, R3
	BEQ	end
	MOVB	(R1), R4
	MOVB	(R2), R5
	EOR	R4, R5, R5
	MOVB	R5, (R0)
end:
	RET

// func xorBytesNEON32(dst, a, b *byte, n int)
TEXT ·xorBytesNEON32(SB), NOSPLIT|NOFRAME, $0
	MOVW	dst+0(FP), R0
	MOVW	a+4(FP), R1
	MOVW	b+8(FP), R2
	MOVW	n+12(FP), R3
	CMP	$32, R3
	BLT	less_than32

loop_32:
	WORD	$0xF421020D // vld1.u8 {q0, q1}, [r1]!
	WORD	$0xF422420D // vld1.u8 {q2, q3}, [r2]!
	WORD	$0xF3004154 // veor q2, q0, q2
	WORD	$0xF3026156 // veor q3, q1, q3
	WORD	$0xF400420D // vst1.u8 {q2, q3}, [r0]!

	SUB	$32, R3
	CMP	$32, R3
	BGE	loop_32

less_than32:
	CMP	$16, R3
	BLT	less_than16
	WORD	$0xF4210A0D // vld1.u8 q0, [r1]!
	WORD	$0xF4222A0D // vld1.u8 q1, [r2]!
	WORD	$0xF3002152 // veor q1, q0, q1
	WORD	$0xF4002A0D // vst1.u8 {q1}, [r0]!

	SUB	$16, R3

less_than16:
	CMP	$8, R3
	BLT	less_than8
	WORD	$0xF421070D // vld1.u8 d0, [r1]!
	WORD	$0xF422170D // vld1.u8 d1, [r2]!
	WORD	$0xF3001111 // veor d1, d0, d1
	WORD	$0xF400170D // vst1.u8 {d1}, [r0]!

	SUB	$8, R3

less_than8:
	CMP	$4, R3
	BLT	less_than4
	MOVW.P	4(R1), R4
	MOVW.P	4(R2), R5
	EOR	R4, R5, R5
	MOVW.P	R5, 4(R0)

	SUB	$4, R3

less_than4:
	CMP	$2, R3
	BLT	less_than2
	MOVH.P	2(R1), R4
	MOVH.P	2(R2), R5
	EOR	R4, R5, R5
	MOVH.P	R5, 2(R0)

	SUB	$2, R3

less_than2:
	CMP	$0, R3
	BEQ	end
	MOVB	(R1), R4
	MOVB	(R2), R5
	EOR	R4, R5, R5
	MOVB	R5, (R0)
end:
	RET
