// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// func xorKeyStream(dst, src *byte, n int, state *[256]byte, i, j *uint8)
TEXT ·xorKeyStream(SB),7,$0
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX
	MOVQ state+24(FP), R8

	MOVQ xPtr+32(FP), AX
	MOVBQZX (AX), AX
	MOVQ yPtr+40(FP), BX
	MOVBQZX (BX), BX

loop:
	CMPQ CX, $0
	JE done

	// c.i += 1
	INCB AX

	// c.j += c.s[c.i]
	MOVB (R8)(AX*1), R9
	ADDB R9, BX

	MOVBQZX (R8)(BX*1), R10

	MOVB R10, (R8)(AX*1)
	MOVB R9, (R8)(BX*1)

	// R11 = c.s[c.i]+c.s[c.j]
	MOVQ R10, R11
	ADDB R9, R11

	MOVB (R8)(R11*1), R11
	MOVB (SI), R12
	XORB R11, R12
	MOVB R12, (DI)

	INCQ SI
	INCQ DI
	DECQ CX

	JMP loop
done:
	MOVQ xPtr+32(FP), R8
	MOVB AX, (R8)
	MOVQ yPtr+40(FP), R8
	MOVB BX, (R8)

	RET
