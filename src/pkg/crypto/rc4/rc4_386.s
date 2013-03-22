// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// func xorKeyStream(dst, src *byte, n int, state *[256]byte, i, j *uint8)
TEXT ·xorKeyStream(SB),7,$0
	MOVL dst+0(FP), DI
	MOVL src+4(FP), SI
	MOVL state+12(FP), BP

	MOVL i+16(FP), AX
	MOVBLZX (AX), AX
	MOVL j+20(FP), BX
	MOVBLZX (BX), BX
	CMPL n+8(FP), $0
	JEQ done

loop:
	// i += 1
	INCB AX

	// j += c.s[i]
	MOVBLZX (BP)(AX*4), DX
	ADDB DX, BX
	MOVBLZX BX, BX

	// c.s[i], c.s[j] = c.s[j], c.s[i]
	MOVBLZX (BP)(BX*4), CX
	MOVB CX, (BP)(AX*4)
	MOVB DX, (BP)(BX*4)

	// *dst = *src ^ c.s[c.s[i]+c.s[j]]
	ADDB DX, CX
	MOVBLZX CX, CX
	MOVB (BP)(CX*4), CX
	XORB (SI), CX
	MOVBLZX CX, CX
	MOVB CX, (DI)

	INCL SI
	INCL DI
	DECL n+8(FP)
	JNE loop

done:
	MOVL i+16(FP), CX
	MOVB AX, (CX)
	MOVL j+20(FP), CX
	MOVB BX, (CX)

	RET
