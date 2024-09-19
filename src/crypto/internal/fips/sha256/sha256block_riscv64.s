// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

#include "textflag.h"

// SHA256 block routine. See sha256block.go for Go equivalent.
//
// The algorithm is detailed in FIPS 180-4:
//
//  https://csrc.nist.gov/publications/fips/fips180-4/fips-180-4.pdf
//
// Wt = Mt; for 0 <= t <= 15
// Wt = SIGMA1(Wt-2) + SIGMA0(Wt-15) + Wt-16; for 16 <= t <= 63
//
// a = H0
// b = H1
// c = H2
// d = H3
// e = H4
// f = H5
// g = H6
// h = H7
//
// for t = 0 to 63 {
//    T1 = h + BIGSIGMA1(e) + Ch(e,f,g) + Kt + Wt
//    T2 = BIGSIGMA0(a) + Maj(a,b,c)
//    h = g
//    g = f
//    f = e
//    e = d + T1
//    d = c
//    c = b
//    b = a
//    a = T1 + T2
// }
//
// H0 = a + H0
// H1 = b + H1
// H2 = c + H2
// H3 = d + H3
// H4 = e + H4
// H5 = f + H5
// H6 = g + H6
// H7 = h + H7

// Wt = Mt; for 0 <= t <= 15
#define MSGSCHEDULE0(index) \
	MOVBU	((index*4)+0)(X29), X5; \
	MOVBU	((index*4)+1)(X29), X6; \
	MOVBU	((index*4)+2)(X29), X7; \
	MOVBU	((index*4)+3)(X29), X8; \
	SLL	$24, X5; \
	SLL	$16, X6; \
	OR	X5, X6, X5; \
	SLL	$8, X7; \
	OR	X5, X7, X5; \
	OR	X5, X8, X5; \
	MOVW	X5, (index*4)(X19)

// Wt = SIGMA1(Wt-2) + Wt-7 + SIGMA0(Wt-15) + Wt-16; for 16 <= t <= 63
//   SIGMA0(x) = ROTR(7,x) XOR ROTR(18,x) XOR SHR(3,x)
//   SIGMA1(x) = ROTR(17,x) XOR ROTR(19,x) XOR SHR(10,x)
#define MSGSCHEDULE1(index) \
	MOVWU	(((index-2)&0xf)*4)(X19), X5; \
	MOVWU	(((index-15)&0xf)*4)(X19), X6; \
	MOVWU	(((index-7)&0xf)*4)(X19), X9; \
	MOVWU	(((index-16)&0xf)*4)(X19), X21; \
	RORW	$17, X5, X7; \
	RORW	$19, X5, X8; \
	SRL	$10, X5; \
	XOR	X7, X5; \
	XOR	X8, X5; \
	ADD	X9, X5; \
	RORW	$7, X6, X7; \
	RORW	$18, X6, X8; \
	SRL	$3, X6; \
	XOR	X7, X6; \
	XOR	X8, X6; \
	ADD	X6, X5; \
	ADD	X21, X5; \
	MOVW	X5, ((index&0xf)*4)(X19)

// Calculate T1 in X5.
// h is also used as an accumulator. Wt is passed in X5.
//   T1 = h + BIGSIGMA1(e) + Ch(e, f, g) + Kt + Wt
//     BIGSIGMA1(x) = ROTR(6,x) XOR ROTR(11,x) XOR ROTR(25,x)
//     Ch(x, y, z) = (x AND y) XOR (NOT x AND z)
#define SHA256T1(index, e, f, g, h) \
	MOVWU	(index*4)(X18), X8; \
	ADD	X5, h; \
	RORW	$6, e, X6; \
	ADD	X8, h; \
	RORW	$11, e, X7; \
	XOR	X7, X6; \
	RORW	$25, e, X8; \
	XOR	X8, X6; \
	ADD	X6, h; \
	AND	e, f, X5; \
	NOT	e, X7; \
	AND	g, X7; \
	XOR	X7, X5; \
	ADD	h, X5

// Calculate T2 in X6.
//   T2 = BIGSIGMA0(a) + Maj(a, b, c)
//     BIGSIGMA0(x) = ROTR(2,x) XOR ROTR(13,x) XOR ROTR(22,x)
//     Maj(x, y, z) = (x AND y) XOR (x AND z) XOR (y AND z)
#define SHA256T2(a, b, c) \
	RORW	$2, a, X6; \
	RORW	$13, a, X7; \
	XOR	X7, X6; \
	RORW	$22, a, X8; \
	XOR	X8, X6; \
	AND	a, b, X7; \
	AND	a, c, X8; \
	XOR	X8, X7; \
	AND	b, c, X9; \
	XOR	X9, X7; \
	ADD	X7, X6

// Calculate T1 and T2, then e = d + T1 and a = T1 + T2.
// The values for e and a are stored in d and h, ready for rotation.
#define SHA256ROUND(index, a, b, c, d, e, f, g, h) \
	SHA256T1(index, e, f, g, h); \
	SHA256T2(a, b, c); \
	MOV	X6, h; \
	ADD	X5, d; \
	ADD	X5, h

#define SHA256ROUND0(index, a, b, c, d, e, f, g, h) \
	MSGSCHEDULE0(index); \
	SHA256ROUND(index, a, b, c, d, e, f, g, h)

#define SHA256ROUND1(index, a, b, c, d, e, f, g, h) \
	MSGSCHEDULE1(index); \
	SHA256ROUND(index, a, b, c, d, e, f, g, h)

// Note that 64 bytes of stack space is used as a circular buffer
// for the message schedule (4 bytes * 16 entries).
//
// func block(dig *Digest, p []byte)
TEXT ·block(SB),0,$64-32
	MOV	p_base+8(FP), X29
	MOV	p_len+16(FP), X30
	SRL	$6, X30
	SLL	$6, X30

	ADD	X29, X30, X28
	BEQ	X28, X29, end

	MOV	·_K(SB), X18		// const table
	ADD	$8, X2, X19		// message schedule

	MOV	dig+0(FP), X20
	MOVWU	(0*4)(X20), X10		// a = H0
	MOVWU	(1*4)(X20), X11		// b = H1
	MOVWU	(2*4)(X20), X12		// c = H2
	MOVWU	(3*4)(X20), X13		// d = H3
	MOVWU	(4*4)(X20), X14		// e = H4
	MOVWU	(5*4)(X20), X15		// f = H5
	MOVWU	(6*4)(X20), X16		// g = H6
	MOVWU	(7*4)(X20), X17		// h = H7

loop:
	SHA256ROUND0(0, X10, X11, X12, X13, X14, X15, X16, X17)
	SHA256ROUND0(1, X17, X10, X11, X12, X13, X14, X15, X16)
	SHA256ROUND0(2, X16, X17, X10, X11, X12, X13, X14, X15)
	SHA256ROUND0(3, X15, X16, X17, X10, X11, X12, X13, X14)
	SHA256ROUND0(4, X14, X15, X16, X17, X10, X11, X12, X13)
	SHA256ROUND0(5, X13, X14, X15, X16, X17, X10, X11, X12)
	SHA256ROUND0(6, X12, X13, X14, X15, X16, X17, X10, X11)
	SHA256ROUND0(7, X11, X12, X13, X14, X15, X16, X17, X10)
	SHA256ROUND0(8, X10, X11, X12, X13, X14, X15, X16, X17)
	SHA256ROUND0(9, X17, X10, X11, X12, X13, X14, X15, X16)
	SHA256ROUND0(10, X16, X17, X10, X11, X12, X13, X14, X15)
	SHA256ROUND0(11, X15, X16, X17, X10, X11, X12, X13, X14)
	SHA256ROUND0(12, X14, X15, X16, X17, X10, X11, X12, X13)
	SHA256ROUND0(13, X13, X14, X15, X16, X17, X10, X11, X12)
	SHA256ROUND0(14, X12, X13, X14, X15, X16, X17, X10, X11)
	SHA256ROUND0(15, X11, X12, X13, X14, X15, X16, X17, X10)

	SHA256ROUND1(16, X10, X11, X12, X13, X14, X15, X16, X17)
	SHA256ROUND1(17, X17, X10, X11, X12, X13, X14, X15, X16)
	SHA256ROUND1(18, X16, X17, X10, X11, X12, X13, X14, X15)
	SHA256ROUND1(19, X15, X16, X17, X10, X11, X12, X13, X14)
	SHA256ROUND1(20, X14, X15, X16, X17, X10, X11, X12, X13)
	SHA256ROUND1(21, X13, X14, X15, X16, X17, X10, X11, X12)
	SHA256ROUND1(22, X12, X13, X14, X15, X16, X17, X10, X11)
	SHA256ROUND1(23, X11, X12, X13, X14, X15, X16, X17, X10)
	SHA256ROUND1(24, X10, X11, X12, X13, X14, X15, X16, X17)
	SHA256ROUND1(25, X17, X10, X11, X12, X13, X14, X15, X16)
	SHA256ROUND1(26, X16, X17, X10, X11, X12, X13, X14, X15)
	SHA256ROUND1(27, X15, X16, X17, X10, X11, X12, X13, X14)
	SHA256ROUND1(28, X14, X15, X16, X17, X10, X11, X12, X13)
	SHA256ROUND1(29, X13, X14, X15, X16, X17, X10, X11, X12)
	SHA256ROUND1(30, X12, X13, X14, X15, X16, X17, X10, X11)
	SHA256ROUND1(31, X11, X12, X13, X14, X15, X16, X17, X10)
	SHA256ROUND1(32, X10, X11, X12, X13, X14, X15, X16, X17)
	SHA256ROUND1(33, X17, X10, X11, X12, X13, X14, X15, X16)
	SHA256ROUND1(34, X16, X17, X10, X11, X12, X13, X14, X15)
	SHA256ROUND1(35, X15, X16, X17, X10, X11, X12, X13, X14)
	SHA256ROUND1(36, X14, X15, X16, X17, X10, X11, X12, X13)
	SHA256ROUND1(37, X13, X14, X15, X16, X17, X10, X11, X12)
	SHA256ROUND1(38, X12, X13, X14, X15, X16, X17, X10, X11)
	SHA256ROUND1(39, X11, X12, X13, X14, X15, X16, X17, X10)
	SHA256ROUND1(40, X10, X11, X12, X13, X14, X15, X16, X17)
	SHA256ROUND1(41, X17, X10, X11, X12, X13, X14, X15, X16)
	SHA256ROUND1(42, X16, X17, X10, X11, X12, X13, X14, X15)
	SHA256ROUND1(43, X15, X16, X17, X10, X11, X12, X13, X14)
	SHA256ROUND1(44, X14, X15, X16, X17, X10, X11, X12, X13)
	SHA256ROUND1(45, X13, X14, X15, X16, X17, X10, X11, X12)
	SHA256ROUND1(46, X12, X13, X14, X15, X16, X17, X10, X11)
	SHA256ROUND1(47, X11, X12, X13, X14, X15, X16, X17, X10)
	SHA256ROUND1(48, X10, X11, X12, X13, X14, X15, X16, X17)
	SHA256ROUND1(49, X17, X10, X11, X12, X13, X14, X15, X16)
	SHA256ROUND1(50, X16, X17, X10, X11, X12, X13, X14, X15)
	SHA256ROUND1(51, X15, X16, X17, X10, X11, X12, X13, X14)
	SHA256ROUND1(52, X14, X15, X16, X17, X10, X11, X12, X13)
	SHA256ROUND1(53, X13, X14, X15, X16, X17, X10, X11, X12)
	SHA256ROUND1(54, X12, X13, X14, X15, X16, X17, X10, X11)
	SHA256ROUND1(55, X11, X12, X13, X14, X15, X16, X17, X10)
	SHA256ROUND1(56, X10, X11, X12, X13, X14, X15, X16, X17)
	SHA256ROUND1(57, X17, X10, X11, X12, X13, X14, X15, X16)
	SHA256ROUND1(58, X16, X17, X10, X11, X12, X13, X14, X15)
	SHA256ROUND1(59, X15, X16, X17, X10, X11, X12, X13, X14)
	SHA256ROUND1(60, X14, X15, X16, X17, X10, X11, X12, X13)
	SHA256ROUND1(61, X13, X14, X15, X16, X17, X10, X11, X12)
	SHA256ROUND1(62, X12, X13, X14, X15, X16, X17, X10, X11)
	SHA256ROUND1(63, X11, X12, X13, X14, X15, X16, X17, X10)

	MOVWU	(0*4)(X20), X5
	MOVWU	(1*4)(X20), X6
	MOVWU	(2*4)(X20), X7
	MOVWU	(3*4)(X20), X8
	ADD	X5, X10		// H0 = a + H0
	ADD	X6, X11		// H1 = b + H1
	ADD	X7, X12		// H2 = c + H2
	ADD	X8, X13		// H3 = d + H3
	MOVW	X10, (0*4)(X20)
	MOVW	X11, (1*4)(X20)
	MOVW	X12, (2*4)(X20)
	MOVW	X13, (3*4)(X20)
	MOVWU	(4*4)(X20), X5
	MOVWU	(5*4)(X20), X6
	MOVWU	(6*4)(X20), X7
	MOVWU	(7*4)(X20), X8
	ADD	X5, X14		// H4 = e + H4
	ADD	X6, X15		// H5 = f + H5
	ADD	X7, X16		// H6 = g + H6
	ADD	X8, X17		// H7 = h + H7
	MOVW	X14, (4*4)(X20)
	MOVW	X15, (5*4)(X20)
	MOVW	X16, (6*4)(X20)
	MOVW	X17, (7*4)(X20)

	ADD	$64, X29
	BNE	X28, X29, loop

end:
	RET
