// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

#include "textflag.h"

// SHA512 block routine. See sha512block.go for Go equivalent.
//
// The algorithm is detailed in FIPS 180-4:
//
//  https://csrc.nist.gov/publications/fips/fips180-4/fips-180-4.pdf
//
// Wt = Mt; for 0 <= t <= 15
// Wt = SIGMA1(Wt-2) + SIGMA0(Wt-15) + Wt-16; for 16 <= t <= 79
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
// for t = 0 to 79 {
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
	MOVBU	((index*8)+0)(X29), X5; \
	MOVBU	((index*8)+1)(X29), X6; \
	MOVBU	((index*8)+2)(X29), X7; \
	MOVBU	((index*8)+3)(X29), X8; \
	SLL	$56, X5; \
	SLL	$48, X6; \
	OR	X5, X6, X5; \
	SLL	$40, X7; \
	OR	X5, X7, X5; \
	SLL	$32, X8; \
	OR	X5, X8, X5; \
	MOVBU	((index*8)+4)(X29), X9; \
	MOVBU	((index*8)+5)(X29), X6; \
	MOVBU	((index*8)+6)(X29), X7; \
	MOVBU	((index*8)+7)(X29), X8; \
	SLL	$24, X9; \
	OR	X5, X9, X5; \
	SLL	$16, X6; \
	OR	X5, X6, X5; \
	SLL	$8, X7; \
	OR	X5, X7, X5; \
	OR	X5, X8, X5; \
	MOV	X5, (index*8)(X19)

// Wt = SIGMA1(Wt-2) + Wt-7 + SIGMA0(Wt-15) + Wt-16; for 16 <= t <= 79
//   SIGMA0(x) = ROTR(1,x) XOR ROTR(8,x) XOR SHR(7,x)
//   SIGMA1(x) = ROTR(19,x) XOR ROTR(61,x) XOR SHR(6,x)
#define MSGSCHEDULE1(index) \
	MOV	(((index-2)&0xf)*8)(X19), X5; \
	MOV	(((index-15)&0xf)*8)(X19), X6; \
	MOV	(((index-7)&0xf)*8)(X19), X9; \
	MOV	(((index-16)&0xf)*8)(X19), X21; \
	ROR	$19, X5, X7; \
	ROR	$61, X5, X8; \
	SRL	$6, X5; \
	XOR	X7, X5; \
	XOR	X8, X5; \
	ADD	X9, X5; \
	ROR	$1, X6, X7; \
	ROR	$8, X6, X8; \
	SRL	$7, X6; \
	XOR	X7, X6; \
	XOR	X8, X6; \
	ADD	X6, X5; \
	ADD	X21, X5; \
	MOV	X5, ((index&0xf)*8)(X19)

// Calculate T1 in X5.
// h is also used as an accumulator. Wt is passed in X5.
//   T1 = h + BIGSIGMA1(e) + Ch(e, f, g) + Kt + Wt
//     BIGSIGMA1(x) = ROTR(14,x) XOR ROTR(18,x) XOR ROTR(41,x)
//     Ch(x, y, z) = (x AND y) XOR (NOT x AND z)
//                 = ((y XOR z) AND x) XOR z
#define SHA512T1(index, e, f, g, h) \
	MOV	(index*8)(X18), X8; \
	ADD	X5, h; \
	ROR	$14, e, X6; \
	ADD	X8, h; \
	ROR	$18, e, X7; \
	ROR	$41, e, X8; \
	XOR	X7, X6; \
	XOR	f, g, X5; \
	XOR	X8, X6; \
	AND	e, X5; \
	ADD	X6, h; \
	XOR	g, X5; \
	ADD	h, X5

// Calculate T2 in X6.
//   T2 = BIGSIGMA0(a) + Maj(a, b, c)
//     BIGSIGMA0(x) = ROTR(28,x) XOR ROTR(34,x) XOR ROTR(39,x)
//     Maj(x, y, z) = (x AND y) XOR (x AND z) XOR (y AND z)
//                  = ((y XOR z) AND x) XOR (y AND z)
#define SHA512T2(a, b, c) \
	ROR	$28, a, X6; \
	ROR	$34, a, X7; \
	ROR	$39, a, X8; \
	XOR	X7, X6; \
	XOR	b, c, X9; \
	AND	b, c, X7; \
	AND	a, X9; \
	XOR	X8, X6; \
	XOR	X7, X9; \
	ADD	X9, X6

// Calculate T1 and T2, then e = d + T1 and a = T1 + T2.
// The values for e and a are stored in d and h, ready for rotation.
#define SHA512ROUND(index, a, b, c, d, e, f, g, h) \
	SHA512T1(index, e, f, g, h); \
	SHA512T2(a, b, c); \
	MOV	X6, h; \
	ADD	X5, d; \
	ADD	X5, h

#define SHA512ROUND0(index, a, b, c, d, e, f, g, h) \
	MSGSCHEDULE0(index); \
	SHA512ROUND(index, a, b, c, d, e, f, g, h)

#define SHA512ROUND1(index, a, b, c, d, e, f, g, h) \
	MSGSCHEDULE1(index); \
	SHA512ROUND(index, a, b, c, d, e, f, g, h)

// func block(dig *Digest, p []byte)
TEXT ·block(SB),0,$128-32
	MOV	p_base+8(FP), X29
	MOV	p_len+16(FP), X30
	SRL	$7, X30
	SLL	$7, X30

	ADD	X29, X30, X28
	BEQ	X28, X29, end

	MOV	$·_K(SB), X18		// const table
	ADD	$8, X2, X19		// message schedule

	MOV	dig+0(FP), X20
	MOV	(0*8)(X20), X10		// a = H0
	MOV	(1*8)(X20), X11		// b = H1
	MOV	(2*8)(X20), X12		// c = H2
	MOV	(3*8)(X20), X13		// d = H3
	MOV	(4*8)(X20), X14		// e = H4
	MOV	(5*8)(X20), X15		// f = H5
	MOV	(6*8)(X20), X16		// g = H6
	MOV	(7*8)(X20), X17		// h = H7

loop:
	SHA512ROUND0(0, X10, X11, X12, X13, X14, X15, X16, X17)
	SHA512ROUND0(1, X17, X10, X11, X12, X13, X14, X15, X16)
	SHA512ROUND0(2, X16, X17, X10, X11, X12, X13, X14, X15)
	SHA512ROUND0(3, X15, X16, X17, X10, X11, X12, X13, X14)
	SHA512ROUND0(4, X14, X15, X16, X17, X10, X11, X12, X13)
	SHA512ROUND0(5, X13, X14, X15, X16, X17, X10, X11, X12)
	SHA512ROUND0(6, X12, X13, X14, X15, X16, X17, X10, X11)
	SHA512ROUND0(7, X11, X12, X13, X14, X15, X16, X17, X10)
	SHA512ROUND0(8, X10, X11, X12, X13, X14, X15, X16, X17)
	SHA512ROUND0(9, X17, X10, X11, X12, X13, X14, X15, X16)
	SHA512ROUND0(10, X16, X17, X10, X11, X12, X13, X14, X15)
	SHA512ROUND0(11, X15, X16, X17, X10, X11, X12, X13, X14)
	SHA512ROUND0(12, X14, X15, X16, X17, X10, X11, X12, X13)
	SHA512ROUND0(13, X13, X14, X15, X16, X17, X10, X11, X12)
	SHA512ROUND0(14, X12, X13, X14, X15, X16, X17, X10, X11)
	SHA512ROUND0(15, X11, X12, X13, X14, X15, X16, X17, X10)

	SHA512ROUND1(16, X10, X11, X12, X13, X14, X15, X16, X17)
	SHA512ROUND1(17, X17, X10, X11, X12, X13, X14, X15, X16)
	SHA512ROUND1(18, X16, X17, X10, X11, X12, X13, X14, X15)
	SHA512ROUND1(19, X15, X16, X17, X10, X11, X12, X13, X14)
	SHA512ROUND1(20, X14, X15, X16, X17, X10, X11, X12, X13)
	SHA512ROUND1(21, X13, X14, X15, X16, X17, X10, X11, X12)
	SHA512ROUND1(22, X12, X13, X14, X15, X16, X17, X10, X11)
	SHA512ROUND1(23, X11, X12, X13, X14, X15, X16, X17, X10)
	SHA512ROUND1(24, X10, X11, X12, X13, X14, X15, X16, X17)
	SHA512ROUND1(25, X17, X10, X11, X12, X13, X14, X15, X16)
	SHA512ROUND1(26, X16, X17, X10, X11, X12, X13, X14, X15)
	SHA512ROUND1(27, X15, X16, X17, X10, X11, X12, X13, X14)
	SHA512ROUND1(28, X14, X15, X16, X17, X10, X11, X12, X13)
	SHA512ROUND1(29, X13, X14, X15, X16, X17, X10, X11, X12)
	SHA512ROUND1(30, X12, X13, X14, X15, X16, X17, X10, X11)
	SHA512ROUND1(31, X11, X12, X13, X14, X15, X16, X17, X10)
	SHA512ROUND1(32, X10, X11, X12, X13, X14, X15, X16, X17)
	SHA512ROUND1(33, X17, X10, X11, X12, X13, X14, X15, X16)
	SHA512ROUND1(34, X16, X17, X10, X11, X12, X13, X14, X15)
	SHA512ROUND1(35, X15, X16, X17, X10, X11, X12, X13, X14)
	SHA512ROUND1(36, X14, X15, X16, X17, X10, X11, X12, X13)
	SHA512ROUND1(37, X13, X14, X15, X16, X17, X10, X11, X12)
	SHA512ROUND1(38, X12, X13, X14, X15, X16, X17, X10, X11)
	SHA512ROUND1(39, X11, X12, X13, X14, X15, X16, X17, X10)
	SHA512ROUND1(40, X10, X11, X12, X13, X14, X15, X16, X17)
	SHA512ROUND1(41, X17, X10, X11, X12, X13, X14, X15, X16)
	SHA512ROUND1(42, X16, X17, X10, X11, X12, X13, X14, X15)
	SHA512ROUND1(43, X15, X16, X17, X10, X11, X12, X13, X14)
	SHA512ROUND1(44, X14, X15, X16, X17, X10, X11, X12, X13)
	SHA512ROUND1(45, X13, X14, X15, X16, X17, X10, X11, X12)
	SHA512ROUND1(46, X12, X13, X14, X15, X16, X17, X10, X11)
	SHA512ROUND1(47, X11, X12, X13, X14, X15, X16, X17, X10)
	SHA512ROUND1(48, X10, X11, X12, X13, X14, X15, X16, X17)
	SHA512ROUND1(49, X17, X10, X11, X12, X13, X14, X15, X16)
	SHA512ROUND1(50, X16, X17, X10, X11, X12, X13, X14, X15)
	SHA512ROUND1(51, X15, X16, X17, X10, X11, X12, X13, X14)
	SHA512ROUND1(52, X14, X15, X16, X17, X10, X11, X12, X13)
	SHA512ROUND1(53, X13, X14, X15, X16, X17, X10, X11, X12)
	SHA512ROUND1(54, X12, X13, X14, X15, X16, X17, X10, X11)
	SHA512ROUND1(55, X11, X12, X13, X14, X15, X16, X17, X10)
	SHA512ROUND1(56, X10, X11, X12, X13, X14, X15, X16, X17)
	SHA512ROUND1(57, X17, X10, X11, X12, X13, X14, X15, X16)
	SHA512ROUND1(58, X16, X17, X10, X11, X12, X13, X14, X15)
	SHA512ROUND1(59, X15, X16, X17, X10, X11, X12, X13, X14)
	SHA512ROUND1(60, X14, X15, X16, X17, X10, X11, X12, X13)
	SHA512ROUND1(61, X13, X14, X15, X16, X17, X10, X11, X12)
	SHA512ROUND1(62, X12, X13, X14, X15, X16, X17, X10, X11)
	SHA512ROUND1(63, X11, X12, X13, X14, X15, X16, X17, X10)
	SHA512ROUND1(64, X10, X11, X12, X13, X14, X15, X16, X17)
	SHA512ROUND1(65, X17, X10, X11, X12, X13, X14, X15, X16)
	SHA512ROUND1(66, X16, X17, X10, X11, X12, X13, X14, X15)
	SHA512ROUND1(67, X15, X16, X17, X10, X11, X12, X13, X14)
	SHA512ROUND1(68, X14, X15, X16, X17, X10, X11, X12, X13)
	SHA512ROUND1(69, X13, X14, X15, X16, X17, X10, X11, X12)
	SHA512ROUND1(70, X12, X13, X14, X15, X16, X17, X10, X11)
	SHA512ROUND1(71, X11, X12, X13, X14, X15, X16, X17, X10)
	SHA512ROUND1(72, X10, X11, X12, X13, X14, X15, X16, X17)
	SHA512ROUND1(73, X17, X10, X11, X12, X13, X14, X15, X16)
	SHA512ROUND1(74, X16, X17, X10, X11, X12, X13, X14, X15)
	SHA512ROUND1(75, X15, X16, X17, X10, X11, X12, X13, X14)
	SHA512ROUND1(76, X14, X15, X16, X17, X10, X11, X12, X13)
	SHA512ROUND1(77, X13, X14, X15, X16, X17, X10, X11, X12)
	SHA512ROUND1(78, X12, X13, X14, X15, X16, X17, X10, X11)
	SHA512ROUND1(79, X11, X12, X13, X14, X15, X16, X17, X10)

	MOV	(0*8)(X20), X5
	MOV	(1*8)(X20), X6
	MOV	(2*8)(X20), X7
	MOV	(3*8)(X20), X8
	ADD	X5, X10		// H0 = a + H0
	ADD	X6, X11		// H1 = b + H1
	ADD	X7, X12		// H2 = c + H2
	ADD	X8, X13		// H3 = d + H3
	MOV	X10, (0*8)(X20)
	MOV	X11, (1*8)(X20)
	MOV	X12, (2*8)(X20)
	MOV	X13, (3*8)(X20)
	MOV	(4*8)(X20), X5
	MOV	(5*8)(X20), X6
	MOV	(6*8)(X20), X7
	MOV	(7*8)(X20), X8
	ADD	X5, X14		// H4 = e + H4
	ADD	X6, X15		// H5 = f + H5
	ADD	X7, X16		// H6 = g + H6
	ADD	X8, X17		// H7 = h + H7
	MOV	X14, (4*8)(X20)
	MOV	X15, (5*8)(X20)
	MOV	X16, (6*8)(X20)
	MOV	X17, (7*8)(X20)

	ADD	$128, X29
	BNE	X28, X29, loop

end:
	RET
