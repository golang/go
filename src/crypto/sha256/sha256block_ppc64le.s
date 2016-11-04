// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// SHA256 block routine. See sha256block.go for Go equivalent.
//
// The algorithm is detailed in FIPS 180-4:
//
//  http://csrc.nist.gov/publications/fips/fips180-4/fips-180-4.pdf
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
	MOVWZ	(index*4)(R26), R7; \
	RLWNM	$24, R7, $-1, R11; \
	RLWMI	$8, R7, $0x00FF0000, R11; \
	RLWMI	$8, R7, $0x000000FF, R11; \
	MOVWZ	R11, R7; \
	MOVWZ	R7, (index*4)(R27)

// Wt = SIGMA1(Wt-2) + Wt-7 + SIGMA0(Wt-15) + Wt-16; for 16 <= t <= 63
//   SIGMA0(x) = ROTR(7,x) XOR ROTR(18,x) XOR SHR(3,x)
//   SIGMA1(x) = ROTR(17,x) XOR ROTR(19,x) XOR SHR(10,x)
#define MSGSCHEDULE1(index) \
	MOVWZ	((index-2)*4)(R27), R7; \
	MOVWZ	R7, R9; \
	RLWNM	$32-17, R7, $-1, R7; \
	MOVWZ	R9, R10; \
	RLWNM	$32-19, R9, $-1, R9; \
	SRW	$10, R10; \
	MOVWZ	((index-15)*4)(R27), R8; \
	XOR	R9, R7; \
	MOVWZ	R8, R9; \
	XOR	R10, R7; \
	RLWNM	$32-7, R8, $-1, R8; \
	MOVWZ	R9, R10; \
	SRW	$3, R10; \
	RLWNM	$32-18, R9, $-1, R9; \
	MOVWZ	((index-7)*4)(R27), R11; \
	ADD	R11, R7; \
	XOR	R9, R8; \
	XOR	R10, R8; \
	MOVWZ	((index-16)*4)(R27), R11; \
	ADD	R11, R8; \
	ADD	R8, R7; \
	MOVWZ	R7, ((index)*4)(R27)

// T1 = h + BIGSIGMA1(e) + Ch(e, f, g) + Kt + Wt
//   BIGSIGMA1(x) = ROTR(6,x) XOR ROTR(11,x) XOR ROTR(25,x)
//   Ch(x, y, z) = (x AND y) XOR (NOT x AND z)
#define SHA256T1(const, e, f, g, h) \
	ADD	R7, h; \
	MOVWZ	e, R7; \
	ADD	$const, h; \
	MOVWZ	e, R9; \
	RLWNM	$32-6, R7, $-1, R7; \
	MOVWZ	e, R10; \
	RLWNM	$32-11, R9, $-1, R9; \
	XOR	R9, R7; \
	MOVWZ	e, R9; \
	RLWNM	$32-25, R10, $-1, R10; \
	AND	f, R9; \
	XOR	R7, R10; \
	MOVWZ	e, R7; \
	NOR	R7, R7, R7; \
	ADD	R10, h; \
	AND	g, R7; \
	XOR	R9, R7; \
	ADD	h, R7

// T2 = BIGSIGMA0(a) + Maj(a, b, c)
//   BIGSIGMA0(x) = ROTR(2,x) XOR ROTR(13,x) XOR ROTR(22,x)
//   Maj(x, y, z) = (x AND y) XOR (x AND z) XOR (y AND z)
#define SHA256T2(a, b, c) \
	MOVWZ	a, R28; \
	MOVWZ	c, R8; \
	RLWNM	$32-2, R28, $-1, R28; \
	MOVWZ	a, R10; \
	AND	b, R8; \
	RLWNM	$32-13, R10, $-1, R10; \
	MOVWZ	a, R9; \
	AND	c, R9; \
	XOR	R10, R28; \
	XOR	R9, R8; \
	MOVWZ	a, R10; \
	MOVWZ	b, R9; \
	RLWNM	$32-22, R10, $-1, R10; \
	AND	a, R9; \
	XOR	R9, R8; \
	XOR	R10, R28; \
	ADD	R28, R8

// Calculate T1 and T2, then e = d + T1 and a = T1 + T2.
// The values for e and a are stored in d and h, ready for rotation.
#define SHA256ROUND(index, const, a, b, c, d, e, f, g, h) \
	SHA256T1(const, e, f, g, h); \
	SHA256T2(a, b, c); \
	MOVWZ	R8, h; \
	ADD	R7, d; \
	ADD	R7, h

#define SHA256ROUND0(index, const, a, b, c, d, e, f, g, h) \
	MSGSCHEDULE0(index); \
	SHA256ROUND(index, const, a, b, c, d, e, f, g, h)

#define SHA256ROUND1(index, const, a, b, c, d, e, f, g, h) \
	MSGSCHEDULE1(index); \
	SHA256ROUND(index, const, a, b, c, d, e, f, g, h)

// func block(dig *digest, p []byte)
TEXT ·block(SB),0,$296-32
	MOVD	p_base+8(FP), R26
	MOVD	p_len+16(FP), R29
	SRD	$6, R29
	SLD	$6, R29

	ADD	R26, R29, R28

	MOVD	R28, 256(R1)
	CMP	R26, R28
	BEQ	end

	MOVD	dig+0(FP), R27
	MOVWZ	(0*4)(R27), R14		// a = H0
	MOVWZ	(1*4)(R27), R15		// b = H1
	MOVWZ	(2*4)(R27), R16		// c = H2
	MOVWZ	(3*4)(R27), R17		// d = H3
	MOVWZ	(4*4)(R27), R18		// e = H4
	MOVWZ	(5*4)(R27), R19		// f = H5
	MOVWZ	(6*4)(R27), R20		// g = H6
	MOVWZ	(7*4)(R27), R21		// h = H7

loop:
	MOVD	R1, R27		// R27: message schedule

	SHA256ROUND0(0, 0x428a2f98, R14, R15, R16, R17, R18, R19, R20, R21)
	SHA256ROUND0(1, 0x71374491, R21, R14, R15, R16, R17, R18, R19, R20)
	SHA256ROUND0(2, 0xb5c0fbcf, R20, R21, R14, R15, R16, R17, R18, R19)
	SHA256ROUND0(3, 0xe9b5dba5, R19, R20, R21, R14, R15, R16, R17, R18)
	SHA256ROUND0(4, 0x3956c25b, R18, R19, R20, R21, R14, R15, R16, R17)
	SHA256ROUND0(5, 0x59f111f1, R17, R18, R19, R20, R21, R14, R15, R16)
	SHA256ROUND0(6, 0x923f82a4, R16, R17, R18, R19, R20, R21, R14, R15)
	SHA256ROUND0(7, 0xab1c5ed5, R15, R16, R17, R18, R19, R20, R21, R14)
	SHA256ROUND0(8, 0xd807aa98, R14, R15, R16, R17, R18, R19, R20, R21)
	SHA256ROUND0(9, 0x12835b01, R21, R14, R15, R16, R17, R18, R19, R20)
	SHA256ROUND0(10, 0x243185be, R20, R21, R14, R15, R16, R17, R18, R19)
	SHA256ROUND0(11, 0x550c7dc3, R19, R20, R21, R14, R15, R16, R17, R18)
	SHA256ROUND0(12, 0x72be5d74, R18, R19, R20, R21, R14, R15, R16, R17)
	SHA256ROUND0(13, 0x80deb1fe, R17, R18, R19, R20, R21, R14, R15, R16)
	SHA256ROUND0(14, 0x9bdc06a7, R16, R17, R18, R19, R20, R21, R14, R15)
	SHA256ROUND0(15, 0xc19bf174, R15, R16, R17, R18, R19, R20, R21, R14)

	SHA256ROUND1(16, 0xe49b69c1, R14, R15, R16, R17, R18, R19, R20, R21)
	SHA256ROUND1(17, 0xefbe4786, R21, R14, R15, R16, R17, R18, R19, R20)
	SHA256ROUND1(18, 0x0fc19dc6, R20, R21, R14, R15, R16, R17, R18, R19)
	SHA256ROUND1(19, 0x240ca1cc, R19, R20, R21, R14, R15, R16, R17, R18)
	SHA256ROUND1(20, 0x2de92c6f, R18, R19, R20, R21, R14, R15, R16, R17)
	SHA256ROUND1(21, 0x4a7484aa, R17, R18, R19, R20, R21, R14, R15, R16)
	SHA256ROUND1(22, 0x5cb0a9dc, R16, R17, R18, R19, R20, R21, R14, R15)
	SHA256ROUND1(23, 0x76f988da, R15, R16, R17, R18, R19, R20, R21, R14)
	SHA256ROUND1(24, 0x983e5152, R14, R15, R16, R17, R18, R19, R20, R21)
	SHA256ROUND1(25, 0xa831c66d, R21, R14, R15, R16, R17, R18, R19, R20)
	SHA256ROUND1(26, 0xb00327c8, R20, R21, R14, R15, R16, R17, R18, R19)
	SHA256ROUND1(27, 0xbf597fc7, R19, R20, R21, R14, R15, R16, R17, R18)
	SHA256ROUND1(28, 0xc6e00bf3, R18, R19, R20, R21, R14, R15, R16, R17)
	SHA256ROUND1(29, 0xd5a79147, R17, R18, R19, R20, R21, R14, R15, R16)
	SHA256ROUND1(30, 0x06ca6351, R16, R17, R18, R19, R20, R21, R14, R15)
	SHA256ROUND1(31, 0x14292967, R15, R16, R17, R18, R19, R20, R21, R14)
	SHA256ROUND1(32, 0x27b70a85, R14, R15, R16, R17, R18, R19, R20, R21)
	SHA256ROUND1(33, 0x2e1b2138, R21, R14, R15, R16, R17, R18, R19, R20)
	SHA256ROUND1(34, 0x4d2c6dfc, R20, R21, R14, R15, R16, R17, R18, R19)
	SHA256ROUND1(35, 0x53380d13, R19, R20, R21, R14, R15, R16, R17, R18)
	SHA256ROUND1(36, 0x650a7354, R18, R19, R20, R21, R14, R15, R16, R17)
	SHA256ROUND1(37, 0x766a0abb, R17, R18, R19, R20, R21, R14, R15, R16)
	SHA256ROUND1(38, 0x81c2c92e, R16, R17, R18, R19, R20, R21, R14, R15)
	SHA256ROUND1(39, 0x92722c85, R15, R16, R17, R18, R19, R20, R21, R14)
	SHA256ROUND1(40, 0xa2bfe8a1, R14, R15, R16, R17, R18, R19, R20, R21)
	SHA256ROUND1(41, 0xa81a664b, R21, R14, R15, R16, R17, R18, R19, R20)
	SHA256ROUND1(42, 0xc24b8b70, R20, R21, R14, R15, R16, R17, R18, R19)
	SHA256ROUND1(43, 0xc76c51a3, R19, R20, R21, R14, R15, R16, R17, R18)
	SHA256ROUND1(44, 0xd192e819, R18, R19, R20, R21, R14, R15, R16, R17)
	SHA256ROUND1(45, 0xd6990624, R17, R18, R19, R20, R21, R14, R15, R16)
	SHA256ROUND1(46, 0xf40e3585, R16, R17, R18, R19, R20, R21, R14, R15)
	SHA256ROUND1(47, 0x106aa070, R15, R16, R17, R18, R19, R20, R21, R14)
	SHA256ROUND1(48, 0x19a4c116, R14, R15, R16, R17, R18, R19, R20, R21)
	SHA256ROUND1(49, 0x1e376c08, R21, R14, R15, R16, R17, R18, R19, R20)
	SHA256ROUND1(50, 0x2748774c, R20, R21, R14, R15, R16, R17, R18, R19)
	SHA256ROUND1(51, 0x34b0bcb5, R19, R20, R21, R14, R15, R16, R17, R18)
	SHA256ROUND1(52, 0x391c0cb3, R18, R19, R20, R21, R14, R15, R16, R17)
	SHA256ROUND1(53, 0x4ed8aa4a, R17, R18, R19, R20, R21, R14, R15, R16)
	SHA256ROUND1(54, 0x5b9cca4f, R16, R17, R18, R19, R20, R21, R14, R15)
	SHA256ROUND1(55, 0x682e6ff3, R15, R16, R17, R18, R19, R20, R21, R14)
	SHA256ROUND1(56, 0x748f82ee, R14, R15, R16, R17, R18, R19, R20, R21)
	SHA256ROUND1(57, 0x78a5636f, R21, R14, R15, R16, R17, R18, R19, R20)
	SHA256ROUND1(58, 0x84c87814, R20, R21, R14, R15, R16, R17, R18, R19)
	SHA256ROUND1(59, 0x8cc70208, R19, R20, R21, R14, R15, R16, R17, R18)
	SHA256ROUND1(60, 0x90befffa, R18, R19, R20, R21, R14, R15, R16, R17)
	SHA256ROUND1(61, 0xa4506ceb, R17, R18, R19, R20, R21, R14, R15, R16)
	SHA256ROUND1(62, 0xbef9a3f7, R16, R17, R18, R19, R20, R21, R14, R15)
	SHA256ROUND1(63, 0xc67178f2, R15, R16, R17, R18, R19, R20, R21, R14)

	MOVD	dig+0(FP), R27
	MOVWZ	(0*4)(R27), R11
	ADD	R11, R14	// H0 = a + H0
	MOVWZ	R14, (0*4)(R27)
	MOVWZ	(1*4)(R27), R11
	ADD	R11, R15	// H1 = b + H1
	MOVWZ	R15, (1*4)(R27)
	MOVWZ	(2*4)(R27), R11
	ADD	R11, R16	// H2 = c + H2
	MOVWZ	R16, (2*4)(R27)
	MOVWZ	(3*4)(R27), R11
	ADD	R11, R17	// H3 = d + H3
	MOVWZ	R17, (3*4)(R27)
	MOVWZ	(4*4)(R27), R11
	ADD	R11, R18	// H4 = e + H4
	MOVWZ	R18, (4*4)(R27)
	MOVWZ	(5*4)(R27), R11
	ADD	R11, R19	// H5 = f + H5
	MOVWZ	R19, (5*4)(R27)
	MOVWZ	(6*4)(R27), R11
	ADD	R11, R20	// H6 = g + H6
	MOVWZ	R20, (6*4)(R27)
	MOVWZ	(7*4)(R27), R11
	ADD	R11, R21	// H7 = h + H7
	MOVWZ	R21, (7*4)(R27)

	ADD	$64, R26
	MOVD	256(R1), R11
	CMPU	R26, R11
	BLT	loop

end:
	RET
