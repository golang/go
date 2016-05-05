// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// SHA512 block routine. See sha512block.go for Go equivalent.
//
// The algorithm is detailed in FIPS 180-4:
//
//  http://csrc.nist.gov/publications/fips/fips180-4/fips-180-4.pdf
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
	MOVQ	(index*8)(SI), AX; \
	BSWAPQ	AX; \
	MOVQ	AX, (index*8)(BP)

// Wt = SIGMA1(Wt-2) + Wt-7 + SIGMA0(Wt-15) + Wt-16; for 16 <= t <= 79
//   SIGMA0(x) = ROTR(1,x) XOR ROTR(8,x) XOR SHR(7,x)
//   SIGMA1(x) = ROTR(19,x) XOR ROTR(61,x) XOR SHR(6,x)
#define MSGSCHEDULE1(index) \
	MOVQ	((index-2)*8)(BP), AX; \
	MOVQ	AX, CX; \
	RORQ	$19, AX; \
	MOVQ	CX, DX; \
	RORQ	$61, CX; \
	SHRQ	$6, DX; \
	MOVQ	((index-15)*8)(BP), BX; \
	XORQ	CX, AX; \
	MOVQ	BX, CX; \
	XORQ	DX, AX; \
	RORQ	$1, BX; \
	MOVQ	CX, DX; \
	SHRQ	$7, DX; \
	RORQ	$8, CX; \
	ADDQ	((index-7)*8)(BP), AX; \
	XORQ	CX, BX; \
	XORQ	DX, BX; \
	ADDQ	((index-16)*8)(BP), BX; \
	ADDQ	BX, AX; \
	MOVQ	AX, ((index)*8)(BP)

// Calculate T1 in AX - uses AX, CX and DX registers.
// h is also used as an accumulator. Wt is passed in AX.
//   T1 = h + BIGSIGMA1(e) + Ch(e, f, g) + Kt + Wt
//     BIGSIGMA1(x) = ROTR(14,x) XOR ROTR(18,x) XOR ROTR(41,x)
//     Ch(x, y, z) = (x AND y) XOR (NOT x AND z)
#define SHA512T1(const, e, f, g, h) \
	MOVQ	$const, DX; \
	ADDQ	AX, h; \
	MOVQ	e, AX; \
	ADDQ	DX, h; \
	MOVQ	e, CX; \
	RORQ	$14, AX; \
	MOVQ	e, DX; \
	RORQ	$18, CX; \
	XORQ	CX, AX; \
	MOVQ	e, CX; \
	RORQ	$41, DX; \
	ANDQ	f, CX; \
	XORQ	AX, DX; \
	MOVQ	e, AX; \
	NOTQ	AX; \
	ADDQ	DX, h; \
	ANDQ	g, AX; \
	XORQ	CX, AX; \
	ADDQ	h, AX

// Calculate T2 in BX - uses BX, CX, DX and DI registers.
//   T2 = BIGSIGMA0(a) + Maj(a, b, c)
//     BIGSIGMA0(x) = ROTR(28,x) XOR ROTR(34,x) XOR ROTR(39,x)
//     Maj(x, y, z) = (x AND y) XOR (x AND z) XOR (y AND z)
#define SHA512T2(a, b, c) \
	MOVQ	a, DI; \
	MOVQ	c, BX; \
	RORQ	$28, DI; \
	MOVQ	a, DX; \
	ANDQ	b, BX; \
	RORQ	$34, DX; \
	MOVQ	a, CX; \
	ANDQ	c, CX; \
	XORQ	DX, DI; \
	XORQ	CX, BX; \
	MOVQ	a, DX; \
	MOVQ	b, CX; \
	RORQ	$39, DX; \
	ANDQ	a, CX; \
	XORQ	CX, BX; \
	XORQ	DX, DI; \
	ADDQ	DI, BX

// Calculate T1 and T2, then e = d + T1 and a = T1 + T2.
// The values for e and a are stored in d and h, ready for rotation.
#define SHA512ROUND(index, const, a, b, c, d, e, f, g, h) \
	SHA512T1(const, e, f, g, h); \
	SHA512T2(a, b, c); \
	MOVQ	BX, h; \
	ADDQ	AX, d; \
	ADDQ	AX, h

#define SHA512ROUND0(index, const, a, b, c, d, e, f, g, h) \
	MSGSCHEDULE0(index); \
	SHA512ROUND(index, const, a, b, c, d, e, f, g, h)

#define SHA512ROUND1(index, const, a, b, c, d, e, f, g, h) \
	MSGSCHEDULE1(index); \
	SHA512ROUND(index, const, a, b, c, d, e, f, g, h)

TEXT ·block(SB),0,$648-32
	MOVQ	p_base+8(FP), SI
	MOVQ	p_len+16(FP), DX
	SHRQ	$7, DX
	SHLQ	$7, DX

	LEAQ	(SI)(DX*1), DI
	MOVQ	DI, 640(SP)
	CMPQ	SI, DI
	JEQ	end

	MOVQ	dig+0(FP), BP
	MOVQ	(0*8)(BP), R8		// a = H0
	MOVQ	(1*8)(BP), R9		// b = H1
	MOVQ	(2*8)(BP), R10		// c = H2
	MOVQ	(3*8)(BP), R11		// d = H3
	MOVQ	(4*8)(BP), R12		// e = H4
	MOVQ	(5*8)(BP), R13		// f = H5
	MOVQ	(6*8)(BP), R14		// g = H6
	MOVQ	(7*8)(BP), R15		// h = H7

loop:
	MOVQ	SP, BP			// message schedule

	SHA512ROUND0(0, 0x428a2f98d728ae22, R8, R9, R10, R11, R12, R13, R14, R15)
	SHA512ROUND0(1, 0x7137449123ef65cd, R15, R8, R9, R10, R11, R12, R13, R14)
	SHA512ROUND0(2, 0xb5c0fbcfec4d3b2f, R14, R15, R8, R9, R10, R11, R12, R13)
	SHA512ROUND0(3, 0xe9b5dba58189dbbc, R13, R14, R15, R8, R9, R10, R11, R12)
	SHA512ROUND0(4, 0x3956c25bf348b538, R12, R13, R14, R15, R8, R9, R10, R11)
	SHA512ROUND0(5, 0x59f111f1b605d019, R11, R12, R13, R14, R15, R8, R9, R10)
	SHA512ROUND0(6, 0x923f82a4af194f9b, R10, R11, R12, R13, R14, R15, R8, R9)
	SHA512ROUND0(7, 0xab1c5ed5da6d8118, R9, R10, R11, R12, R13, R14, R15, R8)
	SHA512ROUND0(8, 0xd807aa98a3030242, R8, R9, R10, R11, R12, R13, R14, R15)
	SHA512ROUND0(9, 0x12835b0145706fbe, R15, R8, R9, R10, R11, R12, R13, R14)
	SHA512ROUND0(10, 0x243185be4ee4b28c, R14, R15, R8, R9, R10, R11, R12, R13)
	SHA512ROUND0(11, 0x550c7dc3d5ffb4e2, R13, R14, R15, R8, R9, R10, R11, R12)
	SHA512ROUND0(12, 0x72be5d74f27b896f, R12, R13, R14, R15, R8, R9, R10, R11)
	SHA512ROUND0(13, 0x80deb1fe3b1696b1, R11, R12, R13, R14, R15, R8, R9, R10)
	SHA512ROUND0(14, 0x9bdc06a725c71235, R10, R11, R12, R13, R14, R15, R8, R9)
	SHA512ROUND0(15, 0xc19bf174cf692694, R9, R10, R11, R12, R13, R14, R15, R8)

	SHA512ROUND1(16, 0xe49b69c19ef14ad2, R8, R9, R10, R11, R12, R13, R14, R15)
	SHA512ROUND1(17, 0xefbe4786384f25e3, R15, R8, R9, R10, R11, R12, R13, R14)
	SHA512ROUND1(18, 0x0fc19dc68b8cd5b5, R14, R15, R8, R9, R10, R11, R12, R13)
	SHA512ROUND1(19, 0x240ca1cc77ac9c65, R13, R14, R15, R8, R9, R10, R11, R12)
	SHA512ROUND1(20, 0x2de92c6f592b0275, R12, R13, R14, R15, R8, R9, R10, R11)
	SHA512ROUND1(21, 0x4a7484aa6ea6e483, R11, R12, R13, R14, R15, R8, R9, R10)
	SHA512ROUND1(22, 0x5cb0a9dcbd41fbd4, R10, R11, R12, R13, R14, R15, R8, R9)
	SHA512ROUND1(23, 0x76f988da831153b5, R9, R10, R11, R12, R13, R14, R15, R8)
	SHA512ROUND1(24, 0x983e5152ee66dfab, R8, R9, R10, R11, R12, R13, R14, R15)
	SHA512ROUND1(25, 0xa831c66d2db43210, R15, R8, R9, R10, R11, R12, R13, R14)
	SHA512ROUND1(26, 0xb00327c898fb213f, R14, R15, R8, R9, R10, R11, R12, R13)
	SHA512ROUND1(27, 0xbf597fc7beef0ee4, R13, R14, R15, R8, R9, R10, R11, R12)
	SHA512ROUND1(28, 0xc6e00bf33da88fc2, R12, R13, R14, R15, R8, R9, R10, R11)
	SHA512ROUND1(29, 0xd5a79147930aa725, R11, R12, R13, R14, R15, R8, R9, R10)
	SHA512ROUND1(30, 0x06ca6351e003826f, R10, R11, R12, R13, R14, R15, R8, R9)
	SHA512ROUND1(31, 0x142929670a0e6e70, R9, R10, R11, R12, R13, R14, R15, R8)
	SHA512ROUND1(32, 0x27b70a8546d22ffc, R8, R9, R10, R11, R12, R13, R14, R15)
	SHA512ROUND1(33, 0x2e1b21385c26c926, R15, R8, R9, R10, R11, R12, R13, R14)
	SHA512ROUND1(34, 0x4d2c6dfc5ac42aed, R14, R15, R8, R9, R10, R11, R12, R13)
	SHA512ROUND1(35, 0x53380d139d95b3df, R13, R14, R15, R8, R9, R10, R11, R12)
	SHA512ROUND1(36, 0x650a73548baf63de, R12, R13, R14, R15, R8, R9, R10, R11)
	SHA512ROUND1(37, 0x766a0abb3c77b2a8, R11, R12, R13, R14, R15, R8, R9, R10)
	SHA512ROUND1(38, 0x81c2c92e47edaee6, R10, R11, R12, R13, R14, R15, R8, R9)
	SHA512ROUND1(39, 0x92722c851482353b, R9, R10, R11, R12, R13, R14, R15, R8)
	SHA512ROUND1(40, 0xa2bfe8a14cf10364, R8, R9, R10, R11, R12, R13, R14, R15)
	SHA512ROUND1(41, 0xa81a664bbc423001, R15, R8, R9, R10, R11, R12, R13, R14)
	SHA512ROUND1(42, 0xc24b8b70d0f89791, R14, R15, R8, R9, R10, R11, R12, R13)
	SHA512ROUND1(43, 0xc76c51a30654be30, R13, R14, R15, R8, R9, R10, R11, R12)
	SHA512ROUND1(44, 0xd192e819d6ef5218, R12, R13, R14, R15, R8, R9, R10, R11)
	SHA512ROUND1(45, 0xd69906245565a910, R11, R12, R13, R14, R15, R8, R9, R10)
	SHA512ROUND1(46, 0xf40e35855771202a, R10, R11, R12, R13, R14, R15, R8, R9)
	SHA512ROUND1(47, 0x106aa07032bbd1b8, R9, R10, R11, R12, R13, R14, R15, R8)
	SHA512ROUND1(48, 0x19a4c116b8d2d0c8, R8, R9, R10, R11, R12, R13, R14, R15)
	SHA512ROUND1(49, 0x1e376c085141ab53, R15, R8, R9, R10, R11, R12, R13, R14)
	SHA512ROUND1(50, 0x2748774cdf8eeb99, R14, R15, R8, R9, R10, R11, R12, R13)
	SHA512ROUND1(51, 0x34b0bcb5e19b48a8, R13, R14, R15, R8, R9, R10, R11, R12)
	SHA512ROUND1(52, 0x391c0cb3c5c95a63, R12, R13, R14, R15, R8, R9, R10, R11)
	SHA512ROUND1(53, 0x4ed8aa4ae3418acb, R11, R12, R13, R14, R15, R8, R9, R10)
	SHA512ROUND1(54, 0x5b9cca4f7763e373, R10, R11, R12, R13, R14, R15, R8, R9)
	SHA512ROUND1(55, 0x682e6ff3d6b2b8a3, R9, R10, R11, R12, R13, R14, R15, R8)
	SHA512ROUND1(56, 0x748f82ee5defb2fc, R8, R9, R10, R11, R12, R13, R14, R15)
	SHA512ROUND1(57, 0x78a5636f43172f60, R15, R8, R9, R10, R11, R12, R13, R14)
	SHA512ROUND1(58, 0x84c87814a1f0ab72, R14, R15, R8, R9, R10, R11, R12, R13)
	SHA512ROUND1(59, 0x8cc702081a6439ec, R13, R14, R15, R8, R9, R10, R11, R12)
	SHA512ROUND1(60, 0x90befffa23631e28, R12, R13, R14, R15, R8, R9, R10, R11)
	SHA512ROUND1(61, 0xa4506cebde82bde9, R11, R12, R13, R14, R15, R8, R9, R10)
	SHA512ROUND1(62, 0xbef9a3f7b2c67915, R10, R11, R12, R13, R14, R15, R8, R9)
	SHA512ROUND1(63, 0xc67178f2e372532b, R9, R10, R11, R12, R13, R14, R15, R8)
	SHA512ROUND1(64, 0xca273eceea26619c, R8, R9, R10, R11, R12, R13, R14, R15)
	SHA512ROUND1(65, 0xd186b8c721c0c207, R15, R8, R9, R10, R11, R12, R13, R14)
	SHA512ROUND1(66, 0xeada7dd6cde0eb1e, R14, R15, R8, R9, R10, R11, R12, R13)
	SHA512ROUND1(67, 0xf57d4f7fee6ed178, R13, R14, R15, R8, R9, R10, R11, R12)
	SHA512ROUND1(68, 0x06f067aa72176fba, R12, R13, R14, R15, R8, R9, R10, R11)
	SHA512ROUND1(69, 0x0a637dc5a2c898a6, R11, R12, R13, R14, R15, R8, R9, R10)
	SHA512ROUND1(70, 0x113f9804bef90dae, R10, R11, R12, R13, R14, R15, R8, R9)
	SHA512ROUND1(71, 0x1b710b35131c471b, R9, R10, R11, R12, R13, R14, R15, R8)
	SHA512ROUND1(72, 0x28db77f523047d84, R8, R9, R10, R11, R12, R13, R14, R15)
	SHA512ROUND1(73, 0x32caab7b40c72493, R15, R8, R9, R10, R11, R12, R13, R14)
	SHA512ROUND1(74, 0x3c9ebe0a15c9bebc, R14, R15, R8, R9, R10, R11, R12, R13)
	SHA512ROUND1(75, 0x431d67c49c100d4c, R13, R14, R15, R8, R9, R10, R11, R12)
	SHA512ROUND1(76, 0x4cc5d4becb3e42b6, R12, R13, R14, R15, R8, R9, R10, R11)
	SHA512ROUND1(77, 0x597f299cfc657e2a, R11, R12, R13, R14, R15, R8, R9, R10)
	SHA512ROUND1(78, 0x5fcb6fab3ad6faec, R10, R11, R12, R13, R14, R15, R8, R9)
	SHA512ROUND1(79, 0x6c44198c4a475817, R9, R10, R11, R12, R13, R14, R15, R8)

	MOVQ	dig+0(FP), BP
	ADDQ	(0*8)(BP), R8	// H0 = a + H0
	MOVQ	R8, (0*8)(BP)
	ADDQ	(1*8)(BP), R9	// H1 = b + H1
	MOVQ	R9, (1*8)(BP)
	ADDQ	(2*8)(BP), R10	// H2 = c + H2
	MOVQ	R10, (2*8)(BP)
	ADDQ	(3*8)(BP), R11	// H3 = d + H3
	MOVQ	R11, (3*8)(BP)
	ADDQ	(4*8)(BP), R12	// H4 = e + H4
	MOVQ	R12, (4*8)(BP)
	ADDQ	(5*8)(BP), R13	// H5 = f + H5
	MOVQ	R13, (5*8)(BP)
	ADDQ	(6*8)(BP), R14	// H6 = g + H6
	MOVQ	R14, (6*8)(BP)
	ADDQ	(7*8)(BP), R15	// H7 = h + H7
	MOVQ	R15, (7*8)(BP)

	ADDQ	$128, SI
	CMPQ	SI, 640(SP)
	JB	loop

end:
	RET
