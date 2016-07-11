// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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
	MOVL	(index*4)(SI), AX; \
	BSWAPL	AX; \
	MOVL	AX, (index*4)(BP)

// Wt = SIGMA1(Wt-2) + Wt-7 + SIGMA0(Wt-15) + Wt-16; for 16 <= t <= 63
//   SIGMA0(x) = ROTR(7,x) XOR ROTR(18,x) XOR SHR(3,x)
//   SIGMA1(x) = ROTR(17,x) XOR ROTR(19,x) XOR SHR(10,x)
#define MSGSCHEDULE1(index) \
	MOVL	((index-2)*4)(BP), AX; \
	MOVL	AX, CX; \
	RORL	$17, AX; \
	MOVL	CX, DX; \
	RORL	$19, CX; \
	SHRL	$10, DX; \
	MOVL	((index-15)*4)(BP), BX; \
	XORL	CX, AX; \
	MOVL	BX, CX; \
	XORL	DX, AX; \
	RORL	$7, BX; \
	MOVL	CX, DX; \
	SHRL	$3, DX; \
	RORL	$18, CX; \
	ADDL	((index-7)*4)(BP), AX; \
	XORL	CX, BX; \
	XORL	DX, BX; \
	ADDL	((index-16)*4)(BP), BX; \
	ADDL	BX, AX; \
	MOVL	AX, ((index)*4)(BP)

// Calculate T1 in AX - uses AX, BX, CX and DX registers.
// Wt is passed in AX.
//   T1 = h + BIGSIGMA1(e) + Ch(e, f, g) + Kt + Wt
//     BIGSIGMA1(x) = ROTR(6,x) XOR ROTR(11,x) XOR ROTR(25,x)
//     Ch(x, y, z) = (x AND y) XOR (NOT x AND z)
#define SHA256T1(const, e, f, g, h) \
	MOVL	(h*4)(DI), BX; \
	ADDL	AX, BX; \
	MOVL	(e*4)(DI), AX; \
	ADDL	$const, BX; \
	MOVL	(e*4)(DI), CX; \
	RORL	$6, AX; \
	MOVL	(e*4)(DI), DX; \
	RORL	$11, CX; \
	XORL	CX, AX; \
	MOVL	(e*4)(DI), CX; \
	RORL	$25, DX; \
	ANDL	(f*4)(DI), CX; \
	XORL	AX, DX; \
	MOVL	(e*4)(DI), AX; \
	NOTL	AX; \
	ADDL	DX, BX; \
	ANDL	(g*4)(DI), AX; \
	XORL	CX, AX; \
	ADDL	BX, AX

// Calculate T2 in BX - uses AX, BX, CX and DX registers.
//   T2 = BIGSIGMA0(a) + Maj(a, b, c)
//     BIGSIGMA0(x) = ROTR(2,x) XOR ROTR(13,x) XOR ROTR(22,x)
//     Maj(x, y, z) = (x AND y) XOR (x AND z) XOR (y AND z)
#define SHA256T2(a, b, c) \
	MOVL	(a*4)(DI), AX; \
	MOVL	(c*4)(DI), BX; \
	RORL	$2, AX; \
	MOVL	(a*4)(DI), DX; \
	ANDL	(b*4)(DI), BX; \
	RORL	$13, DX; \
	MOVL	(a*4)(DI), CX; \
	ANDL	(c*4)(DI), CX; \
	XORL	DX, AX; \
	XORL	CX, BX; \
	MOVL	(a*4)(DI), DX; \
	MOVL	(b*4)(DI), CX; \
	RORL	$22, DX; \
	ANDL	(a*4)(DI), CX; \
	XORL	CX, BX; \
	XORL	DX, AX; \
	ADDL	AX, BX

// Calculate T1 and T2, then e = d + T1 and a = T1 + T2.
// The values for e and a are stored in d and h, ready for rotation.
#define SHA256ROUND(index, const, a, b, c, d, e, f, g, h) \
	SHA256T1(const, e, f, g, h); \
	MOVL	AX, 292(SP); \
	SHA256T2(a, b, c); \
	MOVL	292(SP), AX; \
	ADDL	AX, BX; \
	ADDL	AX, (d*4)(DI); \
	MOVL	BX, (h*4)(DI)

#define SHA256ROUND0(index, const, a, b, c, d, e, f, g, h) \
	MSGSCHEDULE0(index); \
	SHA256ROUND(index, const, a, b, c, d, e, f, g, h)

#define SHA256ROUND1(index, const, a, b, c, d, e, f, g, h) \
	MSGSCHEDULE1(index); \
	SHA256ROUND(index, const, a, b, c, d, e, f, g, h)

TEXT ·block(SB),0,$296-16
	MOVL	p_base+4(FP), SI
	MOVL	p_len+8(FP), DX
	SHRL	$6, DX
	SHLL	$6, DX

	LEAL	(SI)(DX*1), DI
	MOVL	DI, 288(SP)
	CMPL	SI, DI
	JEQ	end

	LEAL	256(SP), DI		// variables

	MOVL	dig+0(FP), BP
	MOVL	(0*4)(BP), AX		// a = H0
	MOVL	AX, (0*4)(DI)
	MOVL	(1*4)(BP), BX		// b = H1
	MOVL	BX, (1*4)(DI)
	MOVL	(2*4)(BP), CX		// c = H2
	MOVL	CX, (2*4)(DI)
	MOVL	(3*4)(BP), DX		// d = H3
	MOVL	DX, (3*4)(DI)
	MOVL	(4*4)(BP), AX		// e = H4
	MOVL	AX, (4*4)(DI)
	MOVL	(5*4)(BP), BX		// f = H5
	MOVL	BX, (5*4)(DI)
	MOVL	(6*4)(BP), CX		// g = H6
	MOVL	CX, (6*4)(DI)
	MOVL	(7*4)(BP), DX		// h = H7
	MOVL	DX, (7*4)(DI)

loop:
	MOVL	SP, BP			// message schedule

	SHA256ROUND0(0, 0x428a2f98, 0, 1, 2, 3, 4, 5, 6, 7)
	SHA256ROUND0(1, 0x71374491, 7, 0, 1, 2, 3, 4, 5, 6)
	SHA256ROUND0(2, 0xb5c0fbcf, 6, 7, 0, 1, 2, 3, 4, 5)
	SHA256ROUND0(3, 0xe9b5dba5, 5, 6, 7, 0, 1, 2, 3, 4)
	SHA256ROUND0(4, 0x3956c25b, 4, 5, 6, 7, 0, 1, 2, 3)
	SHA256ROUND0(5, 0x59f111f1, 3, 4, 5, 6, 7, 0, 1, 2)
	SHA256ROUND0(6, 0x923f82a4, 2, 3, 4, 5, 6, 7, 0, 1)
	SHA256ROUND0(7, 0xab1c5ed5, 1, 2, 3, 4, 5, 6, 7, 0)
	SHA256ROUND0(8, 0xd807aa98, 0, 1, 2, 3, 4, 5, 6, 7)
	SHA256ROUND0(9, 0x12835b01, 7, 0, 1, 2, 3, 4, 5, 6)
	SHA256ROUND0(10, 0x243185be, 6, 7, 0, 1, 2, 3, 4, 5)
	SHA256ROUND0(11, 0x550c7dc3, 5, 6, 7, 0, 1, 2, 3, 4)
	SHA256ROUND0(12, 0x72be5d74, 4, 5, 6, 7, 0, 1, 2, 3)
	SHA256ROUND0(13, 0x80deb1fe, 3, 4, 5, 6, 7, 0, 1, 2)
	SHA256ROUND0(14, 0x9bdc06a7, 2, 3, 4, 5, 6, 7, 0, 1)
	SHA256ROUND0(15, 0xc19bf174, 1, 2, 3, 4, 5, 6, 7, 0)

	SHA256ROUND1(16, 0xe49b69c1, 0, 1, 2, 3, 4, 5, 6, 7)
	SHA256ROUND1(17, 0xefbe4786, 7, 0, 1, 2, 3, 4, 5, 6)
	SHA256ROUND1(18, 0x0fc19dc6, 6, 7, 0, 1, 2, 3, 4, 5)
	SHA256ROUND1(19, 0x240ca1cc, 5, 6, 7, 0, 1, 2, 3, 4)
	SHA256ROUND1(20, 0x2de92c6f, 4, 5, 6, 7, 0, 1, 2, 3)
	SHA256ROUND1(21, 0x4a7484aa, 3, 4, 5, 6, 7, 0, 1, 2)
	SHA256ROUND1(22, 0x5cb0a9dc, 2, 3, 4, 5, 6, 7, 0, 1)
	SHA256ROUND1(23, 0x76f988da, 1, 2, 3, 4, 5, 6, 7, 0)
	SHA256ROUND1(24, 0x983e5152, 0, 1, 2, 3, 4, 5, 6, 7)
	SHA256ROUND1(25, 0xa831c66d, 7, 0, 1, 2, 3, 4, 5, 6)
	SHA256ROUND1(26, 0xb00327c8, 6, 7, 0, 1, 2, 3, 4, 5)
	SHA256ROUND1(27, 0xbf597fc7, 5, 6, 7, 0, 1, 2, 3, 4)
	SHA256ROUND1(28, 0xc6e00bf3, 4, 5, 6, 7, 0, 1, 2, 3)
	SHA256ROUND1(29, 0xd5a79147, 3, 4, 5, 6, 7, 0, 1, 2)
	SHA256ROUND1(30, 0x06ca6351, 2, 3, 4, 5, 6, 7, 0, 1)
	SHA256ROUND1(31, 0x14292967, 1, 2, 3, 4, 5, 6, 7, 0)
	SHA256ROUND1(32, 0x27b70a85, 0, 1, 2, 3, 4, 5, 6, 7)
	SHA256ROUND1(33, 0x2e1b2138, 7, 0, 1, 2, 3, 4, 5, 6)
	SHA256ROUND1(34, 0x4d2c6dfc, 6, 7, 0, 1, 2, 3, 4, 5)
	SHA256ROUND1(35, 0x53380d13, 5, 6, 7, 0, 1, 2, 3, 4)
	SHA256ROUND1(36, 0x650a7354, 4, 5, 6, 7, 0, 1, 2, 3)
	SHA256ROUND1(37, 0x766a0abb, 3, 4, 5, 6, 7, 0, 1, 2)
	SHA256ROUND1(38, 0x81c2c92e, 2, 3, 4, 5, 6, 7, 0, 1)
	SHA256ROUND1(39, 0x92722c85, 1, 2, 3, 4, 5, 6, 7, 0)
	SHA256ROUND1(40, 0xa2bfe8a1, 0, 1, 2, 3, 4, 5, 6, 7)
	SHA256ROUND1(41, 0xa81a664b, 7, 0, 1, 2, 3, 4, 5, 6)
	SHA256ROUND1(42, 0xc24b8b70, 6, 7, 0, 1, 2, 3, 4, 5)
	SHA256ROUND1(43, 0xc76c51a3, 5, 6, 7, 0, 1, 2, 3, 4)
	SHA256ROUND1(44, 0xd192e819, 4, 5, 6, 7, 0, 1, 2, 3)
	SHA256ROUND1(45, 0xd6990624, 3, 4, 5, 6, 7, 0, 1, 2)
	SHA256ROUND1(46, 0xf40e3585, 2, 3, 4, 5, 6, 7, 0, 1)
	SHA256ROUND1(47, 0x106aa070, 1, 2, 3, 4, 5, 6, 7, 0)
	SHA256ROUND1(48, 0x19a4c116, 0, 1, 2, 3, 4, 5, 6, 7)
	SHA256ROUND1(49, 0x1e376c08, 7, 0, 1, 2, 3, 4, 5, 6)
	SHA256ROUND1(50, 0x2748774c, 6, 7, 0, 1, 2, 3, 4, 5)
	SHA256ROUND1(51, 0x34b0bcb5, 5, 6, 7, 0, 1, 2, 3, 4)
	SHA256ROUND1(52, 0x391c0cb3, 4, 5, 6, 7, 0, 1, 2, 3)
	SHA256ROUND1(53, 0x4ed8aa4a, 3, 4, 5, 6, 7, 0, 1, 2)
	SHA256ROUND1(54, 0x5b9cca4f, 2, 3, 4, 5, 6, 7, 0, 1)
	SHA256ROUND1(55, 0x682e6ff3, 1, 2, 3, 4, 5, 6, 7, 0)
	SHA256ROUND1(56, 0x748f82ee, 0, 1, 2, 3, 4, 5, 6, 7)
	SHA256ROUND1(57, 0x78a5636f, 7, 0, 1, 2, 3, 4, 5, 6)
	SHA256ROUND1(58, 0x84c87814, 6, 7, 0, 1, 2, 3, 4, 5)
	SHA256ROUND1(59, 0x8cc70208, 5, 6, 7, 0, 1, 2, 3, 4)
	SHA256ROUND1(60, 0x90befffa, 4, 5, 6, 7, 0, 1, 2, 3)
	SHA256ROUND1(61, 0xa4506ceb, 3, 4, 5, 6, 7, 0, 1, 2)
	SHA256ROUND1(62, 0xbef9a3f7, 2, 3, 4, 5, 6, 7, 0, 1)
	SHA256ROUND1(63, 0xc67178f2, 1, 2, 3, 4, 5, 6, 7, 0)

	MOVL	dig+0(FP), BP
	MOVL	(0*4)(BP), AX		// H0 = a + H0
	ADDL	(0*4)(DI), AX
	MOVL	AX, (0*4)(DI)
	MOVL	AX, (0*4)(BP)
	MOVL	(1*4)(BP), BX		// H1 = b + H1
	ADDL	(1*4)(DI), BX
	MOVL	BX, (1*4)(DI)
	MOVL	BX, (1*4)(BP)
	MOVL	(2*4)(BP), CX		// H2 = c + H2
	ADDL	(2*4)(DI), CX
	MOVL	CX, (2*4)(DI)
	MOVL	CX, (2*4)(BP)
	MOVL	(3*4)(BP), DX		// H3 = d + H3
	ADDL	(3*4)(DI), DX
	MOVL	DX, (3*4)(DI)
	MOVL	DX, (3*4)(BP)
	MOVL	(4*4)(BP), AX		// H4 = e + H4
	ADDL	(4*4)(DI), AX
	MOVL	AX, (4*4)(DI)
	MOVL	AX, (4*4)(BP)
	MOVL	(5*4)(BP), BX		// H5 = f + H5
	ADDL	(5*4)(DI), BX
	MOVL	BX, (5*4)(DI)
	MOVL	BX, (5*4)(BP)
	MOVL	(6*4)(BP), CX		// H6 = g + H6
	ADDL	(6*4)(DI), CX
	MOVL	CX, (6*4)(DI)
	MOVL	CX, (6*4)(BP)
	MOVL	(7*4)(BP), DX		// H7 = h + H7
	ADDL	(7*4)(DI), DX
	MOVL	DX, (7*4)(DI)
	MOVL	DX, (7*4)(BP)

	ADDL	$64, SI
	CMPL	SI, 288(SP)
	JB	loop

end:
	RET
