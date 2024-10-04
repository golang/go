// Copyright 2024 The Go Authors. All rights reserved.
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
// W[i] = M[i]; for 0 <= i <= 15
// W[i] = SIGMA1(W[i-2]) + W[i-7] + SIGMA0(W[i-15]) + W[i-16]; for 16 <= i <= 63
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
// for i = 0 to 63 {
//    T1 = h + BIGSIGMA1(e) + Ch(e,f,g) + K[i] + W[i]
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

#define REGTMP	R30
#define REGTMP1	R16
#define REGTMP2	R17
#define REGTMP3	R18
#define REGTMP4	R7
#define REGTMP5	R6

// W[i] = M[i]; for 0 <= i <= 15
#define LOAD0(index) \
	MOVW	(index*4)(R5), REGTMP4; \
	WORD	$0x38e7; \	// REVB2W REGTMP4, REGTMP4 to big-endian
	MOVW	REGTMP4, (index*4)(R3)

// W[i] = SIGMA1(W[i-2]) + W[i-7] + SIGMA0(W[i-15]) + W[i-16]; for 16 <= i <= 63
//   SIGMA0(x) = ROTR(7,x) XOR ROTR(18,x) XOR SHR(3,x)
//   SIGMA1(x) = ROTR(17,x) XOR ROTR(19,x) XOR SHR(10,x)
#define LOAD1(index) \
	MOVW	(((index-2)&0xf)*4)(R3), REGTMP4; \
	MOVW	(((index-15)&0xf)*4)(R3), REGTMP1; \
	MOVW	(((index-7)&0xf)*4)(R3), REGTMP; \
	MOVW	REGTMP4, REGTMP2; \
	MOVW	REGTMP4, REGTMP3; \
	ROTR	$17, REGTMP4; \
	ROTR	$19, REGTMP2; \
	SRL	$10, REGTMP3; \
	XOR	REGTMP2, REGTMP4; \
	XOR	REGTMP3, REGTMP4; \
	ROTR	$7, REGTMP1, REGTMP5; \
	SRL	$3, REGTMP1, REGTMP3; \
	ROTR	$18, REGTMP1, REGTMP2; \
	ADD	REGTMP, REGTMP4; \
	MOVW	(((index-16)&0xf)*4)(R3), REGTMP; \
	XOR	REGTMP3, REGTMP5; \
	XOR	REGTMP2, REGTMP5; \
	ADD	REGTMP, REGTMP5; \
	ADD	REGTMP5, REGTMP4; \
	MOVW	REGTMP4, ((index&0xf)*4)(R3)

// T1 = h + BIGSIGMA1(e) + Ch(e, f, g) + K[i] + W[i]
// BIGSIGMA1(x) = ROTR(6,x) XOR ROTR(11,x) XOR ROTR(25,x)
// Ch(x, y, z) = (x AND y) XOR (NOT x AND z)
// Calculate T1 in REGTMP4
#define SHA256T1(const, e, f, g, h) \
	ADDV	$const, h; \
	ADD	REGTMP4, h; \
	ROTR	$6, e, REGTMP4; \
	ROTR	$11, e, REGTMP; \
	ROTR	$25, e, REGTMP3; \
	AND	f, e, REGTMP2; \
	XOR	REGTMP, REGTMP4; \
	MOVV	$0xffffffff, REGTMP; \
	XOR	REGTMP4, REGTMP3; \
	XOR	REGTMP, e, REGTMP5; \
	ADD	REGTMP3, h; \
	AND	g, REGTMP5; \
	XOR	REGTMP2, REGTMP5; \
	ADD	h, REGTMP5, REGTMP4

// T2 = BIGSIGMA0(a) + Maj(a, b, c)
// BIGSIGMA0(x) = ROTR(2,x) XOR ROTR(13,x) XOR ROTR(22,x)
// Maj(x, y, z) = (x AND y) XOR (x AND z) XOR (y AND z)
// Calculate T2 in REGTMP1
#define SHA256T2(a, b, c) \
	ROTR	$2, a, REGTMP5; \
	AND	b, c, REGTMP1; \
	ROTR	$13, a, REGTMP3; \
	AND	c, a, REGTMP; \
	XOR	REGTMP3, REGTMP5; \
	XOR	REGTMP, REGTMP1; \
	ROTR	$22, a, REGTMP2; \
	AND	a, b, REGTMP3; \
	XOR	REGTMP2, REGTMP5; \
	XOR	REGTMP3, REGTMP1; \
	ADD	REGTMP5, REGTMP1

// Calculate T1 and T2, then e = d + T1 and a = T1 + T2.
// The values for e and a are stored in d and h, ready for rotation.
#define SHA256ROUND(const, a, b, c, d, e, f, g, h) \
	SHA256T1(const, e, f, g, h); \
	SHA256T2(a, b, c); \
	ADD	REGTMP4, d; \
	ADD	REGTMP1, REGTMP4, h

#define SHA256ROUND0(index, const, a, b, c, d, e, f, g, h) \
	LOAD0(index); \
	SHA256ROUND(const, a, b, c, d, e, f, g, h)

#define SHA256ROUND1(index, const, a, b, c, d, e, f, g, h) \
	LOAD1(index); \
	SHA256ROUND(const, a, b, c, d, e, f, g, h)

// A stack frame size of 64 bytes is required here, because
// the frame size used for data expansion is 64 bytes.
// See the definition of the macro LOAD1 above (4 bytes * 16 entries).
//
//func block(dig *digest, p []byte)
TEXT ·block(SB),NOSPLIT,$64-32
	MOVV	p_base+8(FP), R5
	MOVV	p_len+16(FP), R6
	AND	$~63, R6
	BEQ	R6, end

	// p_len >= 64
	MOVV	dig+0(FP), R4
	ADDV	R5, R6, R25
	MOVW	(0*4)(R4), R8	// a = H0
	MOVW	(1*4)(R4), R9	// b = H1
	MOVW	(2*4)(R4), R10	// c = H2
	MOVW	(3*4)(R4), R11	// d = H3
	MOVW	(4*4)(R4), R12	// e = H4
	MOVW	(5*4)(R4), R13	// f = H5
	MOVW	(6*4)(R4), R14	// g = H6
	MOVW	(7*4)(R4), R15	// h = H7

loop:
	SHA256ROUND0(0,  0x428a2f98, R8,  R9,  R10, R11, R12, R13, R14, R15)
	SHA256ROUND0(1,  0x71374491, R15, R8,  R9,  R10, R11, R12, R13, R14)
	SHA256ROUND0(2,  0xb5c0fbcf, R14, R15, R8,  R9,  R10, R11, R12, R13)
	SHA256ROUND0(3,  0xe9b5dba5, R13, R14, R15, R8,  R9,  R10, R11, R12)
	SHA256ROUND0(4,  0x3956c25b, R12, R13, R14, R15, R8,  R9,  R10, R11)
	SHA256ROUND0(5,  0x59f111f1, R11, R12, R13, R14, R15, R8,  R9,  R10)
	SHA256ROUND0(6,  0x923f82a4, R10, R11, R12, R13, R14, R15, R8,  R9)
	SHA256ROUND0(7,  0xab1c5ed5, R9,  R10, R11, R12, R13, R14, R15, R8)
	SHA256ROUND0(8,  0xd807aa98, R8,  R9,  R10, R11, R12, R13, R14, R15)
	SHA256ROUND0(9,  0x12835b01, R15, R8,  R9,  R10, R11, R12, R13, R14)
	SHA256ROUND0(10, 0x243185be, R14, R15, R8,  R9,  R10, R11, R12, R13)
	SHA256ROUND0(11, 0x550c7dc3, R13, R14, R15, R8,  R9,  R10, R11, R12)
	SHA256ROUND0(12, 0x72be5d74, R12, R13, R14, R15, R8,  R9,  R10, R11)
	SHA256ROUND0(13, 0x80deb1fe, R11, R12, R13, R14, R15, R8,  R9,  R10)
	SHA256ROUND0(14, 0x9bdc06a7, R10, R11, R12, R13, R14, R15, R8,  R9)
	SHA256ROUND0(15, 0xc19bf174, R9,  R10, R11, R12, R13, R14, R15, R8)

	SHA256ROUND1(16, 0xe49b69c1, R8,  R9,  R10, R11, R12, R13, R14, R15)
	SHA256ROUND1(17, 0xefbe4786, R15, R8,  R9,  R10, R11, R12, R13, R14)
	SHA256ROUND1(18, 0x0fc19dc6, R14, R15, R8,  R9,  R10, R11, R12, R13)
	SHA256ROUND1(19, 0x240ca1cc, R13, R14, R15, R8,  R9,  R10, R11, R12)
	SHA256ROUND1(20, 0x2de92c6f, R12, R13, R14, R15, R8,  R9,  R10, R11)
	SHA256ROUND1(21, 0x4a7484aa, R11, R12, R13, R14, R15, R8,  R9,  R10)
	SHA256ROUND1(22, 0x5cb0a9dc, R10, R11, R12, R13, R14, R15, R8,  R9)
	SHA256ROUND1(23, 0x76f988da, R9,  R10, R11, R12, R13, R14, R15, R8)
	SHA256ROUND1(24, 0x983e5152, R8,  R9,  R10, R11, R12, R13, R14, R15)
	SHA256ROUND1(25, 0xa831c66d, R15, R8,  R9,  R10, R11, R12, R13, R14)
	SHA256ROUND1(26, 0xb00327c8, R14, R15, R8,  R9,  R10, R11, R12, R13)
	SHA256ROUND1(27, 0xbf597fc7, R13, R14, R15, R8,  R9,  R10, R11, R12)
	SHA256ROUND1(28, 0xc6e00bf3, R12, R13, R14, R15, R8,  R9,  R10, R11)
	SHA256ROUND1(29, 0xd5a79147, R11, R12, R13, R14, R15, R8,  R9,  R10)
	SHA256ROUND1(30, 0x06ca6351, R10, R11, R12, R13, R14, R15, R8,  R9)
	SHA256ROUND1(31, 0x14292967, R9,  R10, R11, R12, R13, R14, R15, R8)
	SHA256ROUND1(32, 0x27b70a85, R8,  R9,  R10, R11, R12, R13, R14, R15)
	SHA256ROUND1(33, 0x2e1b2138, R15, R8,  R9,  R10, R11, R12, R13, R14)
	SHA256ROUND1(34, 0x4d2c6dfc, R14, R15, R8,  R9,  R10, R11, R12, R13)
	SHA256ROUND1(35, 0x53380d13, R13, R14, R15, R8,  R9,  R10, R11, R12)
	SHA256ROUND1(36, 0x650a7354, R12, R13, R14, R15, R8,  R9,  R10, R11)
	SHA256ROUND1(37, 0x766a0abb, R11, R12, R13, R14, R15, R8,  R9,  R10)
	SHA256ROUND1(38, 0x81c2c92e, R10, R11, R12, R13, R14, R15, R8,  R9)
	SHA256ROUND1(39, 0x92722c85, R9,  R10, R11, R12, R13, R14, R15, R8)
	SHA256ROUND1(40, 0xa2bfe8a1, R8,  R9,  R10, R11, R12, R13, R14, R15)
	SHA256ROUND1(41, 0xa81a664b, R15, R8,  R9,  R10, R11, R12, R13, R14)
	SHA256ROUND1(42, 0xc24b8b70, R14, R15, R8,  R9,  R10, R11, R12, R13)
	SHA256ROUND1(43, 0xc76c51a3, R13, R14, R15, R8,  R9,  R10, R11, R12)
	SHA256ROUND1(44, 0xd192e819, R12, R13, R14, R15, R8,  R9,  R10, R11)
	SHA256ROUND1(45, 0xd6990624, R11, R12, R13, R14, R15, R8,  R9,  R10)
	SHA256ROUND1(46, 0xf40e3585, R10, R11, R12, R13, R14, R15, R8,  R9)
	SHA256ROUND1(47, 0x106aa070, R9,  R10, R11, R12, R13, R14, R15, R8)
	SHA256ROUND1(48, 0x19a4c116, R8,  R9,  R10, R11, R12, R13, R14, R15)
	SHA256ROUND1(49, 0x1e376c08, R15, R8,  R9,  R10, R11, R12, R13, R14)
	SHA256ROUND1(50, 0x2748774c, R14, R15, R8,  R9,  R10, R11, R12, R13)
	SHA256ROUND1(51, 0x34b0bcb5, R13, R14, R15, R8,  R9,  R10, R11, R12)
	SHA256ROUND1(52, 0x391c0cb3, R12, R13, R14, R15, R8,  R9,  R10, R11)
	SHA256ROUND1(53, 0x4ed8aa4a, R11, R12, R13, R14, R15, R8,  R9,  R10)
	SHA256ROUND1(54, 0x5b9cca4f, R10, R11, R12, R13, R14, R15, R8,  R9)
	SHA256ROUND1(55, 0x682e6ff3, R9,  R10, R11, R12, R13, R14, R15, R8)
	SHA256ROUND1(56, 0x748f82ee, R8,  R9,  R10, R11, R12, R13, R14, R15)
	SHA256ROUND1(57, 0x78a5636f, R15, R8,  R9,  R10, R11, R12, R13, R14)
	SHA256ROUND1(58, 0x84c87814, R14, R15, R8,  R9,  R10, R11, R12, R13)
	SHA256ROUND1(59, 0x8cc70208, R13, R14, R15, R8,  R9,  R10, R11, R12)
	SHA256ROUND1(60, 0x90befffa, R12, R13, R14, R15, R8,  R9,  R10, R11)
	SHA256ROUND1(61, 0xa4506ceb, R11, R12, R13, R14, R15, R8,  R9,  R10)
	SHA256ROUND1(62, 0xbef9a3f7, R10, R11, R12, R13, R14, R15, R8,  R9)
	SHA256ROUND1(63, 0xc67178f2, R9,  R10, R11, R12, R13, R14, R15, R8)

	MOVW	(0*4)(R4), REGTMP
	MOVW	(1*4)(R4), REGTMP1
	MOVW	(2*4)(R4), REGTMP2
	MOVW	(3*4)(R4), REGTMP3
	ADD	REGTMP, R8	// H0 = a + H0
	ADD	REGTMP1, R9	// H1 = b + H1
	ADD	REGTMP2, R10	// H2 = c + H2
	ADD	REGTMP3, R11	// H3 = d + H3
	MOVW	R8, (0*4)(R4)
	MOVW	R9, (1*4)(R4)
	MOVW	R10, (2*4)(R4)
	MOVW	R11, (3*4)(R4)
	MOVW	(4*4)(R4), REGTMP
	MOVW	(5*4)(R4), REGTMP1
	MOVW	(6*4)(R4), REGTMP2
	MOVW	(7*4)(R4), REGTMP3
	ADD	REGTMP, R12	// H4 = e + H4
	ADD	REGTMP1, R13	// H5 = f + H5
	ADD	REGTMP2, R14	// H6 = g + H6
	ADD	REGTMP3, R15	// H7 = h + H7
	MOVW	R12, (4*4)(R4)
	MOVW	R13, (5*4)(R4)
	MOVW	R14, (6*4)(R4)
	MOVW	R15, (7*4)(R4)

	ADDV	$64, R5
	BNE	R5, R25, loop

end:
	RET
