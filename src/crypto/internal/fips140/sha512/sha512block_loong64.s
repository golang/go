// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

#include "textflag.h"

// SHA512 block routine. See sha512block.go for Go equivalent.

#define REGTMP	R30
#define REGTMP1	R16
#define REGTMP2	R17
#define REGTMP3	R18
#define REGTMP4	R7
#define REGTMP5	R6
#define REG_KT	R19

// W[i] = M[i]; for 0 <= i <= 15
#define LOAD0(index) \
	MOVV	(index*8)(R5), REGTMP4; \
	REVBV	REGTMP4, REGTMP4; \
	MOVV	REGTMP4, (index*8)(R3)

// W[i] = SIGMA1(W[i-2]) + W[i-7] + SIGMA0(W[i-15]) + W[i-16]; for 16 <= i <= 79
//   SIGMA0(x) = ROTR(1,x) XOR ROTR(8,x) XOR SHR(7,x)
//   SIGMA1(x) = ROTR(19,x) XOR ROTR(61,x) XOR SHR(6,x)
#define LOAD1(index) \
	MOVV	(((index-2)&0xf)*8)(R3), REGTMP4; \
	MOVV	(((index-15)&0xf)*8)(R3), REGTMP1; \
	MOVV	(((index-7)&0xf)*8)(R3), REGTMP; \
	MOVV	REGTMP4, REGTMP2; \
	MOVV	REGTMP4, REGTMP3; \
	ROTRV	$19, REGTMP4; \
	ROTRV	$61, REGTMP2; \
	SRLV	$6, REGTMP3; \
	XOR	REGTMP2, REGTMP4; \
	XOR	REGTMP3, REGTMP4; \
	ROTRV	$1, REGTMP1, REGTMP5; \
	SRLV	$7, REGTMP1, REGTMP2; \
	ROTRV	$8, REGTMP1; \
	ADDV	REGTMP, REGTMP4; \
	MOVV	(((index-16)&0xf)*8)(R3), REGTMP; \
	XOR	REGTMP1, REGTMP5; \
	XOR	REGTMP2, REGTMP5; \
	ADDV	REGTMP, REGTMP5; \
	ADDV	REGTMP5, REGTMP4; \
	MOVV	REGTMP4, ((index&0xf)*8)(R3)

// h is also used as an accumulator. Wt is passed in REGTMP4.
//   T1 = h + BIGSIGMA1(e) + Ch(e, f, g) + K[i] + W[i]
//     BIGSIGMA1(x) = ROTR(14,x) XOR ROTR(18,x) XOR ROTR(41,x)
//     Ch(x, y, z) = (x AND y) XOR (NOT x AND z)
//                 = ((y XOR z) AND x) XOR z
// Calculate T1 in REGTMP4
#define SHA512T1(index, e, f, g, h) \
	MOVV	(index*8)(REG_KT), REGTMP5; \
	ADDV	REGTMP5, h; \
	ADDV	REGTMP4, h; \
	ROTRV	$14, e, REGTMP5; \
	ROTRV	$18, e, REGTMP; \
	ROTRV	$41, e, REGTMP3; \
	XOR	f, g, REGTMP2; \
	XOR	REGTMP, REGTMP5; \
	AND	e, REGTMP2; \
	XOR	REGTMP5, REGTMP3; \
	XOR	g, REGTMP2; \
	ADDV	REGTMP3, h; \
	ADDV	h, REGTMP2, REGTMP4

// T2 = BIGSIGMA0(a) + Maj(a, b, c)
// BIGSIGMA0(x) = ROTR(28,x) XOR ROTR(34,x) XOR ROTR(39,x)
// Maj(x, y, z) = (x AND y) XOR (x AND z) XOR (y AND z)
//              = ((y XOR z) AND x) XOR (y AND z)
// Calculate T2 in REGTMP1
#define SHA512T2(a, b, c) \
	ROTRV	$28, a, REGTMP5; \
	ROTRV	$34, a, REGTMP3; \
	ROTRV	$39, a, REGTMP2; \
	XOR	b, c, REGTMP; \
	AND	b, c, REGTMP1; \
	XOR	REGTMP3, REGTMP5; \
	AND	REGTMP, a, REGTMP; \
	XOR	REGTMP2, REGTMP5; \
	XOR	REGTMP, REGTMP1; \
	ADDV	REGTMP5, REGTMP1

// Calculate T1 and T2, then e = d + T1 and a = T1 + T2.
// The values for e and a are stored in d and h, ready for rotation.
#define SHA512ROUND(index, a, b, c, d, e, f, g, h) \
	SHA512T1(index, e, f, g, h); \
	SHA512T2(a, b, c); \
	ADDV	REGTMP4, d; \
	ADDV	REGTMP1, REGTMP4, h

#define SHA512ROUND0(index, a, b, c, d, e, f, g, h) \
	LOAD0(index); \
	SHA512ROUND(index, a, b, c, d, e, f, g, h)

#define SHA512ROUND1(index, a, b, c, d, e, f, g, h) \
	LOAD1(index); \
	SHA512ROUND(index, a, b, c, d, e, f, g, h)

// A stack frame size of 128 bytes is required here, because
// the frame size used for data expansion is 128 bytes.
// See the definition of the macro LOAD1 above (8 bytes * 16 entries).
//
// func blockScalar(dig *Digest, p []byte)
TEXT ·blockScalar(SB),NOSPLIT,$128-32
	MOVV	p_len+16(FP), R6
	MOVV	p_base+8(FP), R5
	AND	$~127, R6
	BEQ	R6, end

	MOVV	$·_K(SB), REG_KT		// const table

	// p_len >= 128
	MOVV	dig+0(FP), R4
	ADDV	R5, R6, R25
	MOVV	(0*8)(R4), R8	// a = H0
	MOVV	(1*8)(R4), R9	// b = H1
	MOVV	(2*8)(R4), R10	// c = H2
	MOVV	(3*8)(R4), R11	// d = H3
	MOVV	(4*8)(R4), R12	// e = H4
	MOVV	(5*8)(R4), R13	// f = H5
	MOVV	(6*8)(R4), R14	// g = H6
	MOVV	(7*8)(R4), R15	// h = H7

loop:
	SHA512ROUND0( 0, R8,  R9,  R10, R11, R12, R13, R14, R15)
	SHA512ROUND0( 1, R15, R8,  R9,  R10, R11, R12, R13, R14)
	SHA512ROUND0( 2, R14, R15, R8,  R9,  R10, R11, R12, R13)
	SHA512ROUND0( 3, R13, R14, R15, R8,  R9,  R10, R11, R12)
	SHA512ROUND0( 4, R12, R13, R14, R15, R8,  R9,  R10, R11)
	SHA512ROUND0( 5, R11, R12, R13, R14, R15, R8,  R9,  R10)
	SHA512ROUND0( 6, R10, R11, R12, R13, R14, R15, R8,  R9)
	SHA512ROUND0( 7, R9,  R10, R11, R12, R13, R14, R15, R8)
	SHA512ROUND0( 8, R8,  R9,  R10, R11, R12, R13, R14, R15)
	SHA512ROUND0( 9, R15, R8,  R9,  R10, R11, R12, R13, R14)
	SHA512ROUND0(10, R14, R15, R8,  R9,  R10, R11, R12, R13)
	SHA512ROUND0(11, R13, R14, R15, R8,  R9,  R10, R11, R12)
	SHA512ROUND0(12, R12, R13, R14, R15, R8,  R9,  R10, R11)
	SHA512ROUND0(13, R11, R12, R13, R14, R15, R8,  R9,  R10)
	SHA512ROUND0(14, R10, R11, R12, R13, R14, R15, R8,  R9)
	SHA512ROUND0(15, R9,  R10, R11, R12, R13, R14, R15, R8)

	SHA512ROUND1(16, R8,  R9,  R10, R11, R12, R13, R14, R15)
	SHA512ROUND1(17, R15, R8,  R9,  R10, R11, R12, R13, R14)
	SHA512ROUND1(18, R14, R15, R8,  R9,  R10, R11, R12, R13)
	SHA512ROUND1(19, R13, R14, R15, R8,  R9,  R10, R11, R12)
	SHA512ROUND1(20, R12, R13, R14, R15, R8,  R9,  R10, R11)
	SHA512ROUND1(21, R11, R12, R13, R14, R15, R8,  R9,  R10)
	SHA512ROUND1(22, R10, R11, R12, R13, R14, R15, R8,  R9)
	SHA512ROUND1(23, R9,  R10, R11, R12, R13, R14, R15, R8)
	SHA512ROUND1(24, R8,  R9,  R10, R11, R12, R13, R14, R15)
	SHA512ROUND1(25, R15, R8,  R9,  R10, R11, R12, R13, R14)
	SHA512ROUND1(26, R14, R15, R8,  R9,  R10, R11, R12, R13)
	SHA512ROUND1(27, R13, R14, R15, R8,  R9,  R10, R11, R12)
	SHA512ROUND1(28, R12, R13, R14, R15, R8,  R9,  R10, R11)
	SHA512ROUND1(29, R11, R12, R13, R14, R15, R8,  R9,  R10)
	SHA512ROUND1(30, R10, R11, R12, R13, R14, R15, R8,  R9)
	SHA512ROUND1(31, R9,  R10, R11, R12, R13, R14, R15, R8)
	SHA512ROUND1(32, R8,  R9,  R10, R11, R12, R13, R14, R15)
	SHA512ROUND1(33, R15, R8,  R9,  R10, R11, R12, R13, R14)
	SHA512ROUND1(34, R14, R15, R8,  R9,  R10, R11, R12, R13)
	SHA512ROUND1(35, R13, R14, R15, R8,  R9,  R10, R11, R12)
	SHA512ROUND1(36, R12, R13, R14, R15, R8,  R9,  R10, R11)
	SHA512ROUND1(37, R11, R12, R13, R14, R15, R8,  R9,  R10)
	SHA512ROUND1(38, R10, R11, R12, R13, R14, R15, R8,  R9)
	SHA512ROUND1(39, R9,  R10, R11, R12, R13, R14, R15, R8)
	SHA512ROUND1(40, R8,  R9,  R10, R11, R12, R13, R14, R15)
	SHA512ROUND1(41, R15, R8,  R9,  R10, R11, R12, R13, R14)
	SHA512ROUND1(42, R14, R15, R8,  R9,  R10, R11, R12, R13)
	SHA512ROUND1(43, R13, R14, R15, R8,  R9,  R10, R11, R12)
	SHA512ROUND1(44, R12, R13, R14, R15, R8,  R9,  R10, R11)
	SHA512ROUND1(45, R11, R12, R13, R14, R15, R8,  R9,  R10)
	SHA512ROUND1(46, R10, R11, R12, R13, R14, R15, R8,  R9)
	SHA512ROUND1(47, R9,  R10, R11, R12, R13, R14, R15, R8)
	SHA512ROUND1(48, R8,  R9,  R10, R11, R12, R13, R14, R15)
	SHA512ROUND1(49, R15, R8,  R9,  R10, R11, R12, R13, R14)
	SHA512ROUND1(50, R14, R15, R8,  R9,  R10, R11, R12, R13)
	SHA512ROUND1(51, R13, R14, R15, R8,  R9,  R10, R11, R12)
	SHA512ROUND1(52, R12, R13, R14, R15, R8,  R9,  R10, R11)
	SHA512ROUND1(53, R11, R12, R13, R14, R15, R8,  R9,  R10)
	SHA512ROUND1(54, R10, R11, R12, R13, R14, R15, R8,  R9)
	SHA512ROUND1(55, R9,  R10, R11, R12, R13, R14, R15, R8)
	SHA512ROUND1(56, R8,  R9,  R10, R11, R12, R13, R14, R15)
	SHA512ROUND1(57, R15, R8,  R9,  R10, R11, R12, R13, R14)
	SHA512ROUND1(58, R14, R15, R8,  R9,  R10, R11, R12, R13)
	SHA512ROUND1(59, R13, R14, R15, R8,  R9,  R10, R11, R12)
	SHA512ROUND1(60, R12, R13, R14, R15, R8,  R9,  R10, R11)
	SHA512ROUND1(61, R11, R12, R13, R14, R15, R8,  R9,  R10)
	SHA512ROUND1(62, R10, R11, R12, R13, R14, R15, R8,  R9)
	SHA512ROUND1(63, R9,  R10, R11, R12, R13, R14, R15, R8)
	SHA512ROUND1(64, R8,  R9,  R10, R11, R12, R13, R14, R15)
	SHA512ROUND1(65, R15, R8,  R9,  R10, R11, R12, R13, R14)
	SHA512ROUND1(66, R14, R15, R8,  R9,  R10, R11, R12, R13)
	SHA512ROUND1(67, R13, R14, R15, R8,  R9,  R10, R11, R12)
	SHA512ROUND1(68, R12, R13, R14, R15, R8,  R9,  R10, R11)
	SHA512ROUND1(69, R11, R12, R13, R14, R15, R8,  R9,  R10)
	SHA512ROUND1(70, R10, R11, R12, R13, R14, R15, R8,  R9)
	SHA512ROUND1(71, R9,  R10, R11, R12, R13, R14, R15, R8)
	SHA512ROUND1(72, R8,  R9,  R10, R11, R12, R13, R14, R15)
	SHA512ROUND1(73, R15, R8,  R9,  R10, R11, R12, R13, R14)
	SHA512ROUND1(74, R14, R15, R8,  R9,  R10, R11, R12, R13)
	SHA512ROUND1(75, R13, R14, R15, R8,  R9,  R10, R11, R12)
	SHA512ROUND1(76, R12, R13, R14, R15, R8,  R9,  R10, R11)
	SHA512ROUND1(77, R11, R12, R13, R14, R15, R8,  R9,  R10)
	SHA512ROUND1(78, R10, R11, R12, R13, R14, R15, R8,  R9)
	SHA512ROUND1(79, R9,  R10, R11, R12, R13, R14, R15, R8)

	MOVV	(0*8)(R4), REGTMP
	MOVV	(1*8)(R4), REGTMP1
	MOVV	(2*8)(R4), REGTMP2
	MOVV	(3*8)(R4), REGTMP3
	ADDV	REGTMP, R8	// H0 = a + H0
	ADDV	REGTMP1, R9	// H1 = b + H1
	ADDV	REGTMP2, R10	// H2 = c + H2
	ADDV	REGTMP3, R11	// H3 = d + H3
	MOVV	R8, (0*8)(R4)
	MOVV	R9, (1*8)(R4)
	MOVV	R10, (2*8)(R4)
	MOVV	R11, (3*8)(R4)
	MOVV	(4*8)(R4), REGTMP
	MOVV	(5*8)(R4), REGTMP1
	MOVV	(6*8)(R4), REGTMP2
	MOVV	(7*8)(R4), REGTMP3
	ADDV	REGTMP, R12	// H4 = e + H4
	ADDV	REGTMP1, R13	// H5 = f + H5
	ADDV	REGTMP2, R14	// H6 = g + H6
	ADDV	REGTMP3, R15	// H7 = h + H7
	MOVV	R12, (4*8)(R4)
	MOVV	R13, (5*8)(R4)
	MOVV	R14, (6*8)(R4)
	MOVV	R15, (7*8)(R4)

	ADDV	$128, R5
	BNE	R5, R25, loop

end:
	RET

// LSX message-schedule path (requires HWCAP LSX).
//
// Ring buffer (R3): bytes 0..127 hold W[0..15] as in blockScalar; bytes
// 128..135 mirror slot 0, so a 16-byte VMOVQ at offset 120 reads
// [W[15], W[0]] without a cross-lane shuffle when a pair wraps around.
//
// V16..V19: operand pairs; V20..V22: scratch; V23: computed W pair.

// bswap64: VSHUFB index table reversing the 8 bytes of each 64-bit lane
// (big-endian message word -> host order).
DATA	bswap64<>+0x00(SB)/8, $0x0001020304050607
DATA	bswap64<>+0x08(SB)/8, $0x08090a0b0c0d0e0f
GLOBL	bswap64<>(SB), RODATA|NOPTR, $16

// WPAIR(i): computes V23 = [W[i], W[i+1]] (i even) and stores it at slot
// (i&15). The four operand pairs are adjacent ring-buffer slots, each
// loaded with a single VMOVQ; the slot-0 mirror handles the wrap case.
#define WPAIR(i) \
	VMOVQ	(((i-16)&15)*8)(R3), V16; \
	VMOVQ	(((i-7)&15)*8)(R3), V17; \
	VMOVQ	(((i-2)&15)*8)(R3), V18; \
	VMOVQ	(((i-15)&15)*8)(R3), V19; \
	VROTRV	$19, V18, V20; \
	VROTRV	$61, V18, V21; \
	VSRLV	$6, V18, V22; \
	VXORV	V21, V20, V20; \
	VXORV	V22, V20, V20; \
	VROTRV	$1, V19, V21; \
	VROTRV	$8, V19, V22; \
	VSRLV	$7, V19, V23; \
	VXORV	V22, V21, V21; \
	VXORV	V23, V21, V21; \
	VADDV	V16, V17, V22; \
	VADDV	V20, V21, V23; \
	VADDV	V22, V23, V23; \
	VMOVQ	V23, ((i&15)*8)(R3)

// WPAIRM: WPAIR plus mirroring the new slot 0 to offset 128.
// Used when i ≡ 0 (mod 16), i.e. i ∈ {16,32,48,64}.
#define WPAIRM(i) \
	WPAIR(i); \
	VMOVQ	V23.V[0], 128(R3)

// SHA512T1K: SHA512T1 variant for when REGTMP4 already holds K[i]+W[i],
// saving the K-table load and one ADDV.
#define SHA512T1K(e, f, g, h) \
	ADDV	REGTMP4, h; \
	ROTRV	$14, e, REGTMP5; \
	ROTRV	$18, e, REGTMP; \
	ROTRV	$41, e, REGTMP3; \
	XOR	f, g, REGTMP2; \
	XOR	REGTMP, REGTMP5; \
	AND	e, REGTMP2; \
	XOR	REGTMP5, REGTMP3; \
	XOR	g, REGTMP2; \
	ADDV	REGTMP3, h; \
	ADDV	h, REGTMP2, REGTMP4

#define SHA512ROUNDK(a, b, c, d, e, f, g, h) \
	SHA512T1K(e, f, g, h); \
	SHA512T2(a, b, c); \
	ADDV	REGTMP4, d; \
	ADDV	REGTMP1, REGTMP4, h

// PAIR0(i): byte-swaps message words i,i+1 (V24 = bswap64 mask), stores
// them to the ring buffer, adds K[i],K[i+1] in V22, and runs two scalar
// rounds off the extracted lanes. i even, 0 <= i <= 14.
#define PAIR0(i, a1,b1,c1,d1,e1,f1,g1,h1, a2,b2,c2,d2,e2,f2,g2,h2) \
	VMOVQ	(i*8)(R5), V23; \
	VSHUFB	V24, V23, V23, V23; \
	VMOVQ	V23, ((i&15)*8)(R3); \
	VMOVQ	(i*8)(REG_KT), V22; \
	VADDV	V22, V23, V22; \
	VMOVQ	V22.V[0], REGTMP4; \
	SHA512ROUNDK(a1,b1,c1,d1,e1,f1,g1,h1); \
	VMOVQ	V22.V[1], REGTMP4; \
	SHA512ROUNDK(a2,b2,c2,d2,e2,f2,g2,h2)

// PAIR(i): WPAIR plus a vector K-add; each W[i]+K[i] is extracted
// straight into REGTMP4 via a lane move (no memory round trip) and fed
// to a scalar round.
#define PAIR(i, a1,b1,c1,d1,e1,f1,g1,h1, a2,b2,c2,d2,e2,f2,g2,h2) \
	WPAIR(i); \
	VMOVQ	((i)*8)(REG_KT), V22; \
	VADDV	V22, V23, V22; \
	VMOVQ	V22.V[0], REGTMP4; \
	SHA512ROUNDK(a1,b1,c1,d1,e1,f1,g1,h1); \
	VMOVQ	V22.V[1], REGTMP4; \
	SHA512ROUNDK(a2,b2,c2,d2,e2,f2,g2,h2)

// PAIRM: PAIR using WPAIRM, for i ∈ {16,32,48,64}.
#define PAIRM(i, a1,b1,c1,d1,e1,f1,g1,h1, a2,b2,c2,d2,e2,f2,g2,h2) \
	WPAIRM(i); \
	VMOVQ	((i)*8)(REG_KT), V22; \
	VADDV	V22, V23, V22; \
	VMOVQ	V22.V[0], REGTMP4; \
	SHA512ROUNDK(a1,b1,c1,d1,e1,f1,g1,h1); \
	VMOVQ	V22.V[1], REGTMP4; \
	SHA512ROUNDK(a2,b2,c2,d2,e2,f2,g2,h2)

// func blockLSX(dig *Digest, p []byte)
TEXT ·blockLSX(SB),NOSPLIT,$144-32
	MOVV	p_len+16(FP), R6
	MOVV	p_base+8(FP), R5
	AND	$~127, R6
	BEQ	R6, lend

	MOVV	$·_K(SB), REG_KT
	MOVV	dig+0(FP), R4
	ADDV	R5, R6, R25
	MOVV	(0*8)(R4), R8
	MOVV	(1*8)(R4), R9
	MOVV	(2*8)(R4), R10
	MOVV	(3*8)(R4), R11
	MOVV	(4*8)(R4), R12
	MOVV	(5*8)(R4), R13
	MOVV	(6*8)(R4), R14
	MOVV	(7*8)(R4), R15

	MOVV	$bswap64<>(SB), R26
	VMOVQ	(R26), V24	// bswap64 mask, reused across blocks

lloop:
	PAIR0( 0, R8,  R9,  R10, R11, R12, R13, R14, R15, R15, R8,  R9,  R10, R11, R12, R13, R14)
	VMOVQ	V23.V[0], 128(R3)	// mirror slot 0 (W[0])
	PAIR0( 2, R14, R15, R8,  R9,  R10, R11, R12, R13, R13, R14, R15, R8,  R9,  R10, R11, R12)
	PAIR0( 4, R12, R13, R14, R15, R8,  R9,  R10, R11, R11, R12, R13, R14, R15, R8,  R9,  R10)
	PAIR0( 6, R10, R11, R12, R13, R14, R15, R8,  R9,  R9,  R10, R11, R12, R13, R14, R15, R8)
	PAIR0( 8, R8,  R9,  R10, R11, R12, R13, R14, R15, R15, R8,  R9,  R10, R11, R12, R13, R14)
	PAIR0(10, R14, R15, R8,  R9,  R10, R11, R12, R13, R13, R14, R15, R8,  R9,  R10, R11, R12)
	PAIR0(12, R12, R13, R14, R15, R8,  R9,  R10, R11, R11, R12, R13, R14, R15, R8,  R9,  R10)
	PAIR0(14, R10, R11, R12, R13, R14, R15, R8,  R9,  R9,  R10, R11, R12, R13, R14, R15, R8)

	PAIRM(16, R8,  R9,  R10, R11, R12, R13, R14, R15, R15, R8,  R9,  R10, R11, R12, R13, R14)
	PAIR(18,  R14, R15, R8,  R9,  R10, R11, R12, R13, R13, R14, R15, R8,  R9,  R10, R11, R12)
	PAIR(20,  R12, R13, R14, R15, R8,  R9,  R10, R11, R11, R12, R13, R14, R15, R8,  R9,  R10)
	PAIR(22,  R10, R11, R12, R13, R14, R15, R8,  R9,  R9,  R10, R11, R12, R13, R14, R15, R8)
	PAIR(24,  R8,  R9,  R10, R11, R12, R13, R14, R15, R15, R8,  R9,  R10, R11, R12, R13, R14)
	PAIR(26,  R14, R15, R8,  R9,  R10, R11, R12, R13, R13, R14, R15, R8,  R9,  R10, R11, R12)
	PAIR(28,  R12, R13, R14, R15, R8,  R9,  R10, R11, R11, R12, R13, R14, R15, R8,  R9,  R10)
	PAIR(30,  R10, R11, R12, R13, R14, R15, R8,  R9,  R9,  R10, R11, R12, R13, R14, R15, R8)
	PAIRM(32, R8,  R9,  R10, R11, R12, R13, R14, R15, R15, R8,  R9,  R10, R11, R12, R13, R14)
	PAIR(34,  R14, R15, R8,  R9,  R10, R11, R12, R13, R13, R14, R15, R8,  R9,  R10, R11, R12)
	PAIR(36,  R12, R13, R14, R15, R8,  R9,  R10, R11, R11, R12, R13, R14, R15, R8,  R9,  R10)
	PAIR(38,  R10, R11, R12, R13, R14, R15, R8,  R9,  R9,  R10, R11, R12, R13, R14, R15, R8)
	PAIR(40,  R8,  R9,  R10, R11, R12, R13, R14, R15, R15, R8,  R9,  R10, R11, R12, R13, R14)
	PAIR(42,  R14, R15, R8,  R9,  R10, R11, R12, R13, R13, R14, R15, R8,  R9,  R10, R11, R12)
	PAIR(44,  R12, R13, R14, R15, R8,  R9,  R10, R11, R11, R12, R13, R14, R15, R8,  R9,  R10)
	PAIR(46,  R10, R11, R12, R13, R14, R15, R8,  R9,  R9,  R10, R11, R12, R13, R14, R15, R8)
	PAIRM(48, R8,  R9,  R10, R11, R12, R13, R14, R15, R15, R8,  R9,  R10, R11, R12, R13, R14)
	PAIR(50,  R14, R15, R8,  R9,  R10, R11, R12, R13, R13, R14, R15, R8,  R9,  R10, R11, R12)
	PAIR(52,  R12, R13, R14, R15, R8,  R9,  R10, R11, R11, R12, R13, R14, R15, R8,  R9,  R10)
	PAIR(54,  R10, R11, R12, R13, R14, R15, R8,  R9,  R9,  R10, R11, R12, R13, R14, R15, R8)
	PAIR(56,  R8,  R9,  R10, R11, R12, R13, R14, R15, R15, R8,  R9,  R10, R11, R12, R13, R14)
	PAIR(58,  R14, R15, R8,  R9,  R10, R11, R12, R13, R13, R14, R15, R8,  R9,  R10, R11, R12)
	PAIR(60,  R12, R13, R14, R15, R8,  R9,  R10, R11, R11, R12, R13, R14, R15, R8,  R9,  R10)
	PAIR(62,  R10, R11, R12, R13, R14, R15, R8,  R9,  R9,  R10, R11, R12, R13, R14, R15, R8)
	PAIRM(64, R8,  R9,  R10, R11, R12, R13, R14, R15, R15, R8,  R9,  R10, R11, R12, R13, R14)
	PAIR(66,  R14, R15, R8,  R9,  R10, R11, R12, R13, R13, R14, R15, R8,  R9,  R10, R11, R12)
	PAIR(68,  R12, R13, R14, R15, R8,  R9,  R10, R11, R11, R12, R13, R14, R15, R8,  R9,  R10)
	PAIR(70,  R10, R11, R12, R13, R14, R15, R8,  R9,  R9,  R10, R11, R12, R13, R14, R15, R8)
	PAIR(72,  R8,  R9,  R10, R11, R12, R13, R14, R15, R15, R8,  R9,  R10, R11, R12, R13, R14)
	PAIR(74,  R14, R15, R8,  R9,  R10, R11, R12, R13, R13, R14, R15, R8,  R9,  R10, R11, R12)
	PAIR(76,  R12, R13, R14, R15, R8,  R9,  R10, R11, R11, R12, R13, R14, R15, R8,  R9,  R10)
	PAIR(78,  R10, R11, R12, R13, R14, R15, R8,  R9,  R9,  R10, R11, R12, R13, R14, R15, R8)

	MOVV	(0*8)(R4), REGTMP
	MOVV	(1*8)(R4), REGTMP1
	MOVV	(2*8)(R4), REGTMP2
	MOVV	(3*8)(R4), REGTMP3
	ADDV	REGTMP, R8
	ADDV	REGTMP1, R9
	ADDV	REGTMP2, R10
	ADDV	REGTMP3, R11
	MOVV	R8, (0*8)(R4)
	MOVV	R9, (1*8)(R4)
	MOVV	R10, (2*8)(R4)
	MOVV	R11, (3*8)(R4)
	MOVV	(4*8)(R4), REGTMP
	MOVV	(5*8)(R4), REGTMP1
	MOVV	(6*8)(R4), REGTMP2
	MOVV	(7*8)(R4), REGTMP3
	ADDV	REGTMP, R12
	ADDV	REGTMP1, R13
	ADDV	REGTMP2, R14
	ADDV	REGTMP3, R15
	MOVV	R12, (4*8)(R4)
	MOVV	R13, (5*8)(R4)
	MOVV	R14, (6*8)(R4)
	MOVV	R15, (7*8)(R4)

	ADDV	$128, R5
	BNE	R5, R25, lloop

lend:
	RET
