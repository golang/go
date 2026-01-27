// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

#include "textflag.h"

#ifndef HasZvknha

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
//                 = ((y XOR z) AND x) XOR z
#define SHA256T1(index, e, f, g, h) \
	MOVWU	(index*4)(X18), X8; \
	ADD	X5, h; \
	RORW	$6, e, X6; \
	ADD	X8, h; \
	RORW	$11, e, X7; \
	RORW	$25, e, X8; \
	XOR	X7, X6; \
	XOR	f, g, X5; \
	XOR	X8, X6; \
	AND	e, X5; \
	ADD	X6, h; \
	XOR	g, X5; \
	ADD	h, X5

// Calculate T2 in X6.
//   T2 = BIGSIGMA0(a) + Maj(a, b, c)
//     BIGSIGMA0(x) = ROTR(2,x) XOR ROTR(13,x) XOR ROTR(22,x)
//     Maj(x, y, z) = (x AND y) XOR (x AND z) XOR (y AND z)
//                  = ((y XOR z) AND x) XOR (y AND z)
#define SHA256T2(a, b, c) \
	RORW	$2, a, X6; \
	RORW	$13, a, X7; \
	RORW	$22, a, X8; \
	XOR	X7, X6; \
	XOR	b, c, X9; \
	AND	b, c, X7; \
	AND	a, X9; \
	XOR	X8, X6; \
	XOR	X7, X9; \
	ADD	X9, X6

// Calculate T1 and T2, then e = d + T1 and a = T1 + T2.
// The values for e and a are stored in d and h, ready for rotation.
#define SHA256ROUND(index, a, b, c, d, e, f, g, h) \
	SHA256T1(index, e, f, g, h); \
	SHA256T2(a, b, c); \
	ADD	X5, d; \
	ADD	X6, X5, h

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

	MOV	$·_K(SB), X18		// const table
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


#else

// Index vector for loading {w4, w9, w10, w11} using VLUXEI32V
// Byte offsets: 0, 20, 24, 28 (for w4, w9, w10, w11 from p_base)
DATA	index_w4_w9_w10_w11<>+0(SB)/4, $0
DATA	index_w4_w9_w10_w11<>+4(SB)/4, $20
DATA	index_w4_w9_w10_w11<>+8(SB)/4, $24
DATA	index_w4_w9_w10_w11<>+12(SB)/4, $28
GLOBL	index_w4_w9_w10_w11<>(SB), RODATA, $16


DATA	vreg_mask<>+0(SB)/4, $1
DATA	vreg_mask<>+4(SB)/4, $1
DATA	vreg_mask<>+8(SB)/4, $1
DATA	vreg_mask<>+12(SB)/4, $0
GLOBL	vreg_mask<>(SB), RODATA, $16

// Index vector to reorder final state from scratch layout {f,e,b,a,h,g,d,c}
// into {a,b,c,d,e,f,g,h}
// offsets: a=28, b=24, c=12, d=8, e=4, f=0, g=20, h=16
DATA	index_final<>+0(SB)/4, $12
DATA	index_final<>+4(SB)/4, $8
DATA	index_final<>+8(SB)/4, $28
DATA	index_final<>+12(SB)/4, $24
DATA	index_final<>+16(SB)/4, $4
DATA	index_final<>+20(SB)/4, $0
DATA	index_final<>+24(SB)/4, $20
DATA	index_final<>+28(SB)/4, $16
GLOBL	index_final<>(SB), RODATA, $64


// VSHA2CL expects vs2={a,b,e,f} with element order [f,e,b,a] (index 0,1,2,3)
// Digest layout: h[0]=a(offset 0), h[1]=b(4), h[2]=c(8), h[3]=d(12),
//                h[4]=e(16), h[5]=f(20), h[6]=g(24), h[7]=h(28)
DATA	index_a_b_e_f<>+0(SB)/4, $20   // element[0] = f (offset 20)
DATA	index_a_b_e_f<>+4(SB)/4, $16   // element[1] = e (offset 16)
DATA	index_a_b_e_f<>+8(SB)/4, $4    // element[2] = b (offset 4)
DATA	index_a_b_e_f<>+12(SB)/4, $0   // element[3] = a (offset 0)
GLOBL	index_a_b_e_f<>(SB), RODATA, $16

// VSHA2CL expects vd={c,d,g,h} with element order [h,g,d,c] (index 0,1,2,3)
DATA	index_c_d_g_h<>+0(SB)/4, $28   // element[0] = h (offset 28)
DATA	index_c_d_g_h<>+4(SB)/4, $24   // element[1] = g (offset 24)
DATA	index_c_d_g_h<>+8(SB)/4, $12   // element[2] = d (offset 12)
DATA	index_c_d_g_h<>+12(SB)/4, $8   // element[3] = c (offset 8)
GLOBL	index_c_d_g_h<>(SB), RODATA, $16

// BlockRISCV64WithTrace performs SHA-256 block compression similar to blockGeneric,
// but also saves the intermediate hash state (a, b, c, d, e, f, g, h) for each of the 64 rounds.
//
// func blockRISCV64WithTrace(dig *Digest, p []byte, temp_dig *[64][8]uint32, temp_kword *[64]uint32)
//
// In Go Plan9 assembly, parameters are accessed via FP:
//   dig *Digest: dig+0(FP) (8 bytes on riscv64)
//   p []byte: p_base+8(FP), p_len+16(FP), p_cap+24(FP) (each 8 bytes on riscv64)
//   temp_dig *[64][8]uint32: temp_dig+32(FP) (8 bytes on riscv64)
//   temp_kword *[64]uint32: temp_kword+40(FP) (8 bytes on riscv64)
//
// Stack frame size: $0-48 (no local variables, 48 bytes for parameters: 8+8+8+8+8+8)




// func block(dig *Digest, p []byte)
TEXT ·block(SB),0,$64-32
	MOV	p_base+8(FP), X28	// X28 = input message pointer (using X28 as dedicated register)
    MOV	dig+0(FP), X6	// X6 = digest pointer
    MOV	p_len+16(FP), X30

    SRL	$6, X30
	SLL	$6, X30
	BEQZ	X30, end

    VSETIVLI	$4, E32, M1, TA, MA, X0

blockloop:
    MOV	X28, X5			// Copy from X28 to X5 at the start of each loop iteration

    // First four rounds
    VLE32V		(X5), V1		// V1 = {W[0], W[1], W[2], W[3]} (big-endian)
    VREV8V		V1, V1			// Convert to little-endian

    // V1 contains W[0..3], add k values
    MOV  $·_K+0(SB), X7         // Will add corresponding offset later
    VLE32V		(X7), V31		// V32 = {K[0], K[1], K[2], K[3]}
    VADDVV		V1, V31, V1		// V1 = {W[0]+K[0], W[1]+K[1], W[2]+K[2], W[3]+K[3]}

    // Load a, b, e, f
    MOV  $index_a_b_e_f<>(SB), X7
    VLE32V		(X7), V31
    VLUXEI32V	(X6), V31, V15  // V15 = {a, b, e, f}

    // Load c, d, g, h
    MOV  $index_c_d_g_h<>(SB), X7
    VLE32V		(X7), V31
    VLUXEI32V	(X6), V31, V16  // V16 = {c, d, g, h}


    VSHA2CLVV   V1, V15, V16    // vs1=V1, vs2=V15, vd=V16 -> V16 = new {f,e,b,a
    VSHA2CHVV   V1, V16, V15


    // ---------------- Rounds 4、5、6、7 (use W[4..7] + K[4..7]) ----------------
    // Load W[4..7]
    ADD $16, X5, X9                  // X9 = p_base + 16
    VLE32V     (X9), V1              // big-endian W4..7
    VREV8V     V1, V1                // little-endian
    // Add K[4..7]
    MOV  $·_K+16(SB), X7
    VLE32V     (X7), V31
    VADDVV     V1, V31, V1           // V1 = {w4+K4, w5+K5, w6+K6, w7+K7}

    // VSHA2CLVV: rounds 4,5 (low words W4,W5)
    VSHA2CLVV  V1, V15, V16          // V16 = new {f,e,b,a}

    // VSHA2CHVV: rounds 6,7 (high words W6,W7)
    VSHA2CHVV  V1, V16, V15          // V15 = new {f,e,b,a}

    // ---------------- Rounds 8、9、10、11 (use W[8..11] + K[8..11]) ----------------
    // Load W[8..11]
    ADD $32, X5, X9                  // X9 = p_base + 32
    VLE32V     (X9), V1              // big-endian W8..11
    VREV8V     V1, V1                // little-endian
    // Add K[8..11]
    MOV  $·_K+32(SB), X7
    VLE32V     (X7), V31
    VADDVV     V1, V31, V1           // V1 = {w8+K8, w9+K9, w10+K10, w11+K11}

    // VSHA2CLVV: rounds 8,9 (low words W8,W9)
    VSHA2CLVV  V1, V15, V16          // V16 = new {f,e,b,a}

    // VSHA2CHVV: rounds 10,11 (high words W10,W11)
    VSHA2CHVV  V1, V16, V15          // V15 = new {f,e,b,a}


    // ---------------- Rounds 12、13、14、15 (use W[12..15] + K[12..15]) ----------------
    // Load W[12..15]
    ADD $48, X5, X9                  // X9 = p_base + 48
    VLE32V     (X9), V1              // big-endian W12..15
    VREV8V     V1, V1                // little-endian
    // Add K[12..15]
    MOV  $·_K+48(SB), X7
    VLE32V     (X7), V31
    VADDVV     V1, V31, V1           // V1 = {w12+K12, w13+K13, w14+K14, w15+K15}

    // VSHA2CLVV: rounds 12,13 (low words W12,W13)
    VSHA2CLVV  V1, V15, V16          // V16 = new {f,e,b,a}

    // VSHA2CHVV: rounds 14,15 (high words W14,W15)
    VSHA2CHVV  V1, V16, V15          // V15 = new {f,e,b,a}

    // ---------------- Rounds 16、17、18、19 (use W[16..19] + K[16..19]) ----------------
    // Recompute W16..19 via VSHA2MS (no buffering)
    // Load W[0..3] -> V1
    VLE32V     (X5), V1
    VREV8V     V1, V1
    // Load {W4, W9, W10, W11} -> V2 using index table
    MOV  $index_w4_w9_w10_w11<>(SB), X7
    VLE32V     (X7), V31
    ADD  $16, X5, X9                  // p_base + 16
    VLUXEI32V  (X9), V31, V2
    VREV8V     V2, V2
    // Load W[12..15] -> V3
    ADD  $48, X5, X9
    VLE32V     (X9), V3
    VREV8V     V3, V3
    // Generate W[16..19] into V1 (descending order {W19,W18,W17,W16}) — keep desc for later rounds
    VSHA2MSVV  V3, V2, V1
    // Add K[16..19] into V30 (preserve V1 desc)
    MOV  $·_K+64(SB), X7
    VLE32V     (X7), V31
    VADDVV     V1, V31, V30          // V30 = {w16+K16, w17+K17, w18+K18, w19+K19}

    // VSHA2CLVV: rounds 16,17 (use low words W16,W17)
    VSHA2CLVV  V30, V15, V16          // V16 = new {f,e,b,a}

    // VSHA2CHVV: rounds 18,19 (use high words W18,W19)
    VSHA2CHVV  V30, V16, V15          // V15 = new {f,e,b,a}

    // ---------------- Rounds 20、21、22、23 (use W[20..23] + K[20..23]) ----------------
    // Generate W20..23 via VSHA2MS (use same regs as msg_expand)
    // Load W[4..7] -> V4 (asc)
    ADD $16, X5, X9
    VLE32V     (X9), V4
    VREV8V     V4, V4
    // Load {W8, W13, W14, W15} -> V5 (asc via index)
    MOV  $index_w4_w9_w10_w11<>(SB), X7
    VLE32V     (X7), V31
    ADD $32, X5, X9                  // p_base + 32
    VLUXEI32V  (X9), V31, V5
    VREV8V     V5, V5
    // VSHA2MS: vd=V4 gets {W23,W22,W21,W20} (descending), vs2=V1 (desc W19..16)
    VSHA2MSVV  V1, V5, V4
    // Add K[20..23] into V30 (preserve V1 desc, V4 desc)
    MOV  $·_K+80(SB), X7
    VLE32V     (X7), V31
    VADDVV     V4, V31, V30         // V30 = {w20+K20, w21+K21, w22+K22, w23+K23}

    // VSHA2CLVV: rounds 20,21 (use low words W20,W21)
    VSHA2CLVV  V30, V15, V16          // V16 = new {f,e,b,a}

    // VSHA2CHVV: rounds 22,23 (use high words W22,W23)
    VSHA2CHVV  V30, V16, V15          // V15 = new {f,e,b,a}

    // ---------------- Rounds 24、25、26、27 (use W[24..27] + K[24..27]) ----------------
    // Prepare inputs for VSHA2MS (follow msg_expand regs):
    //  - vs2: V8 should hold W27..W24 (desc) after previous step; rebuild from V4 desc if needed
 
    //  - vs1: merge {W12,W13,W14,?} and {W19,W18,W17,W16} per mask {1,1,1,0}
    ADD	$32, X5, X7		// Constant increased by 16 compared to previous round
	VLE32V		(X7), V6	// Register number increased by 2	
	VREV8V		V6, V6					
	

	MOV	$vreg_mask<>(SB), X7	
	VLE32V		(X7), V0	// Mask {1,1,1,0}
	VMERGEVVM	V3, V1, V0, V7 	// Merge two source operands, select element from first v when mask is 1
		
	VSHA2MSVV	V4, V7, V6		

    // Add K[24..27] into V30 (preserve V6 desc)
    MOV  $·_K+96(SB), X7
    VLE32V     (X7), V31
    VADDVV     V6, V31, V30          // V30 = {w24+K24, w25+K25, w26+K26, w27+K27}

    // VSHA2CLVV: rounds 24,25 (use low words W24,W25)
    VSHA2CLVV  V30, V15, V16          // V16 = new {f,e,b,a}

    // VSHA2CHVV: rounds 26,27 (use high words W26,W27)
    VSHA2CHVV  V30, V16, V15          // V15 = new {f,e,b,a}

    // ---------------- Rounds 28、29、30、31 (use W[28..31] + K[28..31]) ----------------
    // Align registers to msg_expand style for W28..31:
    // vs2: V8 holds W27..W24 (desc) from previous step
    // vs1: merge V4(desc W23..20) and V6(desc W27..24) with mask -> V31 = {W23,W22,W21,W24}
    ADD	$48, X5, X7		// Constant increased by 16 compared to previous round
	VLE32V		(X7), V8	// Register number increased by 2	
	VREV8V		V8, V8					

	VMERGEVVM	V1, V4, V0, V9 	// Merge two source operands, select element from first v when mask is 1	
		
	VSHA2MSVV	V6, V9, V8	
    // Add K[28..31] into V30 (preserve V9)
    MOV  $·_K+112(SB), X7
    VLE32V     (X7), V31
    VADDVV     V8, V31, V30           // V30 = {w28+K28, w29+K29, w30+K30, w31+K31}

    // VSHA2CLVV: rounds 28,29 (use low words W28,W29)
    VSHA2CLVV  V30, V15, V16          // V16 = new {f,e,b,a}

    // VSHA2CHVV: rounds 30,31 (use high words W30,W31)
    VSHA2CHVV  V30, V16, V15          // V15 = new {f,e,b,a}

	// Fifth group
	VMERGEVVM	V4, V6, V0, V31 	// Merge two source operands, select element from first v when mask is 1	
		
	VSHA2MSVV	V8, V31, V1		
	// V1 currently contains {W35, W34, W33, W32} (descending order)
	MOV	$·_K+128(SB), X7
	VLE32V		(X7), V31
	VADDVV		V1, V31, V30		// V30 = {w32+K32, w33+K33, w34+K34, w35+K35}

	// VSHA2CLVV rounds 32,33
	VSHA2CLVV	V30, V15, V16
	
	// VSHA2CHVV rounds 34,35
	VSHA2CHVV	V30, V16, V15


	// Sixth group
	VMERGEVVM	V6, V8, V0, V31 	// Merge two source operands, select element from first v when mask is 1	
		
	VSHA2MSVV	V1, V31, V4		
    // V4 currently contains {W39, W38, W37, W36} (descending order)
	MOV	$·_K+144(SB), X7
	VLE32V		(X7), V31
	VADDVV		V4, V31, V30		// V30 = {w36+K36, w37+K37, w38+K38, w39+K39}
	
	// VSHA2CLVV rounds 36,37
	VSHA2CLVV	V30, V15, V16
	
	// VSHA2CHVV rounds 38,39
	VSHA2CHVV	V30, V16, V15



	// Seventh group
	VMERGEVVM	V8, V1, V0, V31 	// Merge two source operands, select element from first v when mask is 1	
		
	VSHA2MSVV	V4, V31, V6		
	// V6 currently contains {W43, W42, W41, W40} (descending order)
	MOV	$·_K+160(SB), X7
	VLE32V		(X7), V31
	VADDVV		V6, V31, V30		// V30 = {w40+K40, w41+K41, w42+K42, w43+K43}
	
	// VSHA2CLVV rounds 40,41
	VSHA2CLVV	V30, V15, V16
	
	// VSHA2CHVV rounds 42,43
	VSHA2CHVV	V30, V16, V15

	// Eighth group
	VMERGEVVM	V1, V4, V0, V31 	// Merge two source operands, select element from first v when mask is 1	
		
	VSHA2MSVV	V6, V31, V8		
	// V8 currently contains {W47, W46, W45, W44} (descending order)
	MOV	$·_K+176(SB), X7
	VLE32V		(X7), V31
	VADDVV		V8, V31, V30		// V30 = {w44+K44, w45+K45, w46+K46, w47+K47}

	// VSHA2CLVV rounds 44,45
	VSHA2CLVV	V30, V15, V16

	// VSHA2CHVV rounds 46,47
	VSHA2CHVV	V30, V16, V15

	// Ninth group
	VMERGEVVM	V4, V6, V0, V31 	// Merge two source operands, select element from first v when mask is 1	
		
	VSHA2MSVV	V8, V31, V1		
	// V1 currently contains {W51, W50, W49, W48} (descending order)
	MOV	$·_K+192(SB), X7
	VLE32V		(X7), V31
	VADDVV		V1, V31, V30		// V30 = {w48+K48, w49+K49, w50+K50, w51+K51}
	// VSHA2CLVV rounds 48,49
	VSHA2CLVV	V30, V15, V16
	// VSHA2CHVV rounds 50,51
	VSHA2CHVV	V30, V16, V15

	// Tenth group
	VMERGEVVM	V6, V8, V0, V31 	// Merge two source operands, select element from first v when mask is 1	
		
	VSHA2MSVV	V1, V31, V4		
	// V4 currently contains {W55, W54, W53, W52} (descending order)
	MOV	$·_K+208(SB), X7
	VLE32V		(X7), V31
	VADDVV		V4, V31, V30		// V30 = {w52+K52, w53+K53, w54+K54, w55+K55}
	// VSHA2CLVV rounds 52,53
	VSHA2CLVV	V30, V15, V16
	// VSHA2CHVV rounds 54,55
	VSHA2CHVV	V30, V16, V15

	// Eleventh group
	VMERGEVVM	V8, V1, V0, V31 	// Merge two source operands, select element from first v when mask is 1	
		
	VSHA2MSVV	V4, V31, V6		
	// V6 currently contains {W59, W58, W57, W56} (descending order)
	MOV	$·_K+224(SB), X7
	VLE32V		(X7), V31
	VADDVV		V6, V31, V30		// V30 = {w56+K56, w57+K57, w58+K58, w59+K59}
	
	// VSHA2CLVV rounds 56,57
	VSHA2CLVV	V30, V15, V16
	// VSHA2CHVV rounds 58,59
	VSHA2CHVV	V30, V16, V15

	// Twelfth group
	VMERGEVVM	V1, V4, V0, V31 	// Merge two source operands, select element from first v when mask is 1	
		
	VSHA2MSVV	V6, V31, V8		
	// V8 currently contains {W63, W62, W61, W60} (descending order)
	MOV	$·_K+240(SB), X7
	VLE32V		(X7), V31
	VADDVV		V8, V31, V30		// V30 = {w60+K60, w61+K61, w62+K62, w63+K63}
	
	// VSHA2CLVV rounds 60,61
	VSHA2CLVV	V30, V15, V16

	// VSHA2CHVV rounds 62,63
	VSHA2CHVV	V30, V16, V15
	
	// ---------- Final output of a..h (no accumulation) ----------
    VLE32V		(X6), V1
    ADD	$16, X6, X7
    VLE32V		(X7), V2


	// Scratch to reorder: use temp_dig tail
	// Store final state into scratch: [f,e,b,a] then [h,g,d,c]
	VSE32V		V15, (X6)
	VSE32V		V16, (X7)
	// Scalar load to avoid index ambiguity (corrected offsets)
	// Layout in scratch: 0:f,4:e,8:b,12:a,16:h,20:g,24:d,28:c
	// Use indexed gather to reorder scratch {f,e,b,a,h,g,d,c} -> {a,b,c,d,e,f,g,h}
	MOV  $index_final<>(SB), X8
	VLE32V	(X8), V31
	VLUXEI32V (X6), V31, V3	// V3 = {a,b,c,d}
	MOV  $index_final<>+16(SB), X8
	VLE32V	(X8), V31
	VLUXEI32V (X6), V31, V4	// V4 = {e,f,g,h}

    VADDVV		V1, V3, V1
    VADDVV		V2, V4, V2

    VSE32V		V1, (X6)
    VSE32V		V2, (X7)

    SUB $64, X30, X30
    ADD $64, X28, X28		// Update X28 to point to next 64-byte block
    BNEZ    X30, blockloop

end:
	RET

#endif
