// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build s390x,!gccgo,!appengine

#include "go_asm.h"
#include "textflag.h"

// This is an implementation of the ChaCha20 encryption algorithm as
// specified in RFC 7539. It uses vector instructions to compute
// 4 keystream blocks in parallel (256 bytes) which are then XORed
// with the bytes in the input slice.

GLOBL ·constants<>(SB), RODATA|NOPTR, $32
// BSWAP: swap bytes in each 4-byte element
DATA ·constants<>+0x00(SB)/4, $0x03020100
DATA ·constants<>+0x04(SB)/4, $0x07060504
DATA ·constants<>+0x08(SB)/4, $0x0b0a0908
DATA ·constants<>+0x0c(SB)/4, $0x0f0e0d0c
// J0: [j0, j1, j2, j3]
DATA ·constants<>+0x10(SB)/4, $0x61707865
DATA ·constants<>+0x14(SB)/4, $0x3320646e
DATA ·constants<>+0x18(SB)/4, $0x79622d32
DATA ·constants<>+0x1c(SB)/4, $0x6b206574

// EXRL targets:
TEXT ·mvcSrcToBuf(SB), NOFRAME|NOSPLIT, $0
	MVC $1, (R1), (R8)
	RET

TEXT ·mvcBufToDst(SB), NOFRAME|NOSPLIT, $0
	MVC $1, (R8), (R9)
	RET

#define BSWAP V5
#define J0    V6
#define KEY0  V7
#define KEY1  V8
#define NONCE V9
#define CTR   V10
#define M0    V11
#define M1    V12
#define M2    V13
#define M3    V14
#define INC   V15
#define X0    V16
#define X1    V17
#define X2    V18
#define X3    V19
#define X4    V20
#define X5    V21
#define X6    V22
#define X7    V23
#define X8    V24
#define X9    V25
#define X10   V26
#define X11   V27
#define X12   V28
#define X13   V29
#define X14   V30
#define X15   V31

#define NUM_ROUNDS 20

#define ROUND4(a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, d0, d1, d2, d3) \
	VAF    a1, a0, a0  \
	VAF    b1, b0, b0  \
	VAF    c1, c0, c0  \
	VAF    d1, d0, d0  \
	VX     a0, a2, a2  \
	VX     b0, b2, b2  \
	VX     c0, c2, c2  \
	VX     d0, d2, d2  \
	VERLLF $16, a2, a2 \
	VERLLF $16, b2, b2 \
	VERLLF $16, c2, c2 \
	VERLLF $16, d2, d2 \
	VAF    a2, a3, a3  \
	VAF    b2, b3, b3  \
	VAF    c2, c3, c3  \
	VAF    d2, d3, d3  \
	VX     a3, a1, a1  \
	VX     b3, b1, b1  \
	VX     c3, c1, c1  \
	VX     d3, d1, d1  \
	VERLLF $12, a1, a1 \
	VERLLF $12, b1, b1 \
	VERLLF $12, c1, c1 \
	VERLLF $12, d1, d1 \
	VAF    a1, a0, a0  \
	VAF    b1, b0, b0  \
	VAF    c1, c0, c0  \
	VAF    d1, d0, d0  \
	VX     a0, a2, a2  \
	VX     b0, b2, b2  \
	VX     c0, c2, c2  \
	VX     d0, d2, d2  \
	VERLLF $8, a2, a2  \
	VERLLF $8, b2, b2  \
	VERLLF $8, c2, c2  \
	VERLLF $8, d2, d2  \
	VAF    a2, a3, a3  \
	VAF    b2, b3, b3  \
	VAF    c2, c3, c3  \
	VAF    d2, d3, d3  \
	VX     a3, a1, a1  \
	VX     b3, b1, b1  \
	VX     c3, c1, c1  \
	VX     d3, d1, d1  \
	VERLLF $7, a1, a1  \
	VERLLF $7, b1, b1  \
	VERLLF $7, c1, c1  \
	VERLLF $7, d1, d1

#define PERMUTE(mask, v0, v1, v2, v3) \
	VPERM v0, v0, mask, v0 \
	VPERM v1, v1, mask, v1 \
	VPERM v2, v2, mask, v2 \
	VPERM v3, v3, mask, v3

#define ADDV(x, v0, v1, v2, v3) \
	VAF x, v0, v0 \
	VAF x, v1, v1 \
	VAF x, v2, v2 \
	VAF x, v3, v3

#define XORV(off, dst, src, v0, v1, v2, v3) \
	VLM  off(src), M0, M3          \
	PERMUTE(BSWAP, v0, v1, v2, v3) \
	VX   v0, M0, M0                \
	VX   v1, M1, M1                \
	VX   v2, M2, M2                \
	VX   v3, M3, M3                \
	VSTM M0, M3, off(dst)

#define SHUFFLE(a, b, c, d, t, u, v, w) \
	VMRHF a, c, t \ // t = {a[0], c[0], a[1], c[1]}
	VMRHF b, d, u \ // u = {b[0], d[0], b[1], d[1]}
	VMRLF a, c, v \ // v = {a[2], c[2], a[3], c[3]}
	VMRLF b, d, w \ // w = {b[2], d[2], b[3], d[3]}
	VMRHF t, u, a \ // a = {a[0], b[0], c[0], d[0]}
	VMRLF t, u, b \ // b = {a[1], b[1], c[1], d[1]}
	VMRHF v, w, c \ // c = {a[2], b[2], c[2], d[2]}
	VMRLF v, w, d // d = {a[3], b[3], c[3], d[3]}

// func xorKeyStreamVX(dst, src []byte, key *[8]uint32, nonce *[3]uint32, counter *uint32, buf *[256]byte, len *int)
TEXT ·xorKeyStreamVX(SB), NOSPLIT, $0
	MOVD $·constants<>(SB), R1
	MOVD dst+0(FP), R2         // R2=&dst[0]
	LMG  src+24(FP), R3, R4    // R3=&src[0] R4=len(src)
	MOVD key+48(FP), R5        // R5=key
	MOVD nonce+56(FP), R6      // R6=nonce
	MOVD counter+64(FP), R7    // R7=counter
	MOVD buf+72(FP), R8        // R8=buf
	MOVD len+80(FP), R9        // R9=len

	// load BSWAP and J0
	VLM (R1), BSWAP, J0

	// set up tail buffer
	ADD     $-1, R4, R12
	MOVBZ   R12, R12
	CMPUBEQ R12, $255, aligned
	MOVD    R4, R1
	AND     $~255, R1
	MOVD    $(R3)(R1*1), R1
	EXRL    $·mvcSrcToBuf(SB), R12
	MOVD    $255, R0
	SUB     R12, R0
	MOVD    R0, (R9)               // update len

aligned:
	// setup
	MOVD  $95, R0
	VLM   (R5), KEY0, KEY1
	VLL   R0, (R6), NONCE
	VZERO M0
	VLEIB $7, $32, M0
	VSRLB M0, NONCE, NONCE

	// initialize counter values
	VLREPF (R7), CTR
	VZERO  INC
	VLEIF  $1, $1, INC
	VLEIF  $2, $2, INC
	VLEIF  $3, $3, INC
	VAF    INC, CTR, CTR
	VREPIF $4, INC

chacha:
	VREPF $0, J0, X0
	VREPF $1, J0, X1
	VREPF $2, J0, X2
	VREPF $3, J0, X3
	VREPF $0, KEY0, X4
	VREPF $1, KEY0, X5
	VREPF $2, KEY0, X6
	VREPF $3, KEY0, X7
	VREPF $0, KEY1, X8
	VREPF $1, KEY1, X9
	VREPF $2, KEY1, X10
	VREPF $3, KEY1, X11
	VLR   CTR, X12
	VREPF $1, NONCE, X13
	VREPF $2, NONCE, X14
	VREPF $3, NONCE, X15

	MOVD $(NUM_ROUNDS/2), R1

loop:
	ROUND4(X0, X4, X12,  X8, X1, X5, X13,  X9, X2, X6, X14, X10, X3, X7, X15, X11)
	ROUND4(X0, X5, X15, X10, X1, X6, X12, X11, X2, X7, X13, X8,  X3, X4, X14, X9)

	ADD $-1, R1
	BNE loop

	// decrement length
	ADD $-256, R4
	BLT tail

continue:
	// rearrange vectors
	SHUFFLE(X0, X1, X2, X3, M0, M1, M2, M3)
	ADDV(J0, X0, X1, X2, X3)
	SHUFFLE(X4, X5, X6, X7, M0, M1, M2, M3)
	ADDV(KEY0, X4, X5, X6, X7)
	SHUFFLE(X8, X9, X10, X11, M0, M1, M2, M3)
	ADDV(KEY1, X8, X9, X10, X11)
	VAF CTR, X12, X12
	SHUFFLE(X12, X13, X14, X15, M0, M1, M2, M3)
	ADDV(NONCE, X12, X13, X14, X15)

	// increment counters
	VAF INC, CTR, CTR

	// xor keystream with plaintext
	XORV(0*64, R2, R3, X0, X4,  X8, X12)
	XORV(1*64, R2, R3, X1, X5,  X9, X13)
	XORV(2*64, R2, R3, X2, X6, X10, X14)
	XORV(3*64, R2, R3, X3, X7, X11, X15)

	// increment pointers
	MOVD $256(R2), R2
	MOVD $256(R3), R3

	CMPBNE  R4, $0, chacha
	CMPUBEQ R12, $255, return
	EXRL    $·mvcBufToDst(SB), R12 // len was updated during setup

return:
	VSTEF $0, CTR, (R7)
	RET

tail:
	MOVD R2, R9
	MOVD R8, R2
	MOVD R8, R3
	MOVD $0, R4
	JMP  continue
