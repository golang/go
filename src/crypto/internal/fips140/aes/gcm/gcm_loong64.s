// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

// AES-GCM for loong64.
//
// LoongArch (LA464/LA664) has no carry-less multiply instruction, so
// GHASH is a scalar port of ghash.go: nine 32x32->64 carry-less
// multiplications using the BearSSL "bit hole" technique (MULV with
// masked inputs), Karatsuba recombination, and polynomial reduction.
// AES block encryption uses the same VSHUFB-based S-box as
// aes_loong64.s (SUBBYTES/MIXCOLUMNS macros are algorithm-identical;
// sbox data is duplicated locally because this is a separate package).
//
// productTable layout (loong64-specific, 84 bytes used of 256):
//   +0  h0, h1, h2, h3      (uint32, host order, as in ghash.go)
//   +16 h0^h1, h2^h3, h0^h2, h1^h3, (h0^h2)^(h1^h3)
//   +36 MULT128 scratch: six 64-bit intermediate products
//     (r00, r01 at +36/+44; r10, r11 at +52/+60; r20, r21 at +68/+76)
//
// enc[] round keys are stored in word order (32-bit BE words), so
// aesEnc applies a byteSwap32 VSHUFB to each round key before XOR.

#include "textflag.h"

// -----------------------------------------------------------------------
// Tables (local to this file)
// -----------------------------------------------------------------------

// sboxG: AES encryption S-box (identical contents to sbox0_0 in
// aes_loong64.s).
DATA	sboxG<>+0x00(SB)/8, $0xc56f6bf27b777c63
DATA	sboxG<>+0x08(SB)/8, $0x76abd7fe2b670130
DATA	sboxG<>+0x10(SB)/8, $0xf04759fa7dc982ca
DATA	sboxG<>+0x18(SB)/8, $0xc072a49cafa2d4ad
DATA	sboxG<>+0x20(SB)/8, $0xccf73f362693fdb7
DATA	sboxG<>+0x28(SB)/8, $0x1531d871f1e5a534
DATA	sboxG<>+0x30(SB)/8, $0x9a059618c323c704
DATA	sboxG<>+0x38(SB)/8, $0x75b227ebe2801207
DATA	sboxG<>+0x40(SB)/8, $0xa05a6e1b1a2c8309
DATA	sboxG<>+0x48(SB)/8, $0x842fe329b3d63b52
DATA	sboxG<>+0x50(SB)/8, $0x5bb1fc20ed00d153
DATA	sboxG<>+0x58(SB)/8, $0xcf584c4a39becb6a
DATA	sboxG<>+0x60(SB)/8, $0x85334d43fbaaefd0
DATA	sboxG<>+0x68(SB)/8, $0xa89f3c507f02f945
DATA	sboxG<>+0x70(SB)/8, $0xf5389d928f40a351
DATA	sboxG<>+0x78(SB)/8, $0xd2f3ff1021dab6bc
DATA	sboxG<>+0x80(SB)/8, $0x1744975fec130ccd
DATA	sboxG<>+0x88(SB)/8, $0x73195d643d7ea7c4
DATA	sboxG<>+0x90(SB)/8, $0x88902a22dc4f8160
DATA	sboxG<>+0x98(SB)/8, $0xdb0b5ede14b8ee46
DATA	sboxG<>+0xa0(SB)/8, $0x5c2406490a3a32e0
DATA	sboxG<>+0xa8(SB)/8, $0x79e4959162acd3c2
DATA	sboxG<>+0xb0(SB)/8, $0xa94ed58d6d37c8e7
DATA	sboxG<>+0xb8(SB)/8, $0x08ae7a65eaf4566c
DATA	sboxG<>+0xc0(SB)/8, $0xc6b4a61c2e2578ba
DATA	sboxG<>+0xc8(SB)/8, $0x8a8bbd4b1f74dde8
DATA	sboxG<>+0xd0(SB)/8, $0x0ef6034866b53e70
DATA	sboxG<>+0xd8(SB)/8, $0x9e1dc186b9573561
DATA	sboxG<>+0xe0(SB)/8, $0x948ed9691198f8e1
DATA	sboxG<>+0xe8(SB)/8, $0xdf2855cee9871e9b
DATA	sboxG<>+0xf0(SB)/8, $0x6842e6bf0d89a18c
DATA	sboxG<>+0xf8(SB)/8, $0x16bb54b00f2d9941
GLOBL	sboxG<>(SB), (NOPTR+RODATA), $256

DATA	shiftRowsG<>+0x00(SB)/8, $0x030e09040f0a0500
DATA	shiftRowsG<>+0x08(SB)/8, $0x0b06010c07020d08
GLOBL	shiftRowsG<>(SB), (NOPTR+RODATA), $16

DATA	rot1G<>+0x00(SB)/8, $0x0407060500030201
DATA	rot1G<>+0x08(SB)/8, $0x0c0f0e0d080b0a09
GLOBL	rot1G<>(SB), (NOPTR+RODATA), $16

DATA	rot2G<>+0x00(SB)/8, $0x0504070601000302
DATA	rot2G<>+0x08(SB)/8, $0x0d0c0f0e09080b0a
GLOBL	rot2G<>(SB), (NOPTR+RODATA), $16

DATA	rot3G<>+0x00(SB)/8, $0x0605040702010003
DATA	rot3G<>+0x08(SB)/8, $0x0e0d0c0f0a09080b
GLOBL	rot3G<>(SB), (NOPTR+RODATA), $16

// bit-hole masks for the carry-less multiply (see ghash.go ghashMul).
DATA	mask1111<>+0(SB)/8, $0x1111111111111111
GLOBL	mask1111<>(SB), (NOPTR+RODATA), $8

DATA    byteSwap32G<>+0x00(SB)/8, $0x0405060700010203
DATA    byteSwap32G<>+0x08(SB)/8, $0x0c0d0e0f08090a0b
GLOBL   byteSwap32G<>(SB), (NOPTR+RODATA), $16

// -----------------------------------------------------------------------
// AES macros (algorithm-identical to aes_loong64.s)
// -----------------------------------------------------------------------

#define XTIME(src, dst, tmp) \
	VSLLB	$1, src, dst; \
	VSRAB	$7, src, tmp; \
	VANDB	$0x1b, tmp, tmp; \
	VXORV	tmp, dst, dst

#define MIXCOLUMNS(src, out, r1, r2, r3, t1, t2, t3) \
	VSHUFB	r1, src, src, t1; \
	VXORV	src, t1, t2; \
	VSHUFB	r2, src, src, t3; \
	VXORV	t2, t3, out; \
	VSHUFB	r3, src, src, t3; \
	VXORV	out, t3, out; \
	XTIME(t2, t1, t3); \
	VXORV	src, out, out; \
	VXORV	out, t1, out

// SUBBYTES implements the AES S-box via 4 table-lookup pairs.
// Masks: V31=bit7, V30=bit6, V29=bit5; V26=low5 index.
// Each chunk pair produces a result via VSHUFB+VBITSELV:
//   r01 = chunk0/1, r23 = chunk2/3, r45 = chunk4/5, r67 = chunk6/7
//   r0123 = M6 ? r23 : r01; r4567 = M6 ? r67 : r45
//   out   = M7 ? r4567 : r0123
#define SUBBYTES(src, sbox_ptr, out) \
	VANDB	$0xe0, src, V25; \
	VSRAB	$7, V25, V31; \
	VSLLB	$1, V25, V24; \
	VSRAB	$7, V24, V30; \
	VSLLB	$2, V25, V24; \
	VSRAB	$7, V24, V29; \
	VANDB	$0x1f, src, V26; \
	VMOVQ	0(sbox_ptr), V16; \
	VMOVQ	16(sbox_ptr), V17; \
	VMOVQ	32(sbox_ptr), V18; \
	VMOVQ	48(sbox_ptr), V19; \
	VSHUFB	V26, V16, V17, V28; \
	VSHUFB	V26, V18, V19, V27; \
	VBITSELV	V29, V27, V28, V20; \
	VMOVQ	64(sbox_ptr), V16; \
	VMOVQ	80(sbox_ptr), V17; \
	VMOVQ	96(sbox_ptr), V18; \
	VMOVQ	112(sbox_ptr), V19; \
	VSHUFB	V26, V16, V17, V28; \
	VSHUFB	V26, V18, V19, V27; \
	VBITSELV	V29, V27, V28, V21; \
	VBITSELV	V30, V21, V20, V22; \
	VMOVQ	128(sbox_ptr), V16; \
	VMOVQ	144(sbox_ptr), V17; \
	VMOVQ	160(sbox_ptr), V18; \
	VMOVQ	176(sbox_ptr), V19; \
	VSHUFB	V26, V16, V17, V28; \
	VSHUFB	V26, V18, V19, V27; \
	VBITSELV	V29, V27, V28, V20; \
	VMOVQ	192(sbox_ptr), V16; \
	VMOVQ	208(sbox_ptr), V17; \
	VMOVQ	224(sbox_ptr), V18; \
	VMOVQ	240(sbox_ptr), V19; \
	VSHUFB	V26, V16, V17, V28; \
	VSHUFB	V26, V18, V19, V27; \
	VBITSELV	V29, V27, V28, V21; \
	VBITSELV	V30, V21, V20, V23; \
	VBITSELV	V31, V23, V22, out

// -----------------------------------------------------------------------
// GHASH scalar macros — direct port of ghash.go.
//
// GHASHMUL(x, y, out): out = carryless(x,y), x/y are 32-bit values in
// scalar regs. Clobbers R13-R21, R24-R27. Port of ghashMul().
//
// xm[i] = x & (0x1111111111111111 << i) -> R24..R27; ym[i] -> R15..R18.
// z[i] = (xm0*ym[i] ^ xm1*ym[i+3] ^ xm2*ym[i+2] ^ xm3*ym[i+1])
//         & (0x1111.. << i), accumulated into out.
// -----------------------------------------------------------------------
#define GHASHMUL(xr, yr, out) \
	MOVV	mask1111<>+0(SB), R20; \
	AND	R20, xr, R24; \
	SLLV	$1, R20, R21; AND R21, xr, R25; \
	SLLV	$2, R20, R21; AND R21, xr, R26; \
	SLLV	$3, R20, R21; AND R21, xr, R27; \
	AND	R20, yr, R15; \
	AND	R21, yr, R18; \
	SLLV	$1, R20, R21; AND R21, yr, R16; \
	SLLV	$2, R20, R21; AND R21, yr, R17; \
	MOVV	R20, R19; \
	MULV	R24, R15, out; \
	MULV	R25, R18, R14; XOR R14, out; \
	MULV	R26, R17, R14; XOR R14, out; \
	MULV	R27, R16, R14; XOR R14, out; \
	AND	R19, out, out; \
	MULV	R24, R16, R14; \
	MULV	R25, R15, R13; XOR R13, R14; \
	MULV	R26, R18, R13; XOR R13, R14; \
	MULV	R27, R17, R13; XOR R13, R14; \
	SLLV	$1, R19, R13; AND R13, R14, R14; OR R14, out; \
	MULV	R24, R17, R14; \
	MULV	R25, R16, R13; XOR R13, R14; \
	MULV	R26, R15, R13; XOR R13, R14; \
	MULV	R27, R18, R13; XOR R13, R14; \
	SLLV	$2, R19, R13; AND R13, R14, R14; OR R14, out; \
	MULV	R24, R18, R14; \
	MULV	R25, R17, R13; XOR R13, R14; \
	MULV	R26, R16, R13; XOR R13, R14; \
	MULV	R27, R15, R13; XOR R13, R14; \
	SLLV	$3, R19, R13; AND R13, R14, R14; OR R14, out

// MULT128: y = y (*) H in GF(2^128). y[0..3] in R8-R11, R7 = productTable.
// Clobbers R12-R21, R23-R28, R31. Writes intermediate products to
// productTable+36..76. Port of the ghash() block body.
//
// Three GHASHMUL groups (zLo/zHi/zSum) are spilled to productTable:
//   group 0 (y0,y1 x h0,h1)  -> r00@36, r01@44
//   group 1 (y2,y3 x h2,h3)  -> r10@52, r11@60
//   group 2 (y0^y2, y1^y3)   -> r20@68, r21@76
// Then r20 ^= r00^r10, r21 ^= r01^r11, and the halves are folded:
// r01 ^= r20 (R13), r10 ^= r21 (R14).
// z = r << 1 into R18,R19,R21,R23, followed by the GF(2^128) reduction
// (lw ^ lw>>1 ^ lw>>2 ^ lw>>7 and carries <<63|<<62|<<57).
// Result: R8,R9 = z2 lo/hi, R10,R11 = z3 lo/hi.
#define MULT128() \
	MOVWU	0(R7), R13;  GHASHMUL(R8,  R13, R23); \
	MOVWU	4(R7), R13;  GHASHMUL(R9,  R13, R28); \
	MOVWU	16(R7), R14; XOR R9, R8, R13; GHASHMUL(R13, R14, R31); \
	XOR	R23, R31; XOR R28, R31; \
	SLLV	$32, R31, R13; XOR R13, R23; MOVV R23, 36(R7); \
	SRLV	$32, R31, R13; XOR R13, R28; MOVV R28, 44(R7); \
	MOVWU	8(R7), R13;  GHASHMUL(R10, R13, R23); \
	MOVWU	12(R7), R13; GHASHMUL(R11, R13, R28); \
	MOVWU	20(R7), R14; XOR R11, R10, R13; GHASHMUL(R13, R14, R31); \
	XOR	R23, R31; XOR R28, R31; \
	SLLV	$32, R31, R13; XOR R13, R23; MOVV R23, 52(R7); \
	SRLV	$32, R31, R13; XOR R13, R28; MOVV R28, 60(R7); \
	MOVWU	24(R7), R14; XOR R10, R8, R13; GHASHMUL(R13, R14, R23); \
	MOVWU	28(R7), R14; XOR R11, R9, R13; GHASHMUL(R13, R14, R28); \
	MOVWU	32(R7), R14; \
	XOR	R10, R8, R13; \
	XOR	R11, R9, R12; \
	XOR	R12, R13; \
	GHASHMUL(R13, R14, R31); \
	XOR	R23, R31; XOR R28, R31; \
	SLLV	$32, R31, R13; XOR R13, R23; MOVV R23, 68(R7); \
	SRLV	$32, R31, R13; XOR R13, R28; MOVV R28, 76(R7); \
	MOVV	36(R7), R12; \
	MOVV	44(R7), R13; \
	MOVV	52(R7), R14; \
	MOVV	60(R7), R15; \
	MOVV	68(R7), R16; \
	MOVV	76(R7), R17; \
	XOR	R12, R16; XOR R14, R16; \
	XOR	R13, R17; XOR R15, R17; \
	XOR	R16, R13; \
	XOR	R17, R14; \
	SLLV	$1, R12, R18; \
	SLLV	$1, R13, R19; SRLV $63, R12, R20; OR R20, R19; \
	SLLV	$1, R15, R23; SRLV $63, R14, R20; OR R20, R23; \
	SLLV	$1, R14, R21; SRLV $63, R13, R20; OR R20, R21; \
	SRLV	$1, R18, R20; XOR R20, R18, R20; SRLV $2, R18, R24; XOR R24, R20; \
	SRLV	$7, R18, R24; XOR R24, R20; XOR R20, R21; \
	SLLV	$63, R18, R20; SLLV $62, R18, R24; XOR R24, R20; \
	SLLV	$57, R18, R24; XOR R24, R20; XOR R20, R19; \
	SRLV	$1, R19, R20; XOR R20, R19, R20; SRLV $2, R19, R24; XOR R24, R20; \
	SRLV	$7, R19, R24; XOR R24, R20; XOR R20, R23; \
	SLLV	$63, R19, R20; SLLV $62, R19, R24; XOR R24, R20; \
	SLLV	$57, R19, R24; XOR R24, R20; XOR R20, R21; \
	MOVWU	R21, R8;  SRLV $32, R21, R9; \
	MOVWU	R23, R10; SRLV $32, R23, R11

// GHASHSTEP: XOR the 16-byte block at R4 into y and multiply by H.
#define GHASHSTEP() \
	MOVWU	0(R4), R12;  REVB2W R12, R12; XOR R12, R11; \
	MOVWU	4(R4), R12;  REVB2W R12, R12; XOR R12, R10; \
	MOVWU	8(R4), R12;  REVB2W R12, R12; XOR R12, R9; \
	MOVWU	12(R4), R12; REVB2W R12, R12; XOR R12, R8; \
	MULT128()

// -----------------------------------------------------------------------
// aesEnc: encrypt one block. R4=nr, R5=xk (word order; keys byte-swapped
// per round via V9), V0 in/out.
// Leaf helper; clobbers R24-R26, V4-V9, V15-V31.
// -----------------------------------------------------------------------
TEXT aesEnc<>(SB), NOSPLIT, $0-0
	MOVV	$sboxG<>(SB), R24
	MOVV	$shiftRowsG<>(SB), R25
	VMOVQ	(R25), V7
        MOVV    $byteSwap32G<>(SB), R25
        VMOVQ   (R25), V9
	MOVV	$rot1G<>(SB), R25
	VMOVQ	(R25), V4
	MOVV	$rot2G<>(SB), R25
	VMOVQ	(R25), V5
	MOVV	$rot3G<>(SB), R25
	VMOVQ	(R25), V6

	VMOVQ	(R5), V8
	VXORV	V8, V0, V0
	ADDV	$16, R5

	MOVV	R4, R26
	ADDV	$-1, R26
Lgcm_aes_loop:
	SUBBYTES(V0, R24, V0)
	VSHUFB	V7, V0, V0, V0
	MIXCOLUMNS(V0, V15, V4, V5, V6, V10, V11, V12)
	VMOVQ	(R5), V8
	ADDV	$16, R5
	VXORV	V8, V15, V0
	ADDV	$-1, R26
	BNE	R26, R0, Lgcm_aes_loop

	SUBBYTES(V0, R24, V0)
	VSHUFB	V7, V0, V0, V0
	VMOVQ	(R5), V8
	VXORV	V8, V0, V0
	RET

// -----------------------------------------------------------------------
// func gcmAesInit(productTable *[256]byte, ks []uint32)
// -----------------------------------------------------------------------
TEXT ·gcmAesInit(SB), NOSPLIT, $32-32
	MOVV	R1, 24(R3)		// save return address
	MOVV	productTable+0(FP), R7
	MOVV	ks_base+8(FP), R5
	MOVV	ks_len+16(FP), R4
	SRLV	$2, R4, R4
	ADDV	$-1, R4			// nr = ks_len/4 - 1

	// H = AES(key, 0^128)
	VXORV	V0, V0, V0
	CALL	aesEnc<>(SB)
	VMOVQ	V0, 8(R3)		// spill H to our stack frame

	// h[3-i] = BEU32(H[i*4:])  ->  h0=BEU32(H[12:16]) ... h3=BEU32(H[0:4])
	MOVWU	(8+12)(R3), R8;  REVB2W R8, R8	// h0
	MOVWU	(8+8)(R3), R9;   REVB2W R9, R9	// h1
	MOVWU	(8+4)(R3), R10;  REVB2W R10, R10	// h2
	MOVWU	(8+0)(R3), R11;  REVB2W R11, R11	// h3
	MOVW	R8, 0(R7)
	MOVW	R9, 4(R7)
	MOVW	R10, 8(R7)
	MOVW	R11, 12(R7)
	XOR	R9, R8, R12;  MOVW R12, 16(R7)	// h0^h1
	XOR	R11, R10, R12; MOVW R12, 20(R7)	// h2^h3
	XOR	R10, R8, R12; MOVW R12, 24(R7)	// h0^h2
	XOR	R11, R9, R12; MOVW R12, 28(R7)	// h1^h3
	MOVW	24(R7), R12
	MOVW	28(R7), R13
	XOR	R13, R12; MOVW R12, 32(R7)	// (h0^h2)^(h1^h3)
	MOVV	24(R3), R1			// restore return address
	RET

// -----------------------------------------------------------------------
// func gcmAesData(productTable *[256]byte, data []byte, T *[16]byte)
// -----------------------------------------------------------------------
TEXT ·gcmAesData(SB), NOSPLIT, $32-40
	MOVV	productTable+0(FP), R7
	MOVV	data_base+8(FP), R4
	MOVV	data_len+16(FP), R6
	MOVV	T+32(FP), R5

	// y[3-i] = BEU32(T[i*4:])
	MOVWU	12(R5), R8;  REVB2W R8, R8
	MOVWU	8(R5), R9;   REVB2W R9, R9
	MOVWU	4(R5), R10;  REVB2W R10, R10
	MOVWU	0(R5), R11;  REVB2W R11, R11

data_blocks:
	SGTU	$16, R6, R13
	BNE	R13, data_tail
	SUBV	$16, R6
	GHASHSTEP()
	ADDV	$16, R4
	JMP	data_blocks

data_tail:
	BEQ	R6, data_done
	// zero-pad the partial block in our stack frame
	MOVV	R0, 8(R3)
	MOVV	R0, 16(R3)
	MOVV	R3, R14
	ADDV	$8, R14
	MOVV	R4, R15
data_tail_copy:
	MOVB	(R15), R13
	MOVB	R13, (R14)
	ADDV	$1, R15
	ADDV	$1, R14
	ADDV	$-1, R6
	BNE	R6, R0, data_tail_copy
	MOVV	R3, R4
	ADDV	$8, R4			// R4 = &partialBlock
	GHASHSTEP()

data_done:
	// T[i*4:] = BEU32(y[3-i])
	REVB2W	R11, R12; MOVW R12, 0(R5)
	REVB2W	R10, R12; MOVW R12, 4(R5)
	REVB2W	R9, R12;  MOVW R12, 8(R5)
	REVB2W	R8, R12;  MOVW R12, 12(R5)
	RET

// -----------------------------------------------------------------------
// func gcmAesFinish(productTable *[256]byte, tagMask, T *[16]byte,
//                   pLen, dLen uint64)
// -----------------------------------------------------------------------
TEXT ·gcmAesFinish(SB), NOSPLIT, $0-40
	MOVV	productTable+0(FP), R7
	MOVV	tagMask+8(FP), R6
	MOVV	T+16(FP), R5
	MOVV	pLen+24(FP), R14
	MOVV	dLen+32(FP), R15

	// lenBlock = BE64(dLen*8) || BE64(pLen*8), matching the amd64
	// gcmAesFinish convention (dLen = AAD len first; amd64 stores the
	// same block in its byte-reversed internal representation).
	// As host u32s: BEU32(blk[0:4]) = dLen*8 >> 32, etc.
	SLLV	$3, R14, R14
	SLLV	$3, R15, R15

	// load y from T (BE word order as in gcmAesData)
	MOVWU	12(R5), R8;  REVB2W R8, R8
	MOVWU	8(R5), R9;   REVB2W R9, R9
	MOVWU	4(R5), R10;  REVB2W R10, R10
	MOVWU	0(R5), R11;  REVB2W R11, R11

	// y ^= lenBlock
        SRLV    $32, R15, R12;   XOR R12, R11   // y[3] ^= hi(dLen*8)  block[0:4]
        MOVWU   R15, R12;        XOR R12, R10   // y[2] ^= lo(dLen*8)  block[4:8]
        SRLV    $32, R14, R12;   XOR R12, R9    // y[1] ^= hi(pLen*8)  block[8:12]
        MOVWU   R14, R12;        XOR R12, R8    // y[0] ^= lo(pLen*8)  block[12:16]

	MULT128()

	// tag = y ^ tagMask
	MOVWU	0(R6), R12;  REVB2W R12, R12; XOR R12, R11
	MOVWU	4(R6), R12;  REVB2W R12, R12; XOR R12, R10
	MOVWU	8(R6), R12;  REVB2W R12, R12; XOR R12, R9
	MOVWU	12(R6), R12; REVB2W R12, R12; XOR R12, R8

	REVB2W	R11, R12; MOVW R12, 0(R5)
	REVB2W	R10, R12; MOVW R12, 4(R5)
	REVB2W	R9, R12;  MOVW R12, 8(R5)
	REVB2W	R8, R12;  MOVW R12, 12(R5)
	RET

// helper: encrypt blocks, GHASH ciphertext, ctr++
// Frame layout (offsets from R3):
//   8(R3)=src ptr   16(R3)=dst ptr   24(R3)=remaining len
//   32(R3)=ctr ptr  40(R3)=T ptr     48(R3)=ks base
//   56(R3)=nr       64(R3)=productTable
//   72(R3)-87(R3): 16-byte keystream scratch buffer (tail path)
//   88(R3)-103(R3): 16-byte zero-padded GHASH input buffer (tail path)
//   120(R3): saved R1 (return address)
// y[0..3] stay in R8-R11 across CALL aesEnc<> (aesEnc uses only
// R4, R5, R24-R26 and V registers).
TEXT gcmCryptEnc<>(SB), NOSPLIT, $128-0
	MOVV	R1, 120(R3)
	// ---- spill incoming state to frame ----
	MOVV	R4, 8(R3)	// src
	MOVV	R5, 16(R3)	// dst
	MOVV	R6, 24(R3)	// len
	MOVV	R13, 32(R3)	// ctr
	MOVV	R14, 40(R3)	// T
	MOVV	R16, 48(R3)	// ks base
	MOVV	R15, 56(R3)	// nr
	MOVV	R7, 64(R3)	// productTable

	// y from T (BE word order)
	MOVWU	12(R14), R8;  REVB2W R8, R8
	MOVWU	8(R14), R9;   REVB2W R9, R9
	MOVWU	4(R14), R10;  REVB2W R10, R10
	MOVWU	0(R14), R11;  REVB2W R11, R11

Lenc_block_loop:
	MOVV	24(R3), R6
	SGTU	$16, R6, R12			// R12 = (R6 < 16)
	BEQ	R12, R0, Lenc_have_blocks	// R6 >= 16, jmp
	BNE	R6, R0, Lenc_tail
	JMP	Lenc_done

Lenc_have_blocks:
        // ---- ctr++ FIRST (BE low 32 bits): first block uses inc32(J0) ----
        MOVV    32(R3), R4
        MOVWU   12(R4), R12
        REVB2W  R12, R12
        ADDV    $1, R12
        REVB2W  R12, R12
        MOVW    R12, 12(R4)

	// ---- load ctr into V0 (counter block, BE) ----
	MOVV	32(R3), R4
	VMOVQ	(R4), V0

	// ---- encrypt counter: aesEnc(nr, ks, V0) ----
	MOVV	56(R3), R4	// nr
	MOVV	48(R3), R5	// xk
	CALL	aesEnc<>(SB)	// V0 = AES(ctr)

	// ---- load plaintext block, XOR with AES(ctr), store ----
	MOVV	8(R3), R4
	MOVV	16(R3), R5
	VMOVQ	(R4), V1	// plaintext
	VXORV	V0, V1, V1	// ciphertext
	VMOVQ	V1, (R5)	// store to dst

	// ---- GHASH ciphertext block ----
	MOVV	16(R3), R4	// dst = ciphertext just written
	MOVV	64(R3), R7	// productTable
	GHASHSTEP()

	// ---- advance pointers, decrement len ----
	MOVV	8(R3), R4;  ADDV $16, R4;  MOVV R4, 8(R3)
	MOVV	16(R3), R5; ADDV $16, R5;  MOVV R5, 16(R3)
	MOVV	24(R3), R6; ADDV $-16, R6; MOVV R6, 24(R3)

	JMP	Lenc_block_loop

Lenc_tail:
        // ---- ctr++ first, then encrypt ----
        MOVV    32(R3), R4
        MOVWU   12(R4), R12
        REVB2W  R12, R12
        ADDV    $1, R12
        REVB2W  R12, R12
        MOVW    R12, 12(R4)

	// Remaining len is 1..15 (R6 = 24(R3) != 0 and < 16).
	//
	// Semantics (must match amd64 encBlocks tail + generic):
        //   1. ctr++              -- first block uses inc32(J0)
        //   2. keystream = AES(ctr)
        //   3. dst[i] = src[i] ^ keystream[i]   (i < len, byte-wise)
        //   4. GHASH(zero-padded ciphertext tail)

	// ---- encrypt counter ----
	MOVV	32(R3), R4
	VMOVQ	(R4), V0	// V0 = counter block
	MOVV	56(R3), R4	// nr
	MOVV	48(R3), R5	// xk
	CALL	aesEnc<>(SB)	// V0 = AES(ctr) keystream
	VMOVQ	V0, 72(R3)	// stash keystream on stack

	// ---- zero the 16-byte GHASH input buffer at 88(R3) ----
	MOVV	R0, 88(R3)
	MOVV	R0, 96(R3)

	// ---- byte loop: dst[i] = src[i] ^ ks[i]; ghashbuf[i] = dst[i] ----
	MOVV	8(R3), R4	// src
	MOVV	16(R3), R5	// dst
	MOVV	24(R3), R6	// len (1..15)
	MOVV	R3, R16
	ADDV	$72, R16	// keystream buffer
	MOVV	R3, R17
	ADDV	$88, R17	// ghash buffer

Lenc_tail_xor:
	MOVB	(R4), R12
	MOVB	(R16), R13
	XOR	R13, R12, R12	// ct byte = pt ^ keystream
	MOVB	R12, (R5)	// store to dst
	MOVB	R12, (R17)	// copy ct byte into ghash buffer
	ADDV	$1, R4
	ADDV	$1, R5
	ADDV	$1, R16
	ADDV	$1, R17
	ADDV	$-1, R6
	BNE	R6, R0, Lenc_tail_xor

	// ---- GHASH the zero-padded ciphertext block ----
	MOVV	R3, R4
	ADDV	$88, R4		// R4 = &ghash buffer
	MOVV	64(R3), R7	// productTable
	GHASHSTEP()

Lenc_done:
	// write y back to T
	MOVV	40(R3), R14
	REVB2W	R11, R12; MOVW R12, 0(R14)
	REVB2W	R10, R12; MOVW R12, 4(R14)
	REVB2W	R9, R12;  MOVW R12, 8(R14)
	REVB2W	R8, R12;  MOVW R12, 12(R14)

	MOVV	120(R3), R1	// restore return address
	RET

// helper: decrypt blocks, GHASH ciphertext (src), ctr++
// All loop state lives on our stack frame because CALL aesEnc<>
// clobbers R4, R5, R24-R26 and vector regs.
// Frame layout (offsets from R3):
//   8(R3)=src ptr   16(R3)=dst ptr   24(R3)=remaining len
//   32(R3)=ctr ptr  40(R3)=T ptr     48(R3)=ks base
//   56(R3)=nr       64(R3)=productTable
//   72(R3)-87(R3): keystream scratch   88(R3)-103(R3): ghash pad buffer
//   120(R3): saved R1 (return address)
// y[0..3] stay in R8-R11 across CALL aesEnc<> and are written back to
// T before returning (same convention as gcmCryptEnc).
TEXT gcmCryptDec<>(SB), NOSPLIT, $128-0
	MOVV	R1, 120(R3)		// save return address

	// ---- spill incoming state to frame ----
	MOVV	R4, 8(R3)		// src
	MOVV	R5, 16(R3)		// dst
	MOVV	R6, 24(R3)		// len
	MOVV	R13, 32(R3)		// ctr
	MOVV	R14, 40(R3)		// T
	MOVV	R16, 48(R3)		// ks base
	MOVV	R15, 56(R3)		// nr
	MOVV	R7, 64(R3)		// productTable

        // y[3-i] = BEU32(T[i*4:])
        MOVWU   12(R14), R8;  REVB2W R8, R8
        MOVWU   8(R14), R9;   REVB2W R9, R9
        MOVWU   4(R14), R10;  REVB2W R10, R10
        MOVWU   0(R14), R11;  REVB2W R11, R11

Ldec_block_loop:
	MOVV	24(R3), R6
	BEQ	R6, R0, Ldec_done
	SGTU	$16, R6, R12		// R12 = (R6 < 16)
	BNE	R12, R0, Ldec_tail

	// ---- reload state ----
	MOVV	8(R3), R4		// src ptr
	MOVV	64(R3), R7		// productTable

	// ---- GHASH ciphertext (src) BEFORE decrypting ----
	GHASHSTEP()			// reads (R4), y in R8-R11

        // ---- ctr++ (BE low 32 bits): first block uses inc32(J0) ----
        MOVV    32(R3), R13
        MOVWU   12(R13), R12
        REVB2W  R12, R12
        ADDV    $1, R12
        REVB2W  R12, R12
        MOVW    R12, 12(R13)

	// ---- encrypt counter -> V0 keystream ----
	MOVV	56(R3), R4		// nr
	MOVV	48(R3), R5		// xk
	VMOVQ	(R13), V0		// counter block
	CALL	aesEnc<>(SB)		// clobbers R4, R5, R24-R26, V regs

	// ---- dst = src ^ keystream ----
	MOVV	8(R3), R4
	VMOVQ	(R4), V1
	VXORV	V1, V0, V0
	MOVV	16(R3), R5
	VMOVQ	V0, (R5)

	// ---- advance pointers, decrement len ----
	MOVV	8(R3), R4;  ADDV $16, R4;  MOVV R4, 8(R3)
	MOVV	16(R3), R5; ADDV $16, R5;  MOVV R5, 16(R3)
	MOVV	24(R3), R6; ADDV $-16, R6; MOVV R6, 24(R3)

	JMP	Ldec_block_loop

Ldec_tail:
	// 0 < R6 < 16: R6 is at 24(R3)
	// ---- 1. zero the ghash pad buffer at 88(R3) ----
	// (the keystream buffer at 72(R3) needs no zeroing: it is fully
	// overwritten by the VMOVQ below)
	MOVV	R0, 88(R3)
	MOVV	R0, 96(R3)

	// ---- 2. copy R6 ciphertext bytes into pad buffer 88(R3)
	//        AND GHASH them (ciphertext = input for dec) ----
	MOVV	8(R3), R23		// src tail ptr
	MOVV	R3, R20; ADDV $88, R20	// &pad buffer
	MOVV	24(R3), R21		// tail len
Ldec_tail_copy:
	MOVB	(R23), R12
	MOVB	R12, (R20)
	ADDV	$1, R23; ADDV $1, R20; ADDV $-1, R21
	BNE	R21, R0, Ldec_tail_copy

	// ---- 3. GHASH the zero-padded ciphertext block ----
	MOVV	R3, R4; ADDV $88, R4
	MOVV	64(R3), R7
	GHASHSTEP()

        // ---- ctr++ BEFORE encrypt ----
        MOVV    32(R3), R13
        MOVWU   12(R13), R12
        REVB2W  R12, R12
        ADDV    $1, R12
        REVB2W  R12, R12
        MOVW    R12, 12(R13)

	// ---- 4. encrypt current ctr -> keystream on stack ----
	MOVV	56(R3), R4		// nr
	MOVV	48(R3), R5		// xk
	VMOVQ	(R13), V0
	CALL	aesEnc<>(SB)
	VMOVQ	V0, 72(R3)		// keystream buffer

	// ---- 5. dst[i] = src[i] ^ keystream[i], i < R6 ----
	MOVV	8(R3), R16		// src tail
	MOVV	16(R3), R17		// dst tail
	MOVV	R3, R20; ADDV $72, R20	// &keystream
	MOVV	24(R3), R21		// tail len
Ldec_tail_xor:
	MOVB	(R16), R12
	MOVB	(R20), R18
	XOR	R18, R12
	MOVB	R12, (R17)
	ADDV	$1, R16
	ADDV	$1, R17
	ADDV	$1, R20
	ADDV	$-1, R21
	BNE	R21, R0, Ldec_tail_xor

Ldec_done:
	// write y back to T
	MOVV	40(R3), R14
	REVB2W	R11, R12; MOVW R12, 0(R14)
	REVB2W	R10, R12; MOVW R12, 4(R14)
	REVB2W	R9, R12;  MOVW R12, 8(R14)
	REVB2W	R8, R12;  MOVW R12, 12(R14)

	MOVV	120(R3), R1		// restore return address
	RET

// -----------------------------------------------------------------------
// func gcmAesEnc(productTable *[256]byte, dst, src []byte,
//                ctr, T *[16]byte, ks []uint32)
// -----------------------------------------------------------------------
TEXT ·gcmAesEnc(SB), NOSPLIT, $16-96
	MOVV	R1, 8(R3)		// save return address
	MOVV	productTable+0(FP), R7
	MOVV	dst_base+8(FP), R5
	MOVV	src_base+32(FP), R4
	MOVV	src_len+40(FP), R6
	MOVV	ctr+56(FP), R13
	MOVV	T+64(FP), R14
	MOVV	ks_base+72(FP), R16
	MOVV	ks_len+80(FP), R15
	SRLV	$2, R15, R15
	ADDV	$-1, R15		// nr = ks_len/4 - 1

	CALL	gcmCryptEnc<>(SB)

	MOVV	8(R3), R1		// restore return address
	RET

// -----------------------------------------------------------------------
// func gcmAesDec(productTable *[256]byte, dst, src []byte,
//                ctr, T *[16]byte, ks []uint32)
// -----------------------------------------------------------------------
TEXT ·gcmAesDec(SB), NOSPLIT, $16-96
	MOVV	R1, 8(R3)		// save return address
	MOVV	productTable+0(FP), R7
	MOVV	dst_base+8(FP), R5
	MOVV	src_base+32(FP), R4
	MOVV	src_len+40(FP), R6
	MOVV	ctr+56(FP), R13
	MOVV	T+64(FP), R14
	MOVV	ks_base+72(FP), R16
	MOVV	ks_len+80(FP), R15
	SRLV	$2, R15, R15
	ADDV	$-1, R15

	CALL	gcmCryptDec<>(SB)

	RET
