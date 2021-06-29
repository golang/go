// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// Load data from "p" and covert it from little endian to big endian (AMR64 is little endian)
#define LOAD_AND_CONVERT_ENDIAN(s) \
VLD1.P	16(R1), [s.B16]; \
VREV64   s.B16, s.B16;

#define SIGMA_ROUND(s, q0, q1, q2, q3) \
VLD1.P	16(R3), [V27.D2]; \
VADD	s.D2, V27.D2, V28.D2; \
VEXT	$8, V28.B16, V28.B16, V28.B16; \
VADD	V28.D2, q3.D2, V28.D2; \
VEXT	$8, q3.B16, q2.B16, V29.B16; \
VEXT	$8, q2.B16, q1.B16, V30.B16; \
SHA512H	V30.D2, V29, V28; \
VSHL	$0, V28.D2, V29.D2; \
SHA512H2	q0.D2, q1, V28; \
VSHL	$0, V28.D2, q3.D2; \
VADD	q1.D2, V29.D2, q1.D2;

#define GAMMA_ROUND(s0, s1, s2, s3, s4) \
VEXT	$8, s3.B16, s2.B16, V30.B16; \
SHA512SU0	s1.D2, s0.D2; \
SHA512SU1	V30.D2, s4.D2, s0.D2;

// func blockNEON(dig *digest, p []byte)
TEXT ·blockNEON(SB), NOSPLIT, $0-32
	MOVD dig+0(FP), R0
	MOVD p_base+8(FP), R1
	MOVD p_len+16(FP), R2
	MOVD	$round_consts(SB), R3

	VLD1.P	32(R0), [V0.D2, V1.D2] // ab, cd
	VLD1	(R0), [V2.D2, V3.D2]   // ef, gh

	VSHL	$0, V0.D2, V23.D2
	VSHL	$0, V1.D2, V24.D2
	VSHL	$0, V2.D2, V25.D2
	VSHL	$0, V3.D2, V26.D2	

loop:
	LOAD_AND_CONVERT_ENDIAN(V4)
	SIGMA_ROUND(V4, V0, V1, V2, V3) // 0
	LOAD_AND_CONVERT_ENDIAN(V5)
	SIGMA_ROUND(V5, V3, V0, V1, V2) // 2
	LOAD_AND_CONVERT_ENDIAN(V6)
	SIGMA_ROUND(V6, V2, V3, V0, V1) // 4
	LOAD_AND_CONVERT_ENDIAN(V7)
	SIGMA_ROUND(V7, V1, V2, V3, V0) // 6
	LOAD_AND_CONVERT_ENDIAN(V8)
	SIGMA_ROUND(V8, V0, V1, V2, V3) // 8
	LOAD_AND_CONVERT_ENDIAN(V9)
	SIGMA_ROUND(V9, V3, V0, V1, V2) // 10
	LOAD_AND_CONVERT_ENDIAN(V10)
	SIGMA_ROUND(V10, V2, V3, V0, V1) // 12
	LOAD_AND_CONVERT_ENDIAN(V11)
	SIGMA_ROUND(V11, V1, V2, V3, V0) // 14

	GAMMA_ROUND(V4, V5, V8, V9, V11)
	SIGMA_ROUND(V4, V0, V1, V2, V3) // 16
	GAMMA_ROUND(V5, V6, V9, V10, V4)
	SIGMA_ROUND(V5, V3, V0, V1, V2) // 18
	GAMMA_ROUND(V6, V7, V10, V11, V5)
	SIGMA_ROUND(V6, V2, V3, V0, V1) // 20
	GAMMA_ROUND(V7, V8, V11, V4, V6)
	SIGMA_ROUND(V7, V1, V2, V3, V0) // 22
	GAMMA_ROUND(V8, V9, V4, V5, V7)
	SIGMA_ROUND(V8, V0, V1, V2, V3) // 24
	GAMMA_ROUND(V9, V10, V5, V6, V8)
	SIGMA_ROUND(V9, V3, V0, V1, V2) // 26
	GAMMA_ROUND(V10, V11, V6, V7, V9)
	SIGMA_ROUND(V10, V2, V3, V0, V1) // 28
	GAMMA_ROUND(V11, V4, V7, V8, V10)
	SIGMA_ROUND(V11, V1, V2, V3, V0) // 30

	GAMMA_ROUND(V4, V5, V8, V9, V11)
	SIGMA_ROUND(V4, V0, V1, V2, V3) // 32
	GAMMA_ROUND(V5, V6, V9, V10, V4)
	SIGMA_ROUND(V5, V3, V0, V1, V2) // 34
	GAMMA_ROUND(V6, V7, V10, V11, V5)
	SIGMA_ROUND(V6, V2, V3, V0, V1) // 36
	GAMMA_ROUND(V7, V8, V11, V4, V6)
	SIGMA_ROUND(V7, V1, V2, V3, V0) // 38
	GAMMA_ROUND(V8, V9, V4, V5, V7)
	SIGMA_ROUND(V8, V0, V1, V2, V3) // 40
	GAMMA_ROUND(V9, V10, V5, V6, V8)
	SIGMA_ROUND(V9, V3, V0, V1, V2) // 42
	GAMMA_ROUND(V10, V11, V6, V7, V9)
	SIGMA_ROUND(V10, V2, V3, V0, V1) // 44
	GAMMA_ROUND(V11, V4, V7, V8, V10)
	SIGMA_ROUND(V11, V1, V2, V3, V0) // 46

	GAMMA_ROUND(V4, V5, V8, V9, V11)
	SIGMA_ROUND(V4, V0, V1, V2, V3) // 48
	GAMMA_ROUND(V5, V6, V9, V10, V4)
	SIGMA_ROUND(V5, V3, V0, V1, V2) // 50
	GAMMA_ROUND(V6, V7, V10, V11, V5)
	SIGMA_ROUND(V6, V2, V3, V0, V1) // 52
	GAMMA_ROUND(V7, V8, V11, V4, V6)
	SIGMA_ROUND(V7, V1, V2, V3, V0) // 54
	GAMMA_ROUND(V8, V9, V4, V5, V7)
	SIGMA_ROUND(V8, V0, V1, V2, V3) // 56
	GAMMA_ROUND(V9, V10, V5, V6, V8)
	SIGMA_ROUND(V9, V3, V0, V1, V2) // 58
	GAMMA_ROUND(V10, V11, V6, V7, V9)
	SIGMA_ROUND(V10, V2, V3, V0, V1) // 60
	GAMMA_ROUND(V11, V4, V7, V8, V10)
	SIGMA_ROUND(V11, V1, V2, V3, V0) // 62

	GAMMA_ROUND(V4, V5, V8, V9, V11)
	SIGMA_ROUND(V4, V0, V1, V2, V3) // 64
	GAMMA_ROUND(V5, V6, V9, V10, V4)
	SIGMA_ROUND(V5, V3, V0, V1, V2) // 66
	GAMMA_ROUND(V6, V7, V10, V11, V5)
	SIGMA_ROUND(V6, V2, V3, V0, V1) // 68
	GAMMA_ROUND(V7, V8, V11, V4, V6)
	SIGMA_ROUND(V7, V1, V2, V3, V0) // 70
	GAMMA_ROUND(V8, V9, V4, V5, V7)
	SIGMA_ROUND(V8, V0, V1, V2, V3) // 72
	GAMMA_ROUND(V9, V10, V5, V6, V8)
	SIGMA_ROUND(V9, V3, V0, V1, V2) // 74
	GAMMA_ROUND(V10, V11, V6, V7, V9)
	SIGMA_ROUND(V10, V2, V3, V0, V1) // 76
	GAMMA_ROUND(V11, V4, V7, V8, V10)
	SIGMA_ROUND(V11, V1, V2, V3, V0) // 78

	VADD	V0.D2, V23.D2, V23.D2
	VADD	V1.D2, V24.D2, V24.D2
	VADD	V2.D2, V25.D2, V25.D2
	VADD	V3.D2, V26.D2, V26.D2

	// reset round constant table pointer to the pointer of beginning of the table
	SUBS	$640, R3, R3

	// check the whether consume all the input
	SUBS	$128, R2, R2
	CBNZ	R2, loop

	SUBS	$32, R0, R0
	VST1.P	[V23.D2, V24.D2], 32(R0)
	VST1	[V25.D2, V26.D2], (R0)
	RET

DATA	round_consts<>+0x000(SB)/8, $0x428a2f98d728ae22
DATA	round_consts<>+0x008(SB)/8, $0x7137449123ef65cd
DATA	round_consts<>+0x010(SB)/8, $0xb5c0fbcfec4d3b2f
DATA	round_consts<>+0x018(SB)/8, $0xe9b5dba58189dbbc
DATA	round_consts<>+0x020(SB)/8, $0x3956c25bf348b538
DATA	round_consts<>+0x028(SB)/8, $0x59f111f1b605d019
DATA	round_consts<>+0x030(SB)/8, $0x923f82a4af194f9b
DATA	round_consts<>+0x038(SB)/8, $0xab1c5ed5da6d8118
DATA	round_consts<>+0x040(SB)/8, $0xd807aa98a3030242
DATA	round_consts<>+0x048(SB)/8, $0x12835b0145706fbe
DATA	round_consts<>+0x050(SB)/8, $0x243185be4ee4b28c
DATA	round_consts<>+0x058(SB)/8, $0x550c7dc3d5ffb4e2
DATA	round_consts<>+0x060(SB)/8, $0x72be5d74f27b896f
DATA	round_consts<>+0x068(SB)/8, $0x80deb1fe3b1696b1
DATA	round_consts<>+0x070(SB)/8, $0x9bdc06a725c71235
DATA	round_consts<>+0x078(SB)/8, $0xc19bf174cf692694
DATA	round_consts<>+0x080(SB)/8, $0xe49b69c19ef14ad2
DATA	round_consts<>+0x088(SB)/8, $0xefbe4786384f25e3
DATA	round_consts<>+0x090(SB)/8, $0x0fc19dc68b8cd5b5
DATA	round_consts<>+0x098(SB)/8, $0x240ca1cc77ac9c65
DATA	round_consts<>+0x0A0(SB)/8, $0x2de92c6f592b0275
DATA	round_consts<>+0x0A8(SB)/8, $0x4a7484aa6ea6e483
DATA	round_consts<>+0x0B0(SB)/8, $0x5cb0a9dcbd41fbd4
DATA	round_consts<>+0x0B8(SB)/8, $0x76f988da831153b5
DATA	round_consts<>+0x0C0(SB)/8, $0x983e5152ee66dfab
DATA	round_consts<>+0x0C8(SB)/8, $0xa831c66d2db43210
DATA	round_consts<>+0x0D0(SB)/8, $0xb00327c898fb213f
DATA	round_consts<>+0x0D8(SB)/8, $0xbf597fc7beef0ee4
DATA	round_consts<>+0x0E0(SB)/8, $0xc6e00bf33da88fc2
DATA	round_consts<>+0x0E8(SB)/8, $0xd5a79147930aa725
DATA	round_consts<>+0x0F0(SB)/8, $0x06ca6351e003826f
DATA	round_consts<>+0x0F8(SB)/8, $0x142929670a0e6e70
DATA	round_consts<>+0x100(SB)/8, $0x27b70a8546d22ffc
DATA	round_consts<>+0x108(SB)/8, $0x2e1b21385c26c926
DATA	round_consts<>+0x110(SB)/8, $0x4d2c6dfc5ac42aed
DATA	round_consts<>+0x118(SB)/8, $0x53380d139d95b3df
DATA	round_consts<>+0x120(SB)/8, $0x650a73548baf63de
DATA	round_consts<>+0x128(SB)/8, $0x766a0abb3c77b2a8
DATA	round_consts<>+0x130(SB)/8, $0x81c2c92e47edaee6
DATA	round_consts<>+0x138(SB)/8, $0x92722c851482353b
DATA	round_consts<>+0x140(SB)/8, $0xa2bfe8a14cf10364
DATA	round_consts<>+0x148(SB)/8, $0xa81a664bbc423001
DATA	round_consts<>+0x150(SB)/8, $0xc24b8b70d0f89791
DATA	round_consts<>+0x158(SB)/8, $0xc76c51a30654be30
DATA	round_consts<>+0x160(SB)/8, $0xd192e819d6ef5218
DATA	round_consts<>+0x168(SB)/8, $0xd69906245565a910
DATA	round_consts<>+0x170(SB)/8, $0xf40e35855771202a
DATA	round_consts<>+0x178(SB)/8, $0x106aa07032bbd1b8
DATA	round_consts<>+0x180(SB)/8, $0x19a4c116b8d2d0c8
DATA	round_consts<>+0x188(SB)/8, $0x1e376c085141ab53
DATA	round_consts<>+0x190(SB)/8, $0x2748774cdf8eeb99
DATA	round_consts<>+0x198(SB)/8, $0x34b0bcb5e19b48a8
DATA	round_consts<>+0x1A0(SB)/8, $0x391c0cb3c5c95a63
DATA	round_consts<>+0x1A8(SB)/8, $0x4ed8aa4ae3418acb
DATA	round_consts<>+0x1B0(SB)/8, $0x5b9cca4f7763e373
DATA	round_consts<>+0x1B8(SB)/8, $0x682e6ff3d6b2b8a3
DATA	round_consts<>+0x1C0(SB)/8, $0x748f82ee5defb2fc
DATA	round_consts<>+0x1C8(SB)/8, $0x78a5636f43172f60
DATA	round_consts<>+0x1D0(SB)/8, $0x84c87814a1f0ab72
DATA	round_consts<>+0x1D8(SB)/8, $0x8cc702081a6439ec
DATA	round_consts<>+0x1E0(SB)/8, $0x90befffa23631e28
DATA	round_consts<>+0x1E8(SB)/8, $0xa4506cebde82bde9
DATA	round_consts<>+0x1F0(SB)/8, $0xbef9a3f7b2c67915
DATA	round_consts<>+0x1F8(SB)/8, $0xc67178f2e372532b
DATA	round_consts<>+0x200(SB)/8, $0xca273eceea26619c
DATA	round_consts<>+0x208(SB)/8, $0xd186b8c721c0c207
DATA	round_consts<>+0x210(SB)/8, $0xeada7dd6cde0eb1e
DATA	round_consts<>+0x218(SB)/8, $0xf57d4f7fee6ed178
DATA	round_consts<>+0x220(SB)/8, $0x06f067aa72176fba
DATA	round_consts<>+0x228(SB)/8, $0x0a637dc5a2c898a6
DATA	round_consts<>+0x230(SB)/8, $0x113f9804bef90dae
DATA	round_consts<>+0x238(SB)/8, $0x1b710b35131c471b
DATA	round_consts<>+0x240(SB)/8, $0x28db77f523047d84
DATA	round_consts<>+0x248(SB)/8, $0x32caab7b40c72493
DATA	round_consts<>+0x250(SB)/8, $0x3c9ebe0a15c9bebc
DATA	round_consts<>+0x258(SB)/8, $0x431d67c49c100d4c
DATA	round_consts<>+0x260(SB)/8, $0x4cc5d4becb3e42b6
DATA	round_consts<>+0x268(SB)/8, $0x597f299cfc657e2a
DATA	round_consts<>+0x270(SB)/8, $0x5fcb6fab3ad6faec
DATA	round_consts<>+0x278(SB)/8, $0x6c44198c4a475817
GLOBL	round_consts(SB), (NOPTR+RODATA), $640
