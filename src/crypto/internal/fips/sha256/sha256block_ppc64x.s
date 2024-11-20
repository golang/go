// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (ppc64 || ppc64le) && !purego

// Based on CRYPTOGAMS code with the following comment:
// # ====================================================================
// # Written by Andy Polyakov <appro@openssl.org> for the OpenSSL
// # project. The module is, however, dual licensed under OpenSSL and
// # CRYPTOGAMS licenses depending on where you obtain it. For further
// # details see http://www.openssl.org/~appro/cryptogams/.
// # ====================================================================

#include "textflag.h"

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

#define CTX	R3
#define INP	R4
#define END	R5
#define TBL	R6 // Pointer into kcon table
#define LEN	R9
#define TEMP	R12

#define TBL_STRT	R7 // Pointer to start of kcon table.

#define R_x000	R0
#define R_x010	R8
#define R_x020	R10
#define R_x030	R11
#define R_x040	R14
#define R_x050	R15
#define R_x060	R16
#define R_x070	R17
#define R_x080	R18
#define R_x090	R19
#define R_x0a0	R20
#define R_x0b0	R21
#define R_x0c0	R22
#define R_x0d0	R23
#define R_x0e0	R24
#define R_x0f0	R25
#define R_x100	R26
#define R_x110	R27


// V0-V7 are A-H
// V8-V23 are used for the message schedule
#define KI	V24
#define FUNC	V25
#define S0	V26
#define S1	V27
#define s0	V28
#define s1	V29
#define LEMASK	V31 // Permutation control register for little endian

// 4 copies of each Kt, to fill all 4 words of a vector register
DATA  ·kcon+0x000(SB)/8, $0x428a2f98428a2f98
DATA  ·kcon+0x008(SB)/8, $0x428a2f98428a2f98
DATA  ·kcon+0x010(SB)/8, $0x7137449171374491
DATA  ·kcon+0x018(SB)/8, $0x7137449171374491
DATA  ·kcon+0x020(SB)/8, $0xb5c0fbcfb5c0fbcf
DATA  ·kcon+0x028(SB)/8, $0xb5c0fbcfb5c0fbcf
DATA  ·kcon+0x030(SB)/8, $0xe9b5dba5e9b5dba5
DATA  ·kcon+0x038(SB)/8, $0xe9b5dba5e9b5dba5
DATA  ·kcon+0x040(SB)/8, $0x3956c25b3956c25b
DATA  ·kcon+0x048(SB)/8, $0x3956c25b3956c25b
DATA  ·kcon+0x050(SB)/8, $0x59f111f159f111f1
DATA  ·kcon+0x058(SB)/8, $0x59f111f159f111f1
DATA  ·kcon+0x060(SB)/8, $0x923f82a4923f82a4
DATA  ·kcon+0x068(SB)/8, $0x923f82a4923f82a4
DATA  ·kcon+0x070(SB)/8, $0xab1c5ed5ab1c5ed5
DATA  ·kcon+0x078(SB)/8, $0xab1c5ed5ab1c5ed5
DATA  ·kcon+0x080(SB)/8, $0xd807aa98d807aa98
DATA  ·kcon+0x088(SB)/8, $0xd807aa98d807aa98
DATA  ·kcon+0x090(SB)/8, $0x12835b0112835b01
DATA  ·kcon+0x098(SB)/8, $0x12835b0112835b01
DATA  ·kcon+0x0A0(SB)/8, $0x243185be243185be
DATA  ·kcon+0x0A8(SB)/8, $0x243185be243185be
DATA  ·kcon+0x0B0(SB)/8, $0x550c7dc3550c7dc3
DATA  ·kcon+0x0B8(SB)/8, $0x550c7dc3550c7dc3
DATA  ·kcon+0x0C0(SB)/8, $0x72be5d7472be5d74
DATA  ·kcon+0x0C8(SB)/8, $0x72be5d7472be5d74
DATA  ·kcon+0x0D0(SB)/8, $0x80deb1fe80deb1fe
DATA  ·kcon+0x0D8(SB)/8, $0x80deb1fe80deb1fe
DATA  ·kcon+0x0E0(SB)/8, $0x9bdc06a79bdc06a7
DATA  ·kcon+0x0E8(SB)/8, $0x9bdc06a79bdc06a7
DATA  ·kcon+0x0F0(SB)/8, $0xc19bf174c19bf174
DATA  ·kcon+0x0F8(SB)/8, $0xc19bf174c19bf174
DATA  ·kcon+0x100(SB)/8, $0xe49b69c1e49b69c1
DATA  ·kcon+0x108(SB)/8, $0xe49b69c1e49b69c1
DATA  ·kcon+0x110(SB)/8, $0xefbe4786efbe4786
DATA  ·kcon+0x118(SB)/8, $0xefbe4786efbe4786
DATA  ·kcon+0x120(SB)/8, $0x0fc19dc60fc19dc6
DATA  ·kcon+0x128(SB)/8, $0x0fc19dc60fc19dc6
DATA  ·kcon+0x130(SB)/8, $0x240ca1cc240ca1cc
DATA  ·kcon+0x138(SB)/8, $0x240ca1cc240ca1cc
DATA  ·kcon+0x140(SB)/8, $0x2de92c6f2de92c6f
DATA  ·kcon+0x148(SB)/8, $0x2de92c6f2de92c6f
DATA  ·kcon+0x150(SB)/8, $0x4a7484aa4a7484aa
DATA  ·kcon+0x158(SB)/8, $0x4a7484aa4a7484aa
DATA  ·kcon+0x160(SB)/8, $0x5cb0a9dc5cb0a9dc
DATA  ·kcon+0x168(SB)/8, $0x5cb0a9dc5cb0a9dc
DATA  ·kcon+0x170(SB)/8, $0x76f988da76f988da
DATA  ·kcon+0x178(SB)/8, $0x76f988da76f988da
DATA  ·kcon+0x180(SB)/8, $0x983e5152983e5152
DATA  ·kcon+0x188(SB)/8, $0x983e5152983e5152
DATA  ·kcon+0x190(SB)/8, $0xa831c66da831c66d
DATA  ·kcon+0x198(SB)/8, $0xa831c66da831c66d
DATA  ·kcon+0x1A0(SB)/8, $0xb00327c8b00327c8
DATA  ·kcon+0x1A8(SB)/8, $0xb00327c8b00327c8
DATA  ·kcon+0x1B0(SB)/8, $0xbf597fc7bf597fc7
DATA  ·kcon+0x1B8(SB)/8, $0xbf597fc7bf597fc7
DATA  ·kcon+0x1C0(SB)/8, $0xc6e00bf3c6e00bf3
DATA  ·kcon+0x1C8(SB)/8, $0xc6e00bf3c6e00bf3
DATA  ·kcon+0x1D0(SB)/8, $0xd5a79147d5a79147
DATA  ·kcon+0x1D8(SB)/8, $0xd5a79147d5a79147
DATA  ·kcon+0x1E0(SB)/8, $0x06ca635106ca6351
DATA  ·kcon+0x1E8(SB)/8, $0x06ca635106ca6351
DATA  ·kcon+0x1F0(SB)/8, $0x1429296714292967
DATA  ·kcon+0x1F8(SB)/8, $0x1429296714292967
DATA  ·kcon+0x200(SB)/8, $0x27b70a8527b70a85
DATA  ·kcon+0x208(SB)/8, $0x27b70a8527b70a85
DATA  ·kcon+0x210(SB)/8, $0x2e1b21382e1b2138
DATA  ·kcon+0x218(SB)/8, $0x2e1b21382e1b2138
DATA  ·kcon+0x220(SB)/8, $0x4d2c6dfc4d2c6dfc
DATA  ·kcon+0x228(SB)/8, $0x4d2c6dfc4d2c6dfc
DATA  ·kcon+0x230(SB)/8, $0x53380d1353380d13
DATA  ·kcon+0x238(SB)/8, $0x53380d1353380d13
DATA  ·kcon+0x240(SB)/8, $0x650a7354650a7354
DATA  ·kcon+0x248(SB)/8, $0x650a7354650a7354
DATA  ·kcon+0x250(SB)/8, $0x766a0abb766a0abb
DATA  ·kcon+0x258(SB)/8, $0x766a0abb766a0abb
DATA  ·kcon+0x260(SB)/8, $0x81c2c92e81c2c92e
DATA  ·kcon+0x268(SB)/8, $0x81c2c92e81c2c92e
DATA  ·kcon+0x270(SB)/8, $0x92722c8592722c85
DATA  ·kcon+0x278(SB)/8, $0x92722c8592722c85
DATA  ·kcon+0x280(SB)/8, $0xa2bfe8a1a2bfe8a1
DATA  ·kcon+0x288(SB)/8, $0xa2bfe8a1a2bfe8a1
DATA  ·kcon+0x290(SB)/8, $0xa81a664ba81a664b
DATA  ·kcon+0x298(SB)/8, $0xa81a664ba81a664b
DATA  ·kcon+0x2A0(SB)/8, $0xc24b8b70c24b8b70
DATA  ·kcon+0x2A8(SB)/8, $0xc24b8b70c24b8b70
DATA  ·kcon+0x2B0(SB)/8, $0xc76c51a3c76c51a3
DATA  ·kcon+0x2B8(SB)/8, $0xc76c51a3c76c51a3
DATA  ·kcon+0x2C0(SB)/8, $0xd192e819d192e819
DATA  ·kcon+0x2C8(SB)/8, $0xd192e819d192e819
DATA  ·kcon+0x2D0(SB)/8, $0xd6990624d6990624
DATA  ·kcon+0x2D8(SB)/8, $0xd6990624d6990624
DATA  ·kcon+0x2E0(SB)/8, $0xf40e3585f40e3585
DATA  ·kcon+0x2E8(SB)/8, $0xf40e3585f40e3585
DATA  ·kcon+0x2F0(SB)/8, $0x106aa070106aa070
DATA  ·kcon+0x2F8(SB)/8, $0x106aa070106aa070
DATA  ·kcon+0x300(SB)/8, $0x19a4c11619a4c116
DATA  ·kcon+0x308(SB)/8, $0x19a4c11619a4c116
DATA  ·kcon+0x310(SB)/8, $0x1e376c081e376c08
DATA  ·kcon+0x318(SB)/8, $0x1e376c081e376c08
DATA  ·kcon+0x320(SB)/8, $0x2748774c2748774c
DATA  ·kcon+0x328(SB)/8, $0x2748774c2748774c
DATA  ·kcon+0x330(SB)/8, $0x34b0bcb534b0bcb5
DATA  ·kcon+0x338(SB)/8, $0x34b0bcb534b0bcb5
DATA  ·kcon+0x340(SB)/8, $0x391c0cb3391c0cb3
DATA  ·kcon+0x348(SB)/8, $0x391c0cb3391c0cb3
DATA  ·kcon+0x350(SB)/8, $0x4ed8aa4a4ed8aa4a
DATA  ·kcon+0x358(SB)/8, $0x4ed8aa4a4ed8aa4a
DATA  ·kcon+0x360(SB)/8, $0x5b9cca4f5b9cca4f
DATA  ·kcon+0x368(SB)/8, $0x5b9cca4f5b9cca4f
DATA  ·kcon+0x370(SB)/8, $0x682e6ff3682e6ff3
DATA  ·kcon+0x378(SB)/8, $0x682e6ff3682e6ff3
DATA  ·kcon+0x380(SB)/8, $0x748f82ee748f82ee
DATA  ·kcon+0x388(SB)/8, $0x748f82ee748f82ee
DATA  ·kcon+0x390(SB)/8, $0x78a5636f78a5636f
DATA  ·kcon+0x398(SB)/8, $0x78a5636f78a5636f
DATA  ·kcon+0x3A0(SB)/8, $0x84c8781484c87814
DATA  ·kcon+0x3A8(SB)/8, $0x84c8781484c87814
DATA  ·kcon+0x3B0(SB)/8, $0x8cc702088cc70208
DATA  ·kcon+0x3B8(SB)/8, $0x8cc702088cc70208
DATA  ·kcon+0x3C0(SB)/8, $0x90befffa90befffa
DATA  ·kcon+0x3C8(SB)/8, $0x90befffa90befffa
DATA  ·kcon+0x3D0(SB)/8, $0xa4506ceba4506ceb
DATA  ·kcon+0x3D8(SB)/8, $0xa4506ceba4506ceb
DATA  ·kcon+0x3E0(SB)/8, $0xbef9a3f7bef9a3f7
DATA  ·kcon+0x3E8(SB)/8, $0xbef9a3f7bef9a3f7
DATA  ·kcon+0x3F0(SB)/8, $0xc67178f2c67178f2
DATA  ·kcon+0x3F8(SB)/8, $0xc67178f2c67178f2
DATA  ·kcon+0x400(SB)/8, $0x0000000000000000
DATA  ·kcon+0x408(SB)/8, $0x0000000000000000

#ifdef GOARCH_ppc64le
DATA  ·kcon+0x410(SB)/8, $0x1011121310111213 // permutation control vectors
DATA  ·kcon+0x418(SB)/8, $0x1011121300010203
DATA  ·kcon+0x420(SB)/8, $0x1011121310111213
DATA  ·kcon+0x428(SB)/8, $0x0405060700010203
DATA  ·kcon+0x430(SB)/8, $0x1011121308090a0b
DATA  ·kcon+0x438(SB)/8, $0x0405060700010203
#else
DATA  ·kcon+0x410(SB)/8, $0x1011121300010203
DATA  ·kcon+0x418(SB)/8, $0x1011121310111213 // permutation control vectors
DATA  ·kcon+0x420(SB)/8, $0x0405060700010203
DATA  ·kcon+0x428(SB)/8, $0x1011121310111213
DATA  ·kcon+0x430(SB)/8, $0x0001020304050607
DATA  ·kcon+0x438(SB)/8, $0x08090a0b10111213
#endif

GLOBL ·kcon(SB), RODATA, $1088

#define SHA256ROUND0(a, b, c, d, e, f, g, h, xi, idx) \
	VSEL		g, f, e, FUNC; \
	VSHASIGMAW	$15, e, $1, S1; \
	VADDUWM		xi, h, h; \
	VSHASIGMAW	$0, a, $1, S0; \
	VADDUWM		FUNC, h, h; \
	VXOR		b, a, FUNC; \
	VADDUWM		S1, h, h; \
	VSEL		b, c, FUNC, FUNC; \
	VADDUWM		KI, g, g; \
	VADDUWM		h, d, d; \
	VADDUWM		FUNC, S0, S0; \
	LVX		(TBL)(idx), KI; \
	VADDUWM		S0, h, h

#define SHA256ROUND1(a, b, c, d, e, f, g, h, xi, xj, xj_1, xj_9, xj_14, idx) \
	VSHASIGMAW	$0, xj_1, $0, s0; \
	VSEL		g, f, e, FUNC; \
	VSHASIGMAW	$15, e, $1, S1; \
	VADDUWM		xi, h, h; \
	VSHASIGMAW	$0, a, $1, S0; \
	VSHASIGMAW	$15, xj_14, $0, s1; \
	VADDUWM		FUNC, h, h; \
	VXOR		b, a, FUNC; \
	VADDUWM		xj_9, xj, xj; \
	VADDUWM		S1, h, h; \
	VSEL		b, c, FUNC, FUNC; \
	VADDUWM		KI, g, g; \
	VADDUWM		h, d, d; \
	VADDUWM		FUNC, S0, S0; \
	VADDUWM		s0, xj, xj; \
	LVX		(TBL)(idx), KI; \
	VADDUWM		S0, h, h; \
	VADDUWM		s1, xj, xj

#ifdef GOARCH_ppc64le
#define VPERMLE(va,vb,vc,vt) VPERM va, vb, vc, vt
#else
#define VPERMLE(va,vb,vc,vt)
#endif

// func blockPOWER(dig *Digest, p []byte)
TEXT ·blockPOWER(SB),0,$0-32
	MOVD	dig+0(FP), CTX
	MOVD	p_base+8(FP), INP
	MOVD	p_len+16(FP), LEN

	SRD	$6, LEN
	SLD	$6, LEN
	ADD	INP, LEN, END

	CMP	INP, END
	BEQ	end

	MOVD	$·kcon(SB), TBL_STRT
	MOVD	$0x10, R_x010

#ifdef GOARCH_ppc64le
	MOVWZ	$8, TEMP
	LVSL	(TEMP)(R0), LEMASK
	VSPLTISB	$0x0F, KI
	VXOR	KI, LEMASK, LEMASK
#endif

	LXVW4X	(CTX)(R_x000), V0
	LXVW4X	(CTX)(R_x010), V4

	// unpack the input values into vector registers
	VSLDOI	$4, V0, V0, V1
	VSLDOI	$8, V0, V0, V2
	VSLDOI	$12, V0, V0, V3
	VSLDOI	$4, V4, V4, V5
	VSLDOI	$8, V4, V4, V6
	VSLDOI	$12, V4, V4, V7

	MOVD	$0x020, R_x020
	MOVD	$0x030, R_x030
	MOVD	$0x040, R_x040
	MOVD	$0x050, R_x050
	MOVD	$0x060, R_x060
	MOVD	$0x070, R_x070
	MOVD	$0x080, R_x080
	MOVD	$0x090, R_x090
	MOVD	$0x0a0, R_x0a0
	MOVD	$0x0b0, R_x0b0
	MOVD	$0x0c0, R_x0c0
	MOVD	$0x0d0, R_x0d0
	MOVD	$0x0e0, R_x0e0
	MOVD	$0x0f0, R_x0f0
	MOVD	$0x100, R_x100
	MOVD	$0x110, R_x110

loop:
	MOVD	TBL_STRT, TBL
	LVX	(TBL)(R_x000), KI

	LXVD2X	(INP)(R_x000), V8 // load v8 in advance

	// Offload to VSR24-31 (aka FPR24-31)
	XXLOR	V0, V0, VS24
	XXLOR	V1, V1, VS25
	XXLOR	V2, V2, VS26
	XXLOR	V3, V3, VS27
	XXLOR	V4, V4, VS28
	XXLOR	V5, V5, VS29
	XXLOR	V6, V6, VS30
	XXLOR	V7, V7, VS31

	VADDUWM	KI, V7, V7        // h+K[i]
	LVX	(TBL)(R_x010), KI

	VPERMLE(V8, V8, LEMASK, V8)
	SHA256ROUND0(V0, V1, V2, V3, V4, V5, V6, V7, V8, R_x020)
	VSLDOI	$4, V8, V8, V9
	SHA256ROUND0(V7, V0, V1, V2, V3, V4, V5, V6, V9, R_x030)
	VSLDOI	$4, V9, V9, V10
	SHA256ROUND0(V6, V7, V0, V1, V2, V3, V4, V5, V10, R_x040)
	LXVD2X	(INP)(R_x010), V12 // load v12 in advance
	VSLDOI	$4, V10, V10, V11
	SHA256ROUND0(V5, V6, V7, V0, V1, V2, V3, V4, V11, R_x050)
	VPERMLE(V12, V12, LEMASK, V12)
	SHA256ROUND0(V4, V5, V6, V7, V0, V1, V2, V3, V12, R_x060)
	VSLDOI	$4, V12, V12, V13
	SHA256ROUND0(V3, V4, V5, V6, V7, V0, V1, V2, V13, R_x070)
	VSLDOI	$4, V13, V13, V14
	SHA256ROUND0(V2, V3, V4, V5, V6, V7, V0, V1, V14, R_x080)
	LXVD2X	(INP)(R_x020), V16 // load v16 in advance
	VSLDOI	$4, V14, V14, V15
	SHA256ROUND0(V1, V2, V3, V4, V5, V6, V7, V0, V15, R_x090)
	VPERMLE(V16, V16, LEMASK, V16)
	SHA256ROUND0(V0, V1, V2, V3, V4, V5, V6, V7, V16, R_x0a0)
	VSLDOI	$4, V16, V16, V17
	SHA256ROUND0(V7, V0, V1, V2, V3, V4, V5, V6, V17, R_x0b0)
	VSLDOI	$4, V17, V17, V18
	SHA256ROUND0(V6, V7, V0, V1, V2, V3, V4, V5, V18, R_x0c0)
	VSLDOI	$4, V18, V18, V19
	LXVD2X	(INP)(R_x030), V20 // load v20 in advance
	SHA256ROUND0(V5, V6, V7, V0, V1, V2, V3, V4, V19, R_x0d0)
	VPERMLE(V20, V20, LEMASK, V20)
	SHA256ROUND0(V4, V5, V6, V7, V0, V1, V2, V3, V20, R_x0e0)
	VSLDOI	$4, V20, V20, V21
	SHA256ROUND0(V3, V4, V5, V6, V7, V0, V1, V2, V21, R_x0f0)
	VSLDOI	$4, V21, V21, V22
	SHA256ROUND0(V2, V3, V4, V5, V6, V7, V0, V1, V22, R_x100)
	VSLDOI	$4, V22, V22, V23
	SHA256ROUND1(V1, V2, V3, V4, V5, V6, V7, V0, V23, V8, V9, V17, V22, R_x110)

	MOVD	$3, TEMP
	MOVD	TEMP, CTR
	ADD	$0x120, TBL
	ADD	$0x40, INP

L16_xx:
	SHA256ROUND1(V0, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V18, V23, R_x000)
	SHA256ROUND1(V7, V0, V1, V2, V3, V4, V5, V6, V9, V10, V11, V19, V8, R_x010)
	SHA256ROUND1(V6, V7, V0, V1, V2, V3, V4, V5, V10, V11, V12, V20, V9, R_x020)
	SHA256ROUND1(V5, V6, V7, V0, V1, V2, V3, V4, V11, V12, V13, V21, V10, R_x030)
	SHA256ROUND1(V4, V5, V6, V7, V0, V1, V2, V3, V12, V13, V14, V22, V11, R_x040)
	SHA256ROUND1(V3, V4, V5, V6, V7, V0, V1, V2, V13, V14, V15, V23, V12, R_x050)
	SHA256ROUND1(V2, V3, V4, V5, V6, V7, V0, V1, V14, V15, V16, V8, V13, R_x060)
	SHA256ROUND1(V1, V2, V3, V4, V5, V6, V7, V0, V15, V16, V17, V9, V14, R_x070)
	SHA256ROUND1(V0, V1, V2, V3, V4, V5, V6, V7, V16, V17, V18, V10, V15, R_x080)
	SHA256ROUND1(V7, V0, V1, V2, V3, V4, V5, V6, V17, V18, V19, V11, V16, R_x090)
	SHA256ROUND1(V6, V7, V0, V1, V2, V3, V4, V5, V18, V19, V20, V12, V17, R_x0a0)
	SHA256ROUND1(V5, V6, V7, V0, V1, V2, V3, V4, V19, V20, V21, V13, V18, R_x0b0)
	SHA256ROUND1(V4, V5, V6, V7, V0, V1, V2, V3, V20, V21, V22, V14, V19, R_x0c0)
	SHA256ROUND1(V3, V4, V5, V6, V7, V0, V1, V2, V21, V22, V23, V15, V20, R_x0d0)
	SHA256ROUND1(V2, V3, V4, V5, V6, V7, V0, V1, V22, V23, V8, V16, V21, R_x0e0)
	SHA256ROUND1(V1, V2, V3, V4, V5, V6, V7, V0, V23, V8, V9, V17, V22, R_x0f0)
	ADD	$0x100, TBL

	BDNZ	L16_xx

	XXLOR	VS24, VS24, V10

	XXLOR	VS25, VS25, V11
	VADDUWM	V10, V0, V0
	XXLOR	VS26, VS26, V12
	VADDUWM	V11, V1, V1
	XXLOR	VS27, VS27, V13
	VADDUWM	V12, V2, V2
	XXLOR	VS28, VS28, V14
	VADDUWM	V13, V3, V3
	XXLOR	VS29, VS29, V15
	VADDUWM	V14, V4, V4
	XXLOR	VS30, VS30, V16
	VADDUWM	V15, V5, V5
	XXLOR	VS31, VS31, V17
	VADDUWM	V16, V6, V6
	VADDUWM	V17, V7, V7

	CMPU	INP, END
	BLT	loop

	LVX	(TBL)(R_x000), V8
	VPERM	V0, V1, KI, V0
	LVX	(TBL)(R_x010), V9
	VPERM	V4, V5, KI, V4
	VPERM	V0, V2, V8, V0
	VPERM	V4, V6, V8, V4
	VPERM	V0, V3, V9, V0
	VPERM	V4, V7, V9, V4
	STXVD2X	V0, (CTX+R_x000)
	STXVD2X	V4, (CTX+R_x010)

end:
	RET

