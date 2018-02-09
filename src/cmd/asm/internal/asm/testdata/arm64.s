// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This input was created by taking the instruction productions in
// the old assembler's (7a's) grammar and hand-writing complete
// instructions for each rule, to guarantee we cover the same space.

#include "../../../../../runtime/textflag.h"

TEXT	foo(SB), DUPOK|NOSPLIT, $-8

//
// ADD
//
//	LTYPE1 imsr ',' spreg ',' reg
//	{
//		outcode($1, &$2, $4, &$6);
//	}
// imsr comes from the old 7a, we only support immediates and registers
	ADDW	$1, R2, R3
	ADDW	R1, R2, R3
	ADDW	R1, ZR, R3
	ADD	$1, R2, R3
	ADD	R1, R2, R3
	ADD	R1, ZR, R3
	ADD	$1, R2, R3
	ADD	R1>>11, R2, R3
	ADD	R1<<22, R2, R3
	ADD	R1->33, R2, R3
	AND	R1@>33, R2, R3
	ADD	R1.UXTB, R2, R3                 // 4300218b
	ADD	R1.UXTB<<4, R2, R3              // 4310218b
	ADDW	R2.SXTW, R10, R12               // 4cc1220b
	ADD	R18.UXTX, R14, R17              // d161328b
	ADDSW	R18.UXTW, R14, R17              // d141322b
	ADDS	R12.SXTX, R3, R1                // 61e02cab
	SUB	R19.UXTH<<4, R2, R21            // 553033cb
	SUBW	R1.UXTX<<1, R3, R2              // 6264214b
	SUBS	R3.UXTX, R8, R9                 // 096123eb
	SUBSW	R17.UXTH, R15, R21              // f521316b
	CMP	R2.SXTH, R13                    // bfa122eb
	CMN	R1.SXTX<<2, R10                 // 5fe921ab
	CMPW	R2.UXTH<<3, R11                 // 7f2d226b
	CMNW	R1.SXTB, R9                     // 3f81212b
	VADDP	V1.B16, V2.B16, V3.B16          // 43bc214e
	VADDP	V1.S4, V2.S4, V3.S4             // 43bca14e
	VADDP	V1.D2, V2.D2, V3.D2             // 43bce14e
	VAND	V21.B8, V12.B8, V3.B8           // 831d350e
	VCMEQ	V1.H4, V2.H4, V3.H4             // 438c612e
	VORR	V5.B16, V4.B16, V3.B16          // 831ca54e
	VADD	V16.S4, V5.S4, V9.S4            // a984b04e
	VEOR	V0.B16, V1.B16, V0.B16          // 201c206e
	SHA256H	V9.S4, V3, V2                   // 6240095e
	SHA256H2	V9.S4, V4, V3           // 8350095e
	SHA256SU0	V8.S4, V7.S4            // 0729285e
	SHA256SU1	V6.S4, V5.S4, V7.S4     // a760065e
	SHA1SU0	V11.S4, V8.S4, V6.S4            // 06310b5e
	SHA1SU1	V5.S4, V1.S4                    // a118285e
	SHA1C	V1.S4, V2, V3                   // 4300015e
	SHA1H	V5, V4                          // a408285e
	SHA1M	V8.S4, V7, V6                   // e620085e
	SHA1P	V11.S4, V10, V9                 // 49110b5e
	VADDV	V0.S4, V0                       // 00b8b14e
	VMOVI	$82, V0.B16                     // 40e6024f
	VUADDLV	V6.B16, V6                      // c638306e
	VADD	V1, V2, V3                      // 4384e15e
	VADD	V1, V3, V3                      // 6384e15e
	VSUB	V12, V30, V30                   // de87ec7e
	VSUB	V12, V20, V30                   // 9e86ec7e

//	LTYPE1 imsr ',' spreg ','
//	{
//		outcode($1, &$2, $4, &nullgen);
//	}
//	LTYPE1 imsr ',' reg
//	{
//		outcode($1, &$2, NREG, &$4);
//	}
	ADDW	$1, R2
	ADDW	R1, R2
	ADD	$1, R2
	ADD	R1, R2
	ADD	R1>>11, R2
	ADD	R1<<22, R2
	ADD	R1->33, R2
	AND	R1@>33, R2

// logical ops
// make sure constants get encoded into an instruction when it could
	AND	$(1<<63), R1   // AND	$-9223372036854775808, R1 // 21004192
	AND	$(1<<63-1), R1 // AND	$9223372036854775807, R1  // 21f84092
	ORR	$(1<<63), R1   // ORR	$-9223372036854775808, R1 // 210041b2
	ORR	$(1<<63-1), R1 // ORR	$9223372036854775807, R1  // 21f840b2
	EOR	$(1<<63), R1   // EOR	$-9223372036854775808, R1 // 210041d2
	EOR	$(1<<63-1), R1 // EOR	$9223372036854775807, R1  // 21f840d2

//
// CLS
//
//	LTYPE2 imsr ',' reg
//	{
//		outcode($1, &$2, NREG, &$4);
//	}
	CLSW	R1, R2
	CLS	R1, R2

//
// MOV
//
//	LTYPE3 addr ',' addr
//	{
//		outcode($1, &$2, NREG, &$4);
//	}
	MOVW	R1, R2
	MOVW	ZR, R1
	MOVW	R1, ZR
	MOVW	$1, ZR
	MOVW	$1, R1
	MOVW	ZR, (R1)
	MOVD	R1, R2
	MOVD	ZR, R1
	MOVD	$1, ZR
	MOVD	$1, R1
	MOVD	ZR, (R1)
	VLD1	(R8), [V1.B16, V2.B16]                          // 01a1404c
	VLD1.P	(R3), [V31.H8, V0.H8]                           // 7fa4df4c
	VLD1.P	(R8)(R20), [V21.B16, V22.B16]                   // VLD1.P	(R8)(R20*1), [V21.B16,V22.B16] // 15a1d44c
	VLD1.P	64(R1), [V5.B16, V6.B16, V7.B16, V8.B16]        // 2520df4c
	VST1.P	[V4.S4, V5.S4], 32(R1)                          // 24a89f4c
	VST1	[V0.S4, V1.S4], (R0)                            // 00a8004c
	VMOVS	V20, (R0)                                       // 140000bd
	VMOVS.P	V20, 4(R0)                                      // 144400bc
	VMOVS.W	V20, 4(R0)                                      // 144c00bc
	VMOVS	(R0), V20                                       // 140040bd
	VMOVS.P	8(R0), V20                                      // 148440bc
	VMOVS.W	8(R0), V20                                      // 148c40bc

	// small offset fits into instructions
	MOVB	1(R1), R2 // 22048039
	MOVH	1(R1), R2 // 22108078
	MOVH	2(R1), R2 // 22048079
	MOVW	1(R1), R2 // 221080b8
	MOVW	4(R1), R2 // 220480b9
	MOVD	1(R1), R2 // 221040f8
	MOVD	8(R1), R2 // 220440f9
	FMOVS	1(R1), F2 // 221040bc
	FMOVS	4(R1), F2 // 220440bd
	FMOVD	1(R1), F2 // 221040fc
	FMOVD	8(R1), F2 // 220440fd
	MOVB	R1, 1(R2) // 41040039
	MOVH	R1, 1(R2) // 41100078
	MOVH	R1, 2(R2) // 41040079
	MOVW	R1, 1(R2) // 411000b8
	MOVW	R1, 4(R2) // 410400b9
	MOVD	R1, 1(R2) // 411000f8
	MOVD	R1, 8(R2) // 410400f9
	FMOVS	F1, 1(R2) // 411000bc
	FMOVS	F1, 4(R2) // 410400bd
	FMOVD	F1, 1(R2) // 411000fc
	FMOVD	F1, 8(R2) // 410400fd

	// large aligned offset, use two instructions
	MOVB	0x1001(R1), R2 // MOVB	4097(R1), R2  // 3b04409162078039
	MOVH	0x2002(R1), R2 // MOVH	8194(R1), R2  // 3b08409162078079
	MOVW	0x4004(R1), R2 // MOVW	16388(R1), R2 // 3b104091620780b9
	MOVD	0x8008(R1), R2 // MOVD	32776(R1), R2 // 3b204091620740f9
	FMOVS	0x4004(R1), F2 // FMOVS	16388(R1), F2 // 3b104091620740bd
	FMOVD	0x8008(R1), F2 // FMOVD	32776(R1), F2 // 3b204091620740fd
	MOVB	R1, 0x1001(R2) // MOVB	R1, 4097(R2)  // 5b04409161070039
	MOVH	R1, 0x2002(R2) // MOVH	R1, 8194(R2)  // 5b08409161070079
	MOVW	R1, 0x4004(R2) // MOVW	R1, 16388(R2) // 5b104091610700b9
	MOVD	R1, 0x8008(R2) // MOVD	R1, 32776(R2) // 5b204091610700f9
	FMOVS	F1, 0x4004(R2) // FMOVS	F1, 16388(R2) // 5b104091610700bd
	FMOVD	F1, 0x8008(R2) // FMOVD	F1, 32776(R2) // 5b204091610700fd

	// very large or unaligned offset uses constant pool
	// the encoding cannot be checked as the address of the constant pool is unknown.
	// here we only test that they can be assembled.
	MOVB	0x44332211(R1), R2 // MOVB	1144201745(R1), R2
	MOVH	0x44332211(R1), R2 // MOVH	1144201745(R1), R2
	MOVW	0x44332211(R1), R2 // MOVW	1144201745(R1), R2
	MOVD	0x44332211(R1), R2 // MOVD	1144201745(R1), R2
	FMOVS	0x44332211(R1), F2 // FMOVS	1144201745(R1), F2
	FMOVD	0x44332211(R1), F2 // FMOVD	1144201745(R1), F2
	MOVB	R1, 0x44332211(R2) // MOVB	R1, 1144201745(R2)
	MOVH	R1, 0x44332211(R2) // MOVH	R1, 1144201745(R2)
	MOVW	R1, 0x44332211(R2) // MOVW	R1, 1144201745(R2)
	MOVD	R1, 0x44332211(R2) // MOVD	R1, 1144201745(R2)
	FMOVS	F1, 0x44332211(R2) // FMOVS	F1, 1144201745(R2)
	FMOVD	F1, 0x44332211(R2) // FMOVD	F1, 1144201745(R2)

//
// MOVK
//
//		LMOVK imm ',' reg
//	{
//		outcode($1, &$2, NREG, &$4);
//	}
	MOVK	$1, R1
	VMOV	V8.S[1], R1       // 013d0c0e
	VMOV	V0.D[0], R11      // 0b3c084e
	VMOV	V0.D[1], R11      // 0b3c184e
	VMOV	R20, V1.S[0]      // 811e044e
	VMOV	R1, V9.H4         // 290c020e
	VMOV	R22, V11.D2       // cb0e084e
	VMOV 	V2.B16, V4.B16    // 441ca24e
	VMOV	V20.S[0], V20     // 9406045e
	VREV32	V5.B16, V5.B16    // a508206e
	VDUP	V19.S[0], V17.S4  // 7106044e
//
// B/BL
//
//		LTYPE4 comma rel
//	{
//		outcode($1, &nullgen, NREG, &$3);
//	}
	BL	1(PC) // CALL 1(PC)

//		LTYPE4 comma nireg
//	{
//		outcode($1, &nullgen, NREG, &$3);
//	}
	BL	(R2) // CALL (R2)
	BL	foo(SB) // CALL foo(SB)
	BL	bar<>(SB) // CALL bar<>(SB)
//
// BEQ
//
//		LTYPE5 comma rel
//	{
//		outcode($1, &nullgen, NREG, &$3);
//	}
	BEQ	1(PC)
//
// SVC
//
//		LTYPE6
//	{
//		outcode($1, &nullgen, NREG, &nullgen);
//	}
	SVC

//
// CMP
//
//		LTYPE7 imsr ',' spreg comma
//	{
//		outcode($1, &$2, $4, &nullgen);
//	}
	CMP	$3, R2
	CMP	R1, R2
	CMP	R1->11, R2
	CMP	R1>>22, R2
	CMP	R1<<33, R2
	CMP	R22.SXTX, RSP // ffe336eb
//
// CBZ
//
//		LTYPE8 reg ',' rel
//	{
//		outcode($1, &$2, NREG, &$4);
//	}
again:
	CBZ	R1, again // CBZ R1

//
// CSET
//
//		LTYPER cond ',' reg
//	{
//		outcode($1, &$2, NREG, &$4);
//	}
	CSET	GT, R1	// e1d79f9a
//
// CSEL/CSINC/CSNEG/CSINV
//
//		LTYPES cond ',' reg ',' reg ',' reg
//	{
//		outgcode($1, &$2, $6.reg, &$4, &$8);
//	}
	CSEL	LT, R1, R2, ZR	// 3fb0829a
	CSINC	GT, R1, ZR, R3	// 23c49f9a
	CSNEG	MI, R1, R2, R3	// 234482da
	CSINV	CS, R1, R2, R3	// CSINV HS, R1, R2, R3 // 232082da

//		LTYPES cond ',' reg ',' reg
//	{
//		outcode($1, &$2, $4.reg, &$6);
//	}
	CINC	EQ, R4, R9	// 8914849a
	CINV	PL, R11, R22	// 76418bda
	CNEG	LS, R13, R7	// a7858dda
//
// CCMN
//
//		LTYPEU cond ',' imsr ',' reg ',' imm comma
//	{
//		outgcode($1, &$2, $6.reg, &$4, &$8);
//	}
	CCMN	MI, ZR, R1, $4	// e44341ba

//
// FADDD
//
//		LTYPEK frcon ',' freg
//	{
//		outcode($1, &$2, NREG, &$4);
//	}
	FADDD	$0.5, F1 // FADDD $(0.5), F1
	FADDD	F1, F2

//		LTYPEK frcon ',' freg ',' freg
//	{
//		outcode($1, &$2, $4.reg, &$6);
//	}
	FADDD	$0.7, F1, F2 // FADDD	$(0.69999999999999996), F1, F2
	FADDD	F1, F2, F3

//
// FCMP
//
//		LTYPEL frcon ',' freg comma
//	{
//		outcode($1, &$2, $4.reg, &nullgen);
//	}
//	FCMP	$0.2, F1
//	FCMP	F1, F2

//
// FCCMP
//
//		LTYPEF cond ',' freg ',' freg ',' imm comma
//	{
//		outgcode($1, &$2, $6.reg, &$4, &$8);
//	}
	FCCMPS	LT, F1, F2, $1	// 41b4211e

//
// FMULA
//
//		LTYPE9 freg ',' freg ',' freg ',' freg comma
//	{
//		outgcode($1, &$2, $4.reg, &$6, &$8);
//	}
//	FMULA	F1, F2, F3, F4

//
// FCSEL
//
//		LFCSEL cond ',' freg ',' freg ',' freg
//	{
//		outgcode($1, &$2, $6.reg, &$4, &$8);
//	}
//
// MADD Rn,Rm,Ra,Rd
//
//		LTYPEM reg ',' reg ',' sreg ',' reg
//	{
//		outgcode($1, &$2, $6, &$4, &$8);
//	}
//	MADD	R1, R2, R3, R4

// DMB, HINT
//
//		LDMB imm
//	{
//		outcode($1, &$2, NREG, &nullgen);
//	}
	DMB	$1

//
// STXR
//
//		LSTXR reg ',' addr ',' reg
//	{
//		outcode($1, &$2, &$4, &$6);
//	}
	LDAXRW	(R0), R2
	STLXRW	R1, (R0), R3

// RET
//
//		LTYPEA comma
//	{
//		outcode($1, &nullgen, NREG, &nullgen);
//	}
	BEQ	2(PC)
	RET

// More B/BL cases, and canonical names JMP, CALL.

	BEQ	2(PC)
	B	foo(SB) // JMP foo(SB)
	BL	foo(SB) // CALL foo(SB)
	BEQ	2(PC)
	TBZ	$1, R1, 2(PC)
	TBNZ	$2, R2, 2(PC)
	JMP	foo(SB)
	CALL	foo(SB)

// LDP/STP
	LDP	(R0), (R1, R2)
	LDP	8(R0), (R1, R2)
	LDP.W	8(R0), (R1, R2)
	LDP.P	8(R0), (R1, R2)
	LDP	x(SB), (R1, R2)
	LDP	x+8(SB), (R1, R2)
	STP	(R3, R4), (R5)
	STP	(R3, R4), 8(R5)
	STP.W	(R3, R4), 8(R5)
	STP.P	(R3, R4), 8(R5)
	STP	(R3, R4), x(SB)
	STP	(R3, R4), x+8(SB)

// END
//
//	LTYPEE comma
//	{
//		outcode($1, &nullgen, NREG, &nullgen);
//	}
	END
