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
	ADD	R2, RSP, RSP                    // ff63228b
	ADD	R2.SXTX<<1, RSP, RSP            // ffe7228b
	ADD	ZR.SXTX<<1, R2, R3              // 43e43f8b
	ADDW	R2.SXTW, R10, R12               // 4cc1220b
	ADD	R18.UXTX, R14, R17              // d161328b
	ADDSW	R18.UXTW, R14, R17              // d141322b
	ADDS	R12.SXTX, R3, R1                // 61e02cab
	SUB	R19.UXTH<<4, R2, R21            // 553033cb
	SUBW	R1.UXTX<<1, R3, R2              // 6264214b
	SUBS	R3.UXTX, R8, R9                 // 096123eb
	SUBSW	R17.UXTH, R15, R21              // f521316b
	SUBW	ZR<<14, R19, R13                // 6d3a1f4b
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
	VFMLA	V1.D2, V12.D2, V1.D2            // 81cd614e
	VFMLA	V1.S2, V12.S2, V1.S2            // 81cd210e
	VFMLA	V1.S4, V12.S4, V1.S4            // 81cd214e
	VFMLS	V1.D2, V12.D2, V1.D2            // 81cde14e
	VFMLS	V1.S2, V12.S2, V1.S2            // 81cda10e
	VFMLS	V1.S4, V12.S4, V1.S4            // 81cda14e
	VPMULL	V2.D1, V1.D1, V3.Q1             // 23e0e20e
	VPMULL2	V2.D2, V1.D2, V4.Q1             // 24e0e24e
	VPMULL	V2.B8, V1.B8, V3.H8             // 23e0220e
	VPMULL2	V2.B16, V1.B16, V4.H8           // 24e0224e
	VEXT	$4, V2.B8, V1.B8, V3.B8         // 2320022e
	VEXT	$8, V2.B16, V1.B16, V3.B16      // 2340026e
	VRBIT	V24.B16, V24.B16                // 185b606e
	VRBIT	V24.B8, V24.B8                  // 185b602e
	VUSHR	$56, V1.D2, V2.D2               // 2204486f
	VUSHR	$24, V1.S4, V2.S4               // 2204286f
	VUSHR	$24, V1.S2, V2.S2               // 2204282f
	VUSHR	$8, V1.H4, V2.H4                // 2204182f
	VUSHR	$8, V1.H8, V2.H8                // 2204186f
	VUSHR	$2, V1.B8, V2.B8                // 22040e2f
	VUSHR	$2, V1.B16, V2.B16              // 22040e6f
	VSHL	$56, V1.D2, V2.D2               // 2254784f
	VSHL	$24, V1.S4, V2.S4               // 2254384f
	VSHL	$24, V1.S2, V2.S2               // 2254380f
	VSHL	$8, V1.H4, V2.H4                // 2254180f
	VSHL	$8, V1.H8, V2.H8                // 2254184f
	VSHL	$2, V1.B8, V2.B8                // 22540a0f
	VSHL	$2, V1.B16, V2.B16              // 22540a4f
	VSRI	$56, V1.D2, V2.D2               // 2244486f
	VSRI	$24, V1.S4, V2.S4               // 2244286f
	VSRI	$24, V1.S2, V2.S2               // 2244282f
	VSRI	$8, V1.H4, V2.H4                // 2244182f
	VSRI	$8, V1.H8, V2.H8                // 2244186f
	VSRI	$2, V1.B8, V2.B8                // 22440e2f
	VSRI	$2, V1.B16, V2.B16              // 22440e6f
	VTBL	V22.B16, [V28.B16, V29.B16], V11.B16                                    // 8b23164e
	VTBL	V18.B8, [V17.B16, V18.B16, V19.B16], V22.B8                             // 3642120e
	VTBL	V31.B8, [V14.B16, V15.B16, V16.B16, V17.B16], V15.B8                    // cf611f0e
	VTBL	V14.B16, [V16.B16], V11.B16                                             // 0b020e4e
	VTBL	V28.B16, [V25.B16, V26.B16], V5.B16                                     // 25231c4e
	VTBL	V16.B8, [V4.B16, V5.B16, V6.B16], V12.B8                                // 8c40100e
	VTBL	V4.B8, [V16.B16, V17.B16, V18.B16, V19.B16], V4.B8                      // 0462040e
	VTBL	V15.B8, [V1.B16], V20.B8                                                // 34000f0e
	VTBL	V26.B16, [V2.B16, V3.B16], V26.B16                                      // 5a201a4e
	VTBL	V15.B8, [V6.B16, V7.B16, V8.B16], V2.B8                                 // c2400f0e
	VTBL	V2.B16, [V27.B16, V28.B16, V29.B16, V30.B16], V18.B16                   // 7263024e
	VTBL	V11.B16, [V13.B16], V27.B16                                             // bb010b4e
	VTBL	V3.B8, [V7.B16, V8.B16], V25.B8                                         // f920030e
	VTBL	V14.B16, [V3.B16, V4.B16, V5.B16], V17.B16                              // 71400e4e
	VTBL	V13.B16, [V29.B16, V30.B16, V31.B16, V0.B16], V28.B16                   // bc630d4e
	VTBL	V3.B8, [V27.B16], V8.B8                                                 // 6803030e
	VZIP1	V16.H8, V3.H8, V19.H8           // 7338504e
	VZIP2	V22.D2, V25.D2, V21.D2          // 357bd64e
	VZIP1	V6.D2, V9.D2, V11.D2            // 2b39c64e
	VZIP2	V10.D2, V13.D2, V3.D2           // a379ca4e
	VZIP1	V17.S2, V4.S2, V26.S2           // 9a38910e
	VZIP2	V25.S2, V14.S2, V25.S2          // d979990e
	MOVD	(R2)(R6.SXTW), R4               // 44c866f8
	MOVD	(R3)(R6), R5                    // MOVD	(R3)(R6*1), R5                  // 656866f8
	MOVD	(R2)(R6), R4                    // MOVD	(R2)(R6*1), R4                  // 446866f8
	MOVWU	(R19)(R18<<2), R18              // 727a72b8
	MOVD	(R2)(R6<<3), R4                 // 447866f8
	MOVD	(R3)(R7.SXTX<<3), R8            // 68f867f8
	MOVWU	(R5)(R4.UXTW), R10              // aa4864b8
	MOVBU	(R3)(R9.UXTW), R8               // 68486938
	MOVBU	(R5)(R8), R10                   // MOVBU	(R5)(R8*1), R10         // aa686838
	MOVHU	(R2)(R7.SXTW<<1), R11           // 4bd86778
	MOVHU	(R1)(R2<<1), R5                 // 25786278
	MOVB	(R9)(R3.UXTW), R6               // 2649a338
	MOVB	(R10)(R6), R15                  // MOVB	(R10)(R6*1), R15                // 4f69a638
	MOVH	(R5)(R7.SXTX<<1), R18           // b2f8a778
	MOVH	(R8)(R4<<1), R10                // 0a79a478
	MOVW	(R9)(R8.SXTW<<2), R19           // 33d9a8b8
	MOVW	(R1)(R4.SXTX), R11              // 2be8a4b8
	MOVW	(R1)(R4.SXTX), ZR               // 3fe8a4b8
	MOVW	(R2)(R5), R12                   // MOVW	(R2)(R5*1), R12                  // 4c68a5b8
	MOVD	R5, (R2)(R6<<3)                 // 457826f8
	MOVD	R9, (R6)(R7.SXTX<<3)            // c9f827f8
	MOVD	ZR, (R6)(R7.SXTX<<3)            // dff827f8
	MOVW	R8, (R2)(R3.UXTW<<2)            // 485823b8
	MOVW	R7, (R3)(R4.SXTW)               // 67c824b8
	MOVB	R4, (R2)(R6.SXTX)               // 44e82638
	MOVB	R8, (R3)(R9.UXTW)               // 68482938
	MOVB	R10, (R5)(R8)                   // MOVB	R10, (R5)(R8*1)                  // aa682838
	MOVH	R11, (R2)(R7.SXTW<<1)           // 4bd82778
	MOVH	R5, (R1)(R2<<1)                 // 25782278
	MOVH	R7, (R2)(R5.SXTX<<1)            // 47f82578
	MOVH	R8, (R3)(R6.UXTW)               // 68482678
	MOVB	(R29)(R30<<0), R14              // ae7bbe38
	MOVB	(R29)(R30), R14                 // MOVB	(R29)(R30*1), R14                // ae6bbe38
	MOVB	R4, (R2)(R6.SXTX)               // 44e82638

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

	AND	$0x22220000, R3, R4   // AND $572653568, R3, R4   // 5b44a4d264001b8a
	ORR	$0x22220000, R3, R4   // ORR $572653568, R3, R4   // 5b44a4d264001baa
	EOR	$0x22220000, R3, R4   // EOR $572653568, R3, R4   // 5b44a4d264001bca
	BIC	$0x22220000, R3, R4   // BIC $572653568, R3, R4   // 5b44a4d264003b8a
	ORN	$0x22220000, R3, R4   // ORN $572653568, R3, R4   // 5b44a4d264003baa
	EON	$0x22220000, R3, R4   // EON $572653568, R3, R4   // 5b44a4d264003bca
	ANDS	$0x22220000, R3, R4   // ANDS $572653568, R3, R4  // 5b44a4d264001bea
	BICS	$0x22220000, R3, R4   // BICS $572653568, R3, R4  // 5b44a4d264003bea

	AND	$8, R0, RSP // 1f007d92
	ORR	$8, R0, RSP // 1f007db2
	EOR	$8, R0, RSP // 1f007dd2
	BIC	$8, R0, RSP // 1ff87c92
	ORN	$8, R0, RSP // 1ff87cb2
	EON	$8, R0, RSP // 1ff87cd2

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
	VLD1.P	1(R0), V4.B[15]                                 // 041cdf4d
	VLD1.P	2(R0), V4.H[7]                                  // 0458df4d
	VLD1.P	4(R0), V4.S[3]                                  // 0490df4d
	VLD1.P	8(R0), V4.D[1]                                  // 0484df4d
	VLD1.P	(R0)(R1), V4.D[1]                               // VLD1.P	(R0)(R1*1), V4.D[1] // 0484c14d
	VLD1	(R0), V4.D[1]                                   // 0484404d
	VST1.P	[V4.S4, V5.S4], 32(R1)                          // 24a89f4c
	VST1	[V0.S4, V1.S4], (R0)                            // 00a8004c
	VLD1	(R30), [V15.S2, V16.S2]                         // cfab400c
	VLD1.P	24(R30), [V3.S2,V4.S2,V5.S2]                    // c36bdf0c
	VST1.P	[V24.S2], 8(R2)                                 // 58789f0c
	VST1	[V29.S2, V30.S2], (R29)                         // bdab000c
	VST1	[V14.H4, V15.H4, V16.H4], (R27)                 // 6e67000c
	VST1.P	V4.B[15], 1(R0)                                 // 041c9f4d
	VST1.P	V4.H[7], 2(R0)                                  // 04589f4d
	VST1.P	V4.S[3], 4(R0)                                  // 04909f4d
	VST1.P	V4.D[1], 8(R0)                                  // 04849f4d
	VST1.P	V4.D[1], (R0)(R1)                               // VST1.P	V4.D[1], (R0)(R1*1) // 0484814d
	VST1	V4.D[1], (R0)                                   // 0484004d
	FMOVS	F20, (R0)                                       // 140000bd
	FMOVS.P	F20, 4(R0)                                      // 144400bc
	FMOVS.W	F20, 4(R0)                                      // 144c00bc
	FMOVS	(R0), F20                                       // 140040bd
	FMOVS.P	8(R0), F20                                      // 148440bc
	FMOVS.W	8(R0), F20                                      // 148c40bc
	FMOVD	F20, (R2)                                       // 540000fd
	FMOVD.P	F20, 8(R1)                                      // 348400fc
	FMOVD.W	8(R1), F20                                      // 348c40fc
	PRFM	(R2), PLDL1KEEP                                 // 400080f9
	PRFM	16(R2), PLDL1KEEP                               // 400880f9
	PRFM	48(R6), PSTL2STRM                               // d31880f9
	PRFM	8(R12), PLIL3STRM                               // 8d0580f9
	PRFM	(R8), $25                                       // 190180f9
	PRFM	8(R9), $30                                      // 3e0580f9

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
	VMOV	V8.S[1], R1           // 013d0c0e
	VMOV	V0.D[0], R11          // 0b3c084e
	VMOV	V0.D[1], R11          // 0b3c184e
	VMOV	R20, V1.S[0]          // 811e044e
	VMOV	R20, V1.S[1]          // 811e0c4e
	VMOV	R1, V9.H4             // 290c020e
	VMOV	R22, V11.D2           // cb0e084e
	VMOV	V2.B16, V4.B16        // 441ca24e
	VMOV	V20.S[0], V20         // 9406045e
	VMOV	V12.D[0], V12.D[1]    // 8c05186e
	VMOV	V10.S[0], V12.S[1]    // 4c050c6e
	VMOV	V9.H[0], V12.H[1]     // 2c05066e
	VMOV	V8.B[0], V12.B[1]     // 0c05036e
	VMOV	V8.B[7], V4.B[8]      // 043d116e
	VREV32	V5.B16, V5.B16        // a508206e
	VREV64	V2.S2, V3.S2          // 4308a00e
	VREV64	V2.S4, V3.S4          // 4308a04e
	VDUP	V19.S[0], V17.S4      // 7106044e
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

	CMP	$0x22220000, RSP  // CMP $572653568, RSP   // 5b44a4d2ff633beb
	CMPW	$0x22220000, RSP  // CMPW $572653568, RSP  // 5b44a4d2ff633b6b

// TST
	TST	$15, R2                               // 5f0c40f2
	TST	R1, R2                                // 5f0001ea
	TST	R1->11, R2                            // 5f2c81ea
	TST	R1>>22, R2                            // 5f5841ea
	TST	R1<<33, R2                            // 5f8401ea
	TST	$0x22220000, R3 // TST $572653568, R3 // 5b44a4d27f001bea

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
	CSETW	HI, R2	// e2979f1a
//
// CSEL/CSINC/CSNEG/CSINV
//
//		LTYPES cond ',' reg ',' reg ',' reg
//	{
//		outgcode($1, &$2, $6.reg, &$4, &$8);
//	}
	CSEL	LT, R1, R2, ZR	// 3fb0829a
	CSELW	LT, R2, R3, R4	// 44b0831a
	CSINC	GT, R1, ZR, R3	// 23c49f9a
	CSNEG	MI, R1, R2, R3	// 234482da
	CSINV	CS, R1, R2, R3	// CSINV HS, R1, R2, R3 // 232082da
	CSINVW	MI, R2, ZR, R2	// 42409f5a

//		LTYPES cond ',' reg ',' reg
//	{
//		outcode($1, &$2, $4.reg, &$6);
//	}
	CINC	EQ, R4, R9	// 8914849a
	CINCW	PL, R2, ZR	// 5f44821a
	CINV	PL, R11, R22	// 76418bda
	CINVW	LS, R7, R13	// ed80875a
	CNEG	LS, R13, R7	// a7858dda
	CNEGW	EQ, R8, R13	// 0d15885a
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

	FMADDS	F1, F3, F2, F4          // 440c011f
	FMADDD	F4, F5, F4, F4          // 8414441f
	FMSUBS	F13, F21, F13, F19      // b3d50d1f
	FMSUBD	F11, F7, F15, F31       // ff9d4b1f
	FNMADDS	F1, F3, F2, F4          // 440c211f
	FNMADDD	F1, F3, F2, F4          // 440c611f
	FNMSUBS	F1, F3, F2, F4          // 448c211f
	FNMSUBD	F1, F3, F2, F4          // 448c611f

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
	LDARB	(R25), R2                            // 22ffdf08
	LDARH	(R5), R7                             // a7fcdf48
	LDAXPW	(R10), (R20, R16)                    // 54c17f88
	LDAXP	(R25), (R30, R11)                    // 3eaf7fc8
	LDAXRW	(R0), R2                             // 02fc5f88
	LDXPW	(R24), (R23, R11)                    // 172f7f88
	LDXP	(R0), (R16, R13)                     // 10347fc8
	STLRB	R11, (R22)                           // cbfe9f08
	STLRH	R16, (R23)                           // f0fe9f48
	STLXP	(R6, R3), (R10), R2                  // 468d22c8
	STLXPW	(R6, R11), (R22), R21                // c6ae3588
	STLXRW	R1, (R0), R3                         // 01fc0388
	STXP	(R1, R2), (R3), R10                  // 61082ac8
	STXP	(R1, R2), (RSP), R10                 // e10b2ac8
	STXPW	(R1, R2), (R3), R10                  // 61082a88
	STXPW	(R1, R2), (RSP), R10                 // e10b2a88
	SWPD	R5, (R6), R7                         // c78025f8
	SWPD	R5, (RSP), R7                        // e78325f8
	SWPW	R5, (R6), R7                         // c78025b8
	SWPW	R5, (RSP), R7                        // e78325b8
	SWPH	R5, (R6), R7                         // c7802578
	SWPH	R5, (RSP), R7                        // e7832578
	SWPB	R5, (R6), R7                         // c7802538
	SWPB	R5, (RSP), R7                        // e7832538
	LDADDD	R5, (R6), R7                         // c70025f8
	LDADDD	R5, (RSP), R7                        // e70325f8
	LDADDW	R5, (R6), R7                         // c70025b8
	LDADDW	R5, (RSP), R7                        // e70325b8
	LDADDH	R5, (R6), R7                         // c7002578
	LDADDH	R5, (RSP), R7                        // e7032578
	LDADDB	R5, (R6), R7                         // c7002538
	LDADDB	R5, (RSP), R7                        // e7032538
	LDANDD	R5, (R6), R7                         // c71025f8
	LDANDD	R5, (RSP), R7                        // e71325f8
	LDANDW	R5, (R6), R7                         // c71025b8
	LDANDW	R5, (RSP), R7                        // e71325b8
	LDANDH	R5, (R6), R7                         // c7102578
	LDANDH	R5, (RSP), R7                        // e7132578
	LDANDB	R5, (R6), R7                         // c7102538
	LDANDB	R5, (RSP), R7                        // e7132538
	LDEORD	R5, (R6), R7                         // c72025f8
	LDEORD	R5, (RSP), R7                        // e72325f8
	LDEORW	R5, (R6), R7                         // c72025b8
	LDEORW	R5, (RSP), R7                        // e72325b8
	LDEORH	R5, (R6), R7                         // c7202578
	LDEORH	R5, (RSP), R7                        // e7232578
	LDEORB	R5, (R6), R7                         // c7202538
	LDEORB	R5, (RSP), R7                        // e7232538
	LDORD	R5, (R6), R7                         // c73025f8
	LDORD	R5, (RSP), R7                        // e73325f8
	LDORW	R5, (R6), R7                         // c73025b8
	LDORW	R5, (RSP), R7                        // e73325b8
	LDORH	R5, (R6), R7                         // c7302578
	LDORH	R5, (RSP), R7                        // e7332578
	LDORB	R5, (R6), R7                         // c7302538
	LDORB	R5, (RSP), R7                        // e7332538
	LDADDALD	R2, (R1), R3                 // 2300e2f8
	LDADDALW	R5, (R4), R6                 // 8600e5b8

// RET
//
//		LTYPEA comma
//	{
//		outcode($1, &nullgen, NREG, &nullgen);
//	}
	BEQ	2(PC)
	RET
	RET	foo(SB)

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
	LDP	(R0), (R1, R2)      // 010840a9
	LDP	8(R0), (R1, R2)     // 018840a9
	LDP	-8(R0), (R1, R2)    // 01887fa9
	LDP	11(R0), (R1, R2)    // 1b2c0091610b40a9
	LDP	1024(R0), (R1, R2)  // 1b001091610b40a9
	LDP.W	8(R0), (R1, R2)     // 0188c0a9
	LDP.P	8(R0), (R1, R2)     // 0188c0a8
	LDP	(RSP), (R1, R2)     // e10b40a9
	LDP	8(RSP), (R1, R2)    // e18b40a9
	LDP	-8(RSP), (R1, R2)   // e18b7fa9
	LDP	11(RSP), (R1, R2)   // fb2f0091610b40a9
	LDP	1024(RSP), (R1, R2) // fb031091610b40a9
	LDP.W	8(RSP), (R1, R2)    // e18bc0a9
	LDP.P	8(RSP), (R1, R2)    // e18bc0a8
	LDP	-31(R0), (R1, R2)   // 1b7c00d1610b40a9
	LDP	-4(R0), (R1, R2)    // 1b1000d1610b40a9
	LDP	-8(R0), (R1, R2)    // 01887fa9
	LDP	x(SB), (R1, R2)
	LDP	x+8(SB), (R1, R2)
	LDPW	-5(R0), (R1, R2)    // 1b1400d1610b4029
	LDPW	(R0), (R1, R2)      // 01084029
	LDPW	4(R0), (R1, R2)     // 01884029
	LDPW	-4(R0), (R1, R2)    // 01887f29
	LDPW.W	4(R0), (R1, R2)     // 0188c029
	LDPW.P	4(R0), (R1, R2)     // 0188c028
	LDPW	11(R0), (R1, R2)    // 1b2c0091610b4029
	LDPW	1024(R0), (R1, R2)  // 1b001091610b4029
	LDPW	(RSP), (R1, R2)     // e10b4029
	LDPW	4(RSP), (R1, R2)    // e18b4029
	LDPW	-4(RSP), (R1, R2)   // e18b7f29
	LDPW.W	4(RSP), (R1, R2)    // e18bc029
	LDPW.P	4(RSP), (R1, R2)    // e18bc028
	LDPW	11(RSP), (R1, R2)   // fb2f0091610b4029
	LDPW	1024(RSP), (R1, R2) // fb031091610b4029
	LDPW	x(SB), (R1, R2)
	LDPW	x+8(SB), (R1, R2)
	LDPSW	(R0), (R1, R2)      // 01084069
	LDPSW	4(R0), (R1, R2)     // 01884069
	LDPSW	-4(R0), (R1, R2)    // 01887f69
	LDPSW.W	4(R0), (R1, R2)     // 0188c069
	LDPSW.P	4(R0), (R1, R2)     // 0188c068
	LDPSW	11(R0), (R1, R2)    // 1b2c0091610b4069
	LDPSW	1024(R0), (R1, R2)  // 1b001091610b4069
	LDPSW	(RSP), (R1, R2)     // e10b4069
	LDPSW	4(RSP), (R1, R2)    // e18b4069
	LDPSW	-4(RSP), (R1, R2)   // e18b7f69
	LDPSW.W	4(RSP), (R1, R2)    // e18bc069
	LDPSW.P	4(RSP), (R1, R2)    // e18bc068
	LDPSW	11(RSP), (R1, R2)   // fb2f0091610b4069
	LDPSW	1024(RSP), (R1, R2) // fb031091610b4069
	LDPSW	x(SB), (R1, R2)
	LDPSW	x+8(SB), (R1, R2)
	STP	(R3, R4), (R5)      // a31000a9
	STP	(R3, R4), 8(R5)     // a39000a9
	STP.W	(R3, R4), 8(R5)     // a39080a9
	STP.P	(R3, R4), 8(R5)     // a39080a8
	STP	(R3, R4), -8(R5)    // a3903fa9
	STP	(R3, R4), -4(R5)    // bb1000d1631300a9
	STP	(R3, R4), 11(R0)    // 1b2c0091631300a9
	STP	(R3, R4), 1024(R0)  // 1b001091631300a9
	STP	(R3, R4), (RSP)     // e31300a9
	STP	(R3, R4), 8(RSP)    // e39300a9
	STP.W	(R3, R4), 8(RSP)    // e39380a9
	STP.P	(R3, R4), 8(RSP)    // e39380a8
	STP	(R3, R4), -8(RSP)   // e3933fa9
	STP	(R3, R4), 11(RSP)   // fb2f0091631300a9
	STP	(R3, R4), 1024(RSP) // fb031091631300a9
	STP	(R3, R4), x(SB)
	STP	(R3, R4), x+8(SB)
	STPW	(R3, R4), (R5)      // a3100029
	STPW	(R3, R4), 4(R5)     // a3900029
	STPW.W	(R3, R4), 4(R5)     // a3908029
	STPW.P	(R3, R4), 4(R5)     // a3908028
	STPW	(R3, R4), -4(R5)    // a3903f29
	STPW	(R3, R4), -5(R5)    // bb1400d163130029
	STPW	(R3, R4), 11(R0)    // 1b2c009163130029
	STPW	(R3, R4), 1024(R0)  // 1b00109163130029
	STPW	(R3, R4), (RSP)     // e3130029
	STPW	(R3, R4), 4(RSP)    // e3930029
	STPW.W	(R3, R4), 4(RSP)    // e3938029
	STPW.P	(R3, R4), 4(RSP)    // e3938028
	STPW	(R3, R4), -4(RSP)   // e3933f29
	STPW	(R3, R4), 11(RSP)   // fb2f009163130029
	STPW	(R3, R4), 1024(RSP) // fb03109163130029
	STPW	(R3, R4), x(SB)
	STPW	(R3, R4), x+8(SB)

// END
//
//	LTYPEE comma
//	{
//		outcode($1, &nullgen, NREG, &nullgen);
//	}
	END
