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
	ADD	$0x000aaa, R2, R3 // ADD $2730, R2, R3     // 43a82a91
	ADD	$0x000aaa, R2     // ADD $2730, R2         // 42a82a91
	ADD	$0xaaa000, R2, R3 // ADD $11182080, R2, R3 // 43a86a91
	ADD	$0xaaa000, R2     // ADD $11182080, R2     // 42a86a91
	ADD	$0xaaaaaa, R2, R3 // ADD $11184810, R2, R3 // 43a82a9163a86a91
	ADD	$0xaaaaaa, R2     // ADD $11184810, R2     // 42a82a9142a86a91
	SUB	$0x000aaa, R2, R3 // SUB $2730, R2, R3     // 43a82ad1
	SUB	$0x000aaa, R2     // SUB $2730, R2         // 42a82ad1
	SUB	$0xaaa000, R2, R3 // SUB $11182080, R2, R3 // 43a86ad1
	SUB	$0xaaa000, R2     // SUB $11182080, R2     // 42a86ad1
	SUB	$0xaaaaaa, R2, R3 // SUB $11184810, R2, R3 // 43a82ad163a86ad1
	SUB	$0xaaaaaa, R2     // SUB $11184810, R2     // 42a82ad142a86ad1
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
	ADD	R19.UXTX, R14, R17              // d161338b
	ADDSW	R19.UXTW, R14, R17              // d141332b
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
	MOVWU	(R19)(R20<<2), R20              // 747a74b8
	MOVD	(R2)(R6<<3), R4                 // 447866f8
	MOVD	(R3)(R7.SXTX<<3), R8            // 68f867f8
	MOVWU	(R5)(R4.UXTW), R10              // aa4864b8
	MOVBU	(R3)(R9.UXTW), R8               // 68486938
	MOVBU	(R5)(R8), R10                   // MOVBU	(R5)(R8*1), R10         // aa686838
	MOVHU	(R2)(R7.SXTW<<1), R11           // 4bd86778
	MOVHU	(R1)(R2<<1), R5                 // 25786278
	MOVB	(R9)(R3.UXTW), R6               // 2649a338
	MOVB	(R10)(R6), R15                  // MOVB	(R10)(R6*1), R15                // 4f69a638
	MOVH	(R5)(R7.SXTX<<1), R19           // b3f8a778
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
	FMOVS	$(4.0), F0                      // 0010221e
	FMOVD	$(4.0), F0                      // 0010621e
	FMOVS	$(0.265625), F1                 // 01302a1e
	FMOVD	$(0.1796875), F2                // 02f0681e
	FMOVS	$(0.96875), F3                  // 03f02d1e
	FMOVD	$(28.0), F4                     // 0490671e

	FMOVS	(R2)(R6), F4       // FMOVS (R2)(R6*1), F4    // 446866bc
	FMOVS	(R2)(R6<<2), F4                               // 447866bc
	FMOVD	(R2)(R6), F4       // FMOVD (R2)(R6*1), F4    // 446866fc
	FMOVD	(R2)(R6<<3), F4                               // 447866fc
	FMOVS	F4, (R2)(R6)       // FMOVS F4, (R2)(R6*1)    // 446826bc
	FMOVS	F4, (R2)(R6<<2)                               // 447826bc
	FMOVD	F4, (R2)(R6)       // FMOVD F4, (R2)(R6*1)    // 446826fc
	FMOVD	F4, (R2)(R6<<3)                               // 447826fc

	CMPW	$40960, R0                      // 1f284071
	CMPW	$27745, R2                      // 3b8c8d525f001b6b
	CMNW	$0x3fffffc0, R2                 // CMNW	$1073741760, R2                   // fb5f1a325f001b2b
	CMPW	$0xffff0, R1                    // CMPW	$1048560, R1                      // fb3f1c323f001b6b
	CMP	$0xffffffffffa0, R3             // CMP	$281474976710560, R3              // fb0b80921b00e0f27f001beb
	CMP	$0xf4240, R1                    // CMP	$1000000, R1                      // 1b4888d2fb01a0f23f001beb
	ADD	$0x186a0, R2, R5                // ADD	$100000, R2, R5                   // 45801a91a5604091
	SUB	$0xe7791f700, R3, R1            // SUB	$62135596800, R3, R1              // 1be09ed23bf2aef2db01c0f261001bcb
	CMP     $3343198598084851058, R3        // 5bae8ed2db8daef23badcdf2bbcce5f27f001beb
	ADD	$0x3fffffffc000, R5             // ADD	$70368744161280, R5               // fb7f72b2a5001b8b
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

	ANDW	$0x3ff00000, R2 // ANDW	$1072693248, R2 // 42240c12
	BICW	$0x3ff00000, R2 // BICW	$1072693248, R2 // 42540212
	ORRW	$0x3ff00000, R2 // ORRW	$1072693248, R2 // 42240c32
	ORNW	$0x3ff00000, R2 // ORNW	$1072693248, R2 // 42540232
	EORW	$0x3ff00000, R2 // EORW	$1072693248, R2 // 42240c52
	EONW	$0x3ff00000, R2 // EONW	$1072693248, R2 // 42540252

	AND	$0x22220000, R3, R4   // AND $572653568, R3, R4   // 5b44a4d264001b8a
	ORR	$0x22220000, R3, R4   // ORR $572653568, R3, R4   // 5b44a4d264001baa
	EOR	$0x22220000, R3, R4   // EOR $572653568, R3, R4   // 5b44a4d264001bca
	BIC	$0x22220000, R3, R4   // BIC $572653568, R3, R4   // 5b44a4d264003b8a
	ORN	$0x22220000, R3, R4   // ORN $572653568, R3, R4   // 5b44a4d264003baa
	EON	$0x22220000, R3, R4   // EON $572653568, R3, R4   // 5b44a4d264003bca
	ANDS	$0x22220000, R3, R4   // ANDS $572653568, R3, R4  // 5b44a4d264001bea
	BICS	$0x22220000, R3, R4   // BICS $572653568, R3, R4  // 5b44a4d264003bea

	EOR	$0xe03fffffffffffff, R20, R22       // EOR	$-2287828610704211969, R20, R22 // 96e243d2
	TSTW	$0x600000006, R1                    // TSTW	$25769803782, R1                // 3f041f72
	TST	$0x4900000049, R0                   // TST	$313532612681, R0               // 3b0980d23b09c0f21f001bea
	ORR	$0x170000, R2, R1                   // ORR	$1507328, R2, R1                // fb02a0d241001baa
	AND	$0xff00ff, R2                       // AND	$16711935, R2                   // fb1f80d2fb1fa0f242001b8a
	AND	$0xff00ffff, R1                     // AND	$4278255615, R1                 // fbff9fd21be0bff221001b8a
	ANDS	$0xffff, R2                         // ANDS	$65535, R2                      // 423c40f2
	AND	$0x7fffffff, R3                     // AND	$2147483647, R3                 // 63784092
	ANDS	$0x0ffffffff80000000, R2            // ANDS	$-2147483648, R2                // 428061f2
	AND	$0xfffff, R2                        // AND	$1048575, R2                    // 424c4092
	ANDW	$0xf00fffff, R1                     // ANDW	$4027580415, R1                 // 215c0412
	ANDSW	$0xff00ffff, R1                     // ANDSW	$4278255615, R1                 // 215c0872
	TST	$0x11223344, R2                     // TST	$287454020, R2                  // 9b6886d25b24a2f25f001bea
	TSTW	$0xa000, R3                         // TSTW	$40960, R3                      // 1b0094527f001b6a
	BICW	$0xa000, R3                         // BICW	$40960, R3                      // 1b00945263003b0a
	ORRW	$0x1b000, R2, R3                    // ORRW	$110592, R2, R3                 // 1b0096523b00a07243001b2a
	TSTW	$0x500000, R1                       // TSTW	$5242880, R1                    // 1b0aa0523f001b6a
	TSTW	$0xff00ff, R1                       // TSTW	$16711935, R1                   // 3f9c0072
	TSTW	$0x60060, R5                        // TSTW	$393312, R5                     // 1b0c8052db00a072bf001b6a
	TSTW	$0x6006000060060, R5                // TSTW	$1689262177517664, R5           // 1b0c8052db00a072bf001b6a
	ANDW	$0x6006000060060, R5                // ANDW	$1689262177517664, R5           // 1b0c8052db00a072a5001b0a
	ANDSW	$0x6006000060060, R5                // ANDSW	$1689262177517664, R5           // 1b0c8052db00a072a5001b6a
	EORW	$0x6006000060060, R5                // EORW	$1689262177517664, R5           // 1b0c8052db00a072a5001b4a
	ORRW	$0x6006000060060, R5                // ORRW	$1689262177517664, R5           // 1b0c8052db00a072a5001b2a
	BICW	$0x6006000060060, R5                // BICW	$1689262177517664, R5           // 1b0c8052db00a072a5003b0a
	EONW	$0x6006000060060, R5                // EONW	$1689262177517664, R5           // 1b0c8052db00a072a5003b4a
	ORNW	$0x6006000060060, R5                // ORNW	$1689262177517664, R5           // 1b0c8052db00a072a5003b2a
	BICSW	$0x6006000060060, R5                // BICSW	$1689262177517664, R5           // 1b0c8052db00a072a5003b6a
	ADDW	$0x60060, R2                        // ADDW	$393312, R2                     // 4280011142804111
	CMPW	$0x60060, R2                        // CMPW	$393312, R2                     // 1b0c8052db00a0725f001b6b

	AND	$8, R0, RSP // 1f007d92
	ORR	$8, R0, RSP // 1f007db2
	EOR	$8, R0, RSP // 1f007dd2
	BIC	$8, R0, RSP // 1ff87c92
	ORN	$8, R0, RSP // 1ff87cb2
	EON	$8, R0, RSP // 1ff87cd2

	MOVD	$0x3fffffffc000, R0           // MOVD	$70368744161280, R0         // e07f72b2
	MOVW	$1000000, R4                  // 04488852e401a072
	MOVW	$0xaaaa0000, R1               // MOVW	$2863267840, R1             // 4155b552
	MOVW	$0xaaaaffff, R1               // MOVW	$2863333375, R1             // a1aaaa12
	MOVW	$0xaaaa, R1                   // MOVW	$43690, R1                  // 41559552
	MOVW	$0xffffaaaa, R1               // MOVW	$4294945450, R1             // a1aa8a12
	MOVW	$0xffff0000, R1               // MOVW	$4294901760, R1             // e1ffbf52
	MOVD	$0xffff00000000000, R1        // MOVD	$1152903912420802560, R1    // e13f54b2
	MOVD	$0x1111000000001111, R1       // MOVD	$1229764173248860433, R1    // 212282d22122e2f2
	MOVD	$0x1111ffff1111ffff, R1       // MOVD	$1230045644216991743, R1    // c1ddbd922122e2f2
	MOVD	$0x1111222233334444, R1       // MOVD	$1229801703532086340, R1    // 818888d26166a6f24144c4f22122e2f2
	MOVD	$0xaaaaffff, R1               // MOVD	$2863333375, R1             // e1ff9fd24155b5f2
	MOVD	$0x11110000, R1               // MOVD	$286326784, R1              // 2122a2d2
	MOVD	$0xaaaa0000aaaa1111, R1       // MOVD	$-6149102338357718767, R1   // 212282d24155b5f24155f5f2
	MOVD	$0x1111ffff1111aaaa, R1       // MOVD	$1230045644216969898, R1    // a1aa8a922122a2f22122e2f2
	MOVD	$0, R1                        // 010080d2
	MOVD	$-1, R1                       // 01008092
	MOVD	$0x210000, R0                 // MOVD	$2162688, R0                 // 2004a0d2
	MOVD	$0xffffffffffffaaaa, R1       // MOVD	$-21846, R1                  // a1aa8a92

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
	VLD2	(R29), [V23.H8, V24.H8]                         // b787404c
	VLD2.P	16(R0), [V18.B8, V19.B8]                        // 1280df0c
	VLD2.P	(R1)(R2), [V15.S2, V16.S2]                      // VLD2.P	(R1)(R2*1), [V15.S2,V16.S2] // 2f88c20c
	VLD3	(R27), [V11.S4, V12.S4, V13.S4]                 // 6b4b404c
	VLD3.P	48(RSP), [V11.S4, V12.S4, V13.S4]               // eb4bdf4c
	VLD3.P	(R30)(R2), [V14.D2, V15.D2, V16.D2]             // VLD3.P	(R30)(R2*1), [V14.D2,V15.D2,V16.D2] // ce4fc24c
	VLD4	(R15), [V10.H4, V11.H4, V12.H4, V13.H4]         // ea05400c
	VLD4.P	32(R24), [V31.B8, V0.B8, V1.B8, V2.B8]          // 1f03df0c
	VLD4.P	(R13)(R9), [V14.S2, V15.S2, V16.S2, V17.S2]     // VLD4.P	(R13)(R9*1), [V14.S2,V15.S2,V16.S2,V17.S2] // ae09c90c
	VLD1R	(R0), [V0.B16]					// 00c0404d
	VLD1R.P	16(R0), [V0.B16]				// 00c0df4d
	VLD1R.P	(R15)(R1), [V15.H4]				// VLD1R.P	(R15)(R1*1), [V15.H4] // efc5c10d
	VLD2R	(R15), [V15.H4, V16.H4]				// efc5600d
	VLD2R.P	32(R0), [V0.D2, V1.D2]				// 00ccff4d
	VLD2R.P	(R0)(R5), [V31.D1, V0.D1]			// VLD2R.P	(R0)(R5*1), [V31.D1, V0.D1] // 1fcce50d
	VLD3R	(RSP), [V31.S2, V0.S2, V1.S2]			// ffeb400d
	VLD3R.P	24(R15), [V15.H4, V16.H4, V17.H4]		// efe5df0d
	VLD3R.P	(R15)(R6), [V15.H8, V16.H8, V17.H8]		// VLD3R.P	(R15)(R6*1), [V15.H8, V16.H8, V17.H8] // efe5c64d
	VLD4R	(R0), [V0.B8, V1.B8, V2.B8, V3.B8]		// 00e0600d
	VLD4R.P	64(RSP), [V31.S4, V0.S4, V1.S4, V2.S4]		// ffebff4d
	VLD4R.P	(R15)(R9), [V15.H4, V16.H4, V17.H4, V18.H4]	// VLD4R.P	(R15)(R9*1), [V15.H4, V16.H4, V17.H4, V18.H4] // efe5e90d
	VST1.P	[V24.S2], 8(R2)                                 // 58789f0c
	VST1	[V29.S2, V30.S2], (R29)                         // bdab000c
	VST1	[V14.H4, V15.H4, V16.H4], (R27)                 // 6e67000c
	VST1.P	V4.B[15], 1(R0)                                 // 041c9f4d
	VST1.P	V4.H[7], 2(R0)                                  // 04589f4d
	VST1.P	V4.S[3], 4(R0)                                  // 04909f4d
	VST1.P	V4.D[1], 8(R0)                                  // 04849f4d
	VST1.P	V4.D[1], (R0)(R1)                               // VST1.P	V4.D[1], (R0)(R1*1) // 0484814d
	VST1	V4.D[1], (R0)                                   // 0484004d
	VST2	[V22.H8, V23.H8], (R23)                         // f686004c
	VST2.P	[V14.H4, V15.H4], 16(R17)                       // 2e869f0c
	VST2.P	[V14.H4, V15.H4], (R3)(R17)                     // VST2.P	[V14.H4,V15.H4], (R3)(R17*1) // 6e84910c
	VST3	[V1.D2, V2.D2, V3.D2], (R11)                    // 614d004c
	VST3.P	[V18.S4, V19.S4, V20.S4], 48(R25)               // 324b9f4c
	VST3.P	[V19.B8, V20.B8, V21.B8], (R3)(R7)              // VST3.P	[V19.B8, V20.B8, V21.B8], (R3)(R7*1) // 7340870c
	VST4	[V22.D2, V23.D2, V24.D2, V25.D2], (R3)          // 760c004c
	VST4.P	[V14.D2, V15.D2, V16.D2, V17.D2], 64(R15)       // ee0d9f4c
	VST4.P	[V24.B8, V25.B8, V26.B8, V27.B8], (R3)(R23)     // VST4.P	[V24.B8, V25.B8, V26.B8, V27.B8], (R3)(R23*1) // 7800970c
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
	CMPW	$0x22220000, RSP  // CMPW $572653568, RSP  // 5b44a452ff633b6b

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
//	FADDD	$0.5, F1 // FADDD $(0.5), F1
	FADDD	F1, F2

//		LTYPEK frcon ',' freg ',' freg
//	{
//		outcode($1, &$2, $4.reg, &$6);
//	}
//	FADDD	$0.7, F1, F2 // FADDD	$(0.69999999999999996), F1, F2
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
	SWPAD	R5, (R6), R7                         // c780a5f8
	SWPAD	R5, (RSP), R7                        // e783a5f8
	SWPAW	R5, (R6), R7                         // c780a5b8
	SWPAW	R5, (RSP), R7                        // e783a5b8
	SWPAH	R5, (R6), R7                         // c780a578
	SWPAH	R5, (RSP), R7                        // e783a578
	SWPAB	R5, (R6), R7                         // c780a538
	SWPAB	R5, (RSP), R7                        // e783a538
	SWPALD	R5, (R6), R7                         // c780e5f8
	SWPALD	R5, (RSP), R7                        // e783e5f8
	SWPALW	R5, (R6), R7                         // c780e5b8
	SWPALW	R5, (RSP), R7                        // e783e5b8
	SWPALH	R5, (R6), R7                         // c780e578
	SWPALH	R5, (RSP), R7                        // e783e578
	SWPALB	R5, (R6), R7                         // c780e538
	SWPALB	R5, (RSP), R7                        // e783e538
	SWPD	R5, (R6), R7                         // c78025f8
	SWPD	R5, (RSP), R7                        // e78325f8
	SWPW	R5, (R6), R7                         // c78025b8
	SWPW	R5, (RSP), R7                        // e78325b8
	SWPH	R5, (R6), R7                         // c7802578
	SWPH	R5, (RSP), R7                        // e7832578
	SWPB	R5, (R6), R7                         // c7802538
	SWPB	R5, (RSP), R7                        // e7832538
	SWPLD	R5, (R6), R7                         // c78065f8
	SWPLD	R5, (RSP), R7                        // e78365f8
	SWPLW	R5, (R6), R7                         // c78065b8
	SWPLW	R5, (RSP), R7                        // e78365b8
	SWPLH	R5, (R6), R7                         // c7806578
	SWPLH	R5, (RSP), R7                        // e7836578
	SWPLB	R5, (R6), R7                         // c7806538
	SWPLB	R5, (RSP), R7                        // e7836538
	LDADDAD	R5, (R6), R7                         // c700a5f8
	LDADDAD	R5, (RSP), R7                        // e703a5f8
	LDADDAW	R5, (R6), R7                         // c700a5b8
	LDADDAW	R5, (RSP), R7                        // e703a5b8
	LDADDAH	R5, (R6), R7                         // c700a578
	LDADDAH	R5, (RSP), R7                        // e703a578
	LDADDAB	R5, (R6), R7                         // c700a538
	LDADDAB	R5, (RSP), R7                        // e703a538
	LDADDALD	R5, (R6), R7                 // c700e5f8
	LDADDALD	R5, (RSP), R7                // e703e5f8
	LDADDALW	R5, (R6), R7                 // c700e5b8
	LDADDALW	R5, (RSP), R7                // e703e5b8
	LDADDALH	R5, (R6), R7                 // c700e578
	LDADDALH	R5, (RSP), R7                // e703e578
	LDADDALB	R5, (R6), R7                 // c700e538
	LDADDALB	R5, (RSP), R7                // e703e538
	LDADDD	R5, (R6), R7                         // c70025f8
	LDADDD	R5, (RSP), R7                        // e70325f8
	LDADDW	R5, (R6), R7                         // c70025b8
	LDADDW	R5, (RSP), R7                        // e70325b8
	LDADDH	R5, (R6), R7                         // c7002578
	LDADDH	R5, (RSP), R7                        // e7032578
	LDADDB	R5, (R6), R7                         // c7002538
	LDADDB	R5, (RSP), R7                        // e7032538
	LDADDLD	R5, (R6), R7                         // c70065f8
	LDADDLD	R5, (RSP), R7                        // e70365f8
	LDADDLW	R5, (R6), R7                         // c70065b8
	LDADDLW	R5, (RSP), R7                        // e70365b8
	LDADDLH	R5, (R6), R7                         // c7006578
	LDADDLH	R5, (RSP), R7                        // e7036578
	LDADDLB	R5, (R6), R7                         // c7006538
	LDADDLB	R5, (RSP), R7                        // e7036538
	LDANDAD	R5, (R6), R7                         // c710a5f8
	LDANDAD	R5, (RSP), R7                        // e713a5f8
	LDANDAW	R5, (R6), R7                         // c710a5b8
	LDANDAW	R5, (RSP), R7                        // e713a5b8
	LDANDAH	R5, (R6), R7                         // c710a578
	LDANDAH	R5, (RSP), R7                        // e713a578
	LDANDAB	R5, (R6), R7                         // c710a538
	LDANDAB	R5, (RSP), R7                        // e713a538
	LDANDALD	R5, (R6), R7                 // c710e5f8
	LDANDALD	R5, (RSP), R7                // e713e5f8
	LDANDALW	R5, (R6), R7                 // c710e5b8
	LDANDALW	R5, (RSP), R7                // e713e5b8
	LDANDALH	R5, (R6), R7                 // c710e578
	LDANDALH	R5, (RSP), R7                // e713e578
	LDANDALB	R5, (R6), R7                 // c710e538
	LDANDALB	R5, (RSP), R7                // e713e538
	LDANDD	R5, (R6), R7                         // c71025f8
	LDANDD	R5, (RSP), R7                        // e71325f8
	LDANDW	R5, (R6), R7                         // c71025b8
	LDANDW	R5, (RSP), R7                        // e71325b8
	LDANDH	R5, (R6), R7                         // c7102578
	LDANDH	R5, (RSP), R7                        // e7132578
	LDANDB	R5, (R6), R7                         // c7102538
	LDANDB	R5, (RSP), R7                        // e7132538
	LDANDLD	R5, (R6), R7                         // c71065f8
	LDANDLD	R5, (RSP), R7                        // e71365f8
	LDANDLW	R5, (R6), R7                         // c71065b8
	LDANDLW	R5, (RSP), R7                        // e71365b8
	LDANDLH	R5, (R6), R7                         // c7106578
	LDANDLH	R5, (RSP), R7                        // e7136578
	LDANDLB	R5, (R6), R7                         // c7106538
	LDANDLB	R5, (RSP), R7                        // e7136538
	LDEORAD	R5, (R6), R7                         // c720a5f8
	LDEORAD	R5, (RSP), R7                        // e723a5f8
	LDEORAW	R5, (R6), R7                         // c720a5b8
	LDEORAW	R5, (RSP), R7                        // e723a5b8
	LDEORAH	R5, (R6), R7                         // c720a578
	LDEORAH	R5, (RSP), R7                        // e723a578
	LDEORAB	R5, (R6), R7                         // c720a538
	LDEORAB	R5, (RSP), R7                        // e723a538
	LDEORALD	R5, (R6), R7                 // c720e5f8
	LDEORALD	R5, (RSP), R7                // e723e5f8
	LDEORALW	R5, (R6), R7                 // c720e5b8
	LDEORALW	R5, (RSP), R7                // e723e5b8
	LDEORALH	R5, (R6), R7                 // c720e578
	LDEORALH	R5, (RSP), R7                // e723e578
	LDEORALB	R5, (R6), R7                 // c720e538
	LDEORALB	R5, (RSP), R7                // e723e538
	LDEORD	R5, (R6), R7                         // c72025f8
	LDEORD	R5, (RSP), R7                        // e72325f8
	LDEORW	R5, (R6), R7                         // c72025b8
	LDEORW	R5, (RSP), R7                        // e72325b8
	LDEORH	R5, (R6), R7                         // c7202578
	LDEORH	R5, (RSP), R7                        // e7232578
	LDEORB	R5, (R6), R7                         // c7202538
	LDEORB	R5, (RSP), R7                        // e7232538
	LDEORLD	R5, (R6), R7                         // c72065f8
	LDEORLD	R5, (RSP), R7                        // e72365f8
	LDEORLW	R5, (R6), R7                         // c72065b8
	LDEORLW	R5, (RSP), R7                        // e72365b8
	LDEORLH	R5, (R6), R7                         // c7206578
	LDEORLH	R5, (RSP), R7                        // e7236578
	LDEORLB	R5, (R6), R7                         // c7206538
	LDEORLB	R5, (RSP), R7                        // e7236538
	LDORAD	R5, (R6), R7                         // c730a5f8
	LDORAD	R5, (RSP), R7                        // e733a5f8
	LDORAW	R5, (R6), R7                         // c730a5b8
	LDORAW	R5, (RSP), R7                        // e733a5b8
	LDORAH	R5, (R6), R7                         // c730a578
	LDORAH	R5, (RSP), R7                        // e733a578
	LDORAB	R5, (R6), R7                         // c730a538
	LDORAB	R5, (RSP), R7                        // e733a538
	LDORALD	R5, (R6), R7                         // c730e5f8
	LDORALD	R5, (RSP), R7                        // e733e5f8
	LDORALW	R5, (R6), R7                         // c730e5b8
	LDORALW	R5, (RSP), R7                        // e733e5b8
	LDORALH	R5, (R6), R7                         // c730e578
	LDORALH	R5, (RSP), R7                        // e733e578
	LDORALB	R5, (R6), R7                         // c730e538
	LDORALB	R5, (RSP), R7                        // e733e538
	LDORD	R5, (R6), R7                         // c73025f8
	LDORD	R5, (RSP), R7                        // e73325f8
	LDORW	R5, (R6), R7                         // c73025b8
	LDORW	R5, (RSP), R7                        // e73325b8
	LDORH	R5, (R6), R7                         // c7302578
	LDORH	R5, (RSP), R7                        // e7332578
	LDORB	R5, (R6), R7                         // c7302538
	LDORB	R5, (RSP), R7                        // e7332538
	LDORLD	R5, (R6), R7                         // c73065f8
	LDORLD	R5, (RSP), R7                        // e73365f8
	LDORLW	R5, (R6), R7                         // c73065b8
	LDORLW	R5, (RSP), R7                        // e73365b8
	LDORLH	R5, (R6), R7                         // c7306578
	LDORLH	R5, (RSP), R7                        // e7336578
	LDORLB	R5, (R6), R7                         // c7306538
	LDORLB	R5, (RSP), R7                        // e7336538
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
	LDP	(R0), (R0, R1)      // 000440a9
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

// bit field operation
	BFI	$0, R1, $1, R2      // 220040b3
	BFIW	$0, R1, $1, R2      // 22000033
	SBFIZ	$0, R1, $1, R2      // 22004093
	SBFIZW	$0, R1, $1, R2      // 22000013
	UBFIZ	$0, R1, $1, R2      // 220040d3
	UBFIZW	$0, R1, $1, R2      // 22000053

// FSTPD/FSTPS/FLDPD/FLDPS
	FLDPD	(R0), (F1, F2)      // 0108406d
	FLDPD	8(R0), (F1, F2)     // 0188406d
	FLDPD	-8(R0), (F1, F2)    // 01887f6d
	FLDPD	11(R0), (F1, F2)    // 1b2c0091610b406d
	FLDPD	1024(R0), (F1, F2)  // 1b001091610b406d
	FLDPD.W	8(R0), (F1, F2)     // 0188c06d
	FLDPD.P	8(R0), (F1, F2)     // 0188c06c
	FLDPD	(RSP), (F1, F2)     // e10b406d
	FLDPD	8(RSP), (F1, F2)    // e18b406d
	FLDPD	-8(RSP), (F1, F2)   // e18b7f6d
	FLDPD	11(RSP), (F1, F2)   // fb2f0091610b406d
	FLDPD	1024(RSP), (F1, F2) // fb031091610b406d
	FLDPD.W	8(RSP), (F1, F2)    // e18bc06d
	FLDPD.P	8(RSP), (F1, F2)    // e18bc06c
	FLDPD	-31(R0), (F1, F2)   // 1b7c00d1610b406d
	FLDPD	-4(R0), (F1, F2)    // 1b1000d1610b406d
	FLDPD	-8(R0), (F1, F2)    // 01887f6d
	FLDPD	x(SB), (F1, F2)
	FLDPD	x+8(SB), (F1, F2)
	FLDPS	-5(R0), (F1, F2)    // 1b1400d1610b402d
	FLDPS	(R0), (F1, F2)      // 0108402d
	FLDPS	4(R0), (F1, F2)     // 0188402d
	FLDPS	-4(R0), (F1, F2)    // 01887f2d
	FLDPS.W	4(R0), (F1, F2)     // 0188c02d
	FLDPS.P	4(R0), (F1, F2)     // 0188c02c
	FLDPS	11(R0), (F1, F2)    // 1b2c0091610b402d
	FLDPS	1024(R0), (F1, F2)  // 1b001091610b402d
	FLDPS	(RSP), (F1, F2)     // e10b402d
	FLDPS	4(RSP), (F1, F2)    // e18b402d
	FLDPS	-4(RSP), (F1, F2)   // e18b7f2d
	FLDPS.W	4(RSP), (F1, F2)    // e18bc02d
	FLDPS.P	4(RSP), (F1, F2)    // e18bc02c
	FLDPS	11(RSP), (F1, F2)   // fb2f0091610b402d
	FLDPS	1024(RSP), (F1, F2) // fb031091610b402d
	FLDPS	x(SB), (F1, F2)
	FLDPS	x+8(SB), (F1, F2)
	FSTPD	(F3, F4), (R5)      // a310006d
	FSTPD	(F3, F4), 8(R5)     // a390006d
	FSTPD.W	(F3, F4), 8(R5)     // a390806d
	FSTPD.P	(F3, F4), 8(R5)     // a390806c
	FSTPD	(F3, F4), -8(R5)    // a3903f6d
	FSTPD	(F3, F4), -4(R5)    // bb1000d16313006d
	FSTPD	(F3, F4), 11(R0)    // 1b2c00916313006d
	FSTPD	(F3, F4), 1024(R0)  // 1b0010916313006d
	FSTPD	(F3, F4), (RSP)     // e313006d
	FSTPD	(F3, F4), 8(RSP)    // e393006d
	FSTPD.W	(F3, F4), 8(RSP)    // e393806d
	FSTPD.P	(F3, F4), 8(RSP)    // e393806c
	FSTPD	(F3, F4), -8(RSP)   // e3933f6d
	FSTPD	(F3, F4), 11(RSP)   // fb2f00916313006d
	FSTPD	(F3, F4), 1024(RSP) // fb0310916313006d
	FSTPD	(F3, F4), x(SB)
	FSTPD	(F3, F4), x+8(SB)
	FSTPS	(F3, F4), (R5)      // a310002d
	FSTPS	(F3, F4), 4(R5)     // a390002d
	FSTPS.W	(F3, F4), 4(R5)     // a390802d
	FSTPS.P	(F3, F4), 4(R5)     // a390802c
	FSTPS	(F3, F4), -4(R5)    // a3903f2d
	FSTPS	(F3, F4), -5(R5)    // bb1400d16313002d
	FSTPS	(F3, F4), 11(R0)    // 1b2c00916313002d
	FSTPS	(F3, F4), 1024(R0)  // 1b0010916313002d
	FSTPS	(F3, F4), (RSP)     // e313002d
	FSTPS	(F3, F4), 4(RSP)    // e393002d
	FSTPS.W	(F3, F4), 4(RSP)    // e393802d
	FSTPS.P	(F3, F4), 4(RSP)    // e393802c
	FSTPS	(F3, F4), -4(RSP)   // e3933f2d
	FSTPS	(F3, F4), 11(RSP)   // fb2f00916313002d
	FSTPS	(F3, F4), 1024(RSP) // fb0310916313002d
	FSTPS	(F3, F4), x(SB)
	FSTPS	(F3, F4), x+8(SB)
	NOOP                        // 1f2003d5
	HINT $0                     // 1f2003d5

// System Register
	MSR	$1, SPSel                          // bf4100d5
	MSR	$9, DAIFSet                        // df4903d5
	MSR	$6, DAIFClr                        // ff4603d5
	MRS	ELR_EL1, R8                        // 284038d5
	MSR	R16, ELR_EL1                       // 304018d5
	MSR	R2, ACTLR_EL1                      // 221018d5
	MRS	TCR_EL1, R5                        // 452038d5
	MRS	PMEVCNTR15_EL0, R12                // ece93bd5
	MSR	R20, PMEVTYPER26_EL0               // 54ef1bd5
	MSR	R10, DBGBCR15_EL1                  // aa0f10d5
	MRS	ACTLR_EL1, R3                      // 231038d5
	MSR	R9, ACTLR_EL1                      // 291018d5
	MRS	AFSR0_EL1, R10                     // 0a5138d5
	MSR	R1, AFSR0_EL1                      // 015118d5
	MRS	AFSR0_EL1, R9                      // 095138d5
	MSR	R30, AFSR0_EL1                     // 1e5118d5
	MRS	AFSR1_EL1, R0                      // 205138d5
	MSR	R1, AFSR1_EL1                      // 215118d5
	MRS	AFSR1_EL1, R8                      // 285138d5
	MSR	R19, AFSR1_EL1                     // 335118d5
	MRS	AIDR_EL1, R11                      // eb0039d5
	MRS	AMAIR_EL1, R0                      // 00a338d5
	MSR	R22, AMAIR_EL1                     // 16a318d5
	MRS	AMAIR_EL1, R14                     // 0ea338d5
	MSR	R0, AMAIR_EL1                      // 00a318d5
	MRS	APDAKeyHi_EL1, R16                 // 302238d5
	MSR	R26, APDAKeyHi_EL1                 // 3a2218d5
	MRS	APDAKeyLo_EL1, R21                 // 152238d5
	MSR	R22, APDAKeyLo_EL1                 // 162218d5
	MRS	APDBKeyHi_EL1, R2                  // 622238d5
	MSR	R6, APDBKeyHi_EL1                  // 662218d5
	MRS	APDBKeyLo_EL1, R5                  // 452238d5
	MSR	R22, APDBKeyLo_EL1                 // 562218d5
	MRS	APGAKeyHi_EL1, R22                 // 362338d5
	MSR	R5, APGAKeyHi_EL1                  // 252318d5
	MRS	APGAKeyLo_EL1, R16                 // 102338d5
	MSR	R22, APGAKeyLo_EL1                 // 162318d5
	MRS	APIAKeyHi_EL1, R23                 // 372138d5
	MSR	R17, APIAKeyHi_EL1                 // 312118d5
	MRS	APIAKeyLo_EL1, R16                 // 102138d5
	MSR	R6, APIAKeyLo_EL1                  // 062118d5
	MRS	APIBKeyHi_EL1, R10                 // 6a2138d5
	MSR	R11, APIBKeyHi_EL1                 // 6b2118d5
	MRS	APIBKeyLo_EL1, R25                 // 592138d5
	MSR	R22, APIBKeyLo_EL1                 // 562118d5
	MRS	CCSIDR_EL1, R25                    // 190039d5
	MRS	CLIDR_EL1, R16                     // 300039d5
	MRS	CNTFRQ_EL0, R20                    // 14e03bd5
	MSR	R16, CNTFRQ_EL0                    // 10e01bd5
	MRS	CNTKCTL_EL1, R26                   // 1ae138d5
	MSR	R0, CNTKCTL_EL1                    // 00e118d5
	MRS	CNTP_CTL_EL0, R14                  // 2ee23bd5
	MSR	R17, CNTP_CTL_EL0                  // 31e21bd5
	MRS	CNTP_CVAL_EL0, R15                 // 4fe23bd5
	MSR	R8, CNTP_CVAL_EL0                  // 48e21bd5
	MRS	CNTP_TVAL_EL0, R6                  // 06e23bd5
	MSR	R29, CNTP_TVAL_EL0                 // 1de21bd5
	MRS	CNTP_CTL_EL0, R22                  // 36e23bd5
	MSR	R0, CNTP_CTL_EL0                   // 20e21bd5
	MRS	CNTP_CVAL_EL0, R9                  // 49e23bd5
	MSR	R4, CNTP_CVAL_EL0                  // 44e21bd5
	MRS	CNTP_TVAL_EL0, R27                 // 1be23bd5
	MSR	R17, CNTP_TVAL_EL0                 // 11e21bd5
	MRS	CNTV_CTL_EL0, R27                  // 3be33bd5
	MSR	R2, CNTV_CTL_EL0                   // 22e31bd5
	MRS	CNTV_CVAL_EL0, R16                 // 50e33bd5
	MSR	R27, CNTV_CVAL_EL0                 // 5be31bd5
	MRS	CNTV_TVAL_EL0, R12                 // 0ce33bd5
	MSR	R19, CNTV_TVAL_EL0                 // 13e31bd5
	MRS	CNTV_CTL_EL0, R14                  // 2ee33bd5
	MSR	R2, CNTV_CTL_EL0                   // 22e31bd5
	MRS	CNTV_CVAL_EL0, R8                  // 48e33bd5
	MSR	R26, CNTV_CVAL_EL0                 // 5ae31bd5
	MRS	CNTV_TVAL_EL0, R6                  // 06e33bd5
	MSR	R19, CNTV_TVAL_EL0                 // 13e31bd5
	MRS	CNTKCTL_EL1, R16                   // 10e138d5
	MSR	R26, CNTKCTL_EL1                   // 1ae118d5
	MRS	CNTPCT_EL0, R9                     // 29e03bd5
	MRS	CNTPS_CTL_EL1, R30                 // 3ee23fd5
	MSR	R26, CNTPS_CTL_EL1                 // 3ae21fd5
	MRS	CNTPS_CVAL_EL1, R8                 // 48e23fd5
	MSR	R26, CNTPS_CVAL_EL1                // 5ae21fd5
	MRS	CNTPS_TVAL_EL1, R7                 // 07e23fd5
	MSR	R13, CNTPS_TVAL_EL1                // 0de21fd5
	MRS	CNTP_CTL_EL0, R2                   // 22e23bd5
	MSR	R10, CNTP_CTL_EL0                  // 2ae21bd5
	MRS	CNTP_CVAL_EL0, R6                  // 46e23bd5
	MSR	R21, CNTP_CVAL_EL0                 // 55e21bd5
	MRS	CNTP_TVAL_EL0, R27                 // 1be23bd5
	MSR	R29, CNTP_TVAL_EL0                 // 1de21bd5
	MRS	CNTVCT_EL0, R13                    // 4de03bd5
	MRS	CNTV_CTL_EL0, R30                  // 3ee33bd5
	MSR	R19, CNTV_CTL_EL0                  // 33e31bd5
	MRS	CNTV_CVAL_EL0, R27                 // 5be33bd5
	MSR	R24, CNTV_CVAL_EL0                 // 58e31bd5
	MRS	CNTV_TVAL_EL0, R24                 // 18e33bd5
	MSR	R5, CNTV_TVAL_EL0                  // 05e31bd5
	MRS	CONTEXTIDR_EL1, R15                // 2fd038d5
	MSR	R27, CONTEXTIDR_EL1                // 3bd018d5
	MRS	CONTEXTIDR_EL1, R29                // 3dd038d5
	MSR	R24, CONTEXTIDR_EL1                // 38d018d5
	MRS	CPACR_EL1, R10                     // 4a1038d5
	MSR	R14, CPACR_EL1                     // 4e1018d5
	MRS	CPACR_EL1, R27                     // 5b1038d5
	MSR	R22, CPACR_EL1                     // 561018d5
	MRS	CSSELR_EL1, R3                     // 03003ad5
	MSR	R4, CSSELR_EL1                     // 04001ad5
	MRS	CTR_EL0, R15                       // 2f003bd5
	MRS	CurrentEL, R1                      // 414238d5
	MRS	DAIF, R24                          // 38423bd5
	MSR	R9, DAIF                           // 29421bd5
	MRS	DBGAUTHSTATUS_EL1, R5              // c57e30d5
	MRS	DBGBCR0_EL1, R29                   // bd0030d5
	MRS	DBGBCR1_EL1, R13                   // ad0130d5
	MRS	DBGBCR2_EL1, R22                   // b60230d5
	MRS	DBGBCR3_EL1, R8                    // a80330d5
	MRS	DBGBCR4_EL1, R2                    // a20430d5
	MRS	DBGBCR5_EL1, R4                    // a40530d5
	MRS	DBGBCR6_EL1, R2                    // a20630d5
	MRS	DBGBCR7_EL1, R6                    // a60730d5
	MRS	DBGBCR8_EL1, R1                    // a10830d5
	MRS	DBGBCR9_EL1, R16                   // b00930d5
	MRS	DBGBCR10_EL1, R23                  // b70a30d5
	MRS	DBGBCR11_EL1, R3                   // a30b30d5
	MRS	DBGBCR12_EL1, R6                   // a60c30d5
	MRS	DBGBCR13_EL1, R16                  // b00d30d5
	MRS	DBGBCR14_EL1, R4                   // a40e30d5
	MRS	DBGBCR15_EL1, R9                   // a90f30d5
	MSR	R4, DBGBCR0_EL1                    // a40010d5
	MSR	R14, DBGBCR1_EL1                   // ae0110d5
	MSR	R7, DBGBCR2_EL1                    // a70210d5
	MSR	R12, DBGBCR3_EL1                   // ac0310d5
	MSR	R6, DBGBCR4_EL1                    // a60410d5
	MSR	R11, DBGBCR5_EL1                   // ab0510d5
	MSR	R6, DBGBCR6_EL1                    // a60610d5
	MSR	R13, DBGBCR7_EL1                   // ad0710d5
	MSR	R17, DBGBCR8_EL1                   // b10810d5
	MSR	R17, DBGBCR9_EL1                   // b10910d5
	MSR	R22, DBGBCR10_EL1                  // b60a10d5
	MSR	R16, DBGBCR11_EL1                  // b00b10d5
	MSR	R24, DBGBCR12_EL1                  // b80c10d5
	MSR	R29, DBGBCR13_EL1                  // bd0d10d5
	MSR	R1, DBGBCR14_EL1                   // a10e10d5
	MSR	R10, DBGBCR15_EL1                  // aa0f10d5
	MRS	DBGBVR0_EL1, R16                   // 900030d5
	MRS	DBGBVR1_EL1, R21                   // 950130d5
	MRS	DBGBVR2_EL1, R13                   // 8d0230d5
	MRS	DBGBVR3_EL1, R12                   // 8c0330d5
	MRS	DBGBVR4_EL1, R20                   // 940430d5
	MRS	DBGBVR5_EL1, R21                   // 950530d5
	MRS	DBGBVR6_EL1, R27                   // 9b0630d5
	MRS	DBGBVR7_EL1, R6                    // 860730d5
	MRS	DBGBVR8_EL1, R14                   // 8e0830d5
	MRS	DBGBVR9_EL1, R5                    // 850930d5
	MRS	DBGBVR10_EL1, R9                   // 890a30d5
	MRS	DBGBVR11_EL1, R25                  // 990b30d5
	MRS	DBGBVR12_EL1, R30                  // 9e0c30d5
	MRS	DBGBVR13_EL1, R1                   // 810d30d5
	MRS	DBGBVR14_EL1, R17                  // 910e30d5
	MRS	DBGBVR15_EL1, R25                  // 990f30d5
	MSR	R15, DBGBVR0_EL1                   // 8f0010d5
	MSR	R6, DBGBVR1_EL1                    // 860110d5
	MSR	R24, DBGBVR2_EL1                   // 980210d5
	MSR	R17, DBGBVR3_EL1                   // 910310d5
	MSR	R3, DBGBVR4_EL1                    // 830410d5
	MSR	R21, DBGBVR5_EL1                   // 950510d5
	MSR	R5, DBGBVR6_EL1                    // 850610d5
	MSR	R6, DBGBVR7_EL1                    // 860710d5
	MSR	R25, DBGBVR8_EL1                   // 990810d5
	MSR	R4, DBGBVR9_EL1                    // 840910d5
	MSR	R25, DBGBVR10_EL1                  // 990a10d5
	MSR	R17, DBGBVR11_EL1                  // 910b10d5
	MSR	R0, DBGBVR12_EL1                   // 800c10d5
	MSR	R5, DBGBVR13_EL1                   // 850d10d5
	MSR	R9, DBGBVR14_EL1                   // 890e10d5
	MSR	R12, DBGBVR15_EL1                  // 8c0f10d5
	MRS	DBGCLAIMCLR_EL1, R27               // db7930d5
	MSR	R0, DBGCLAIMCLR_EL1                // c07910d5
	MRS	DBGCLAIMSET_EL1, R7                // c77830d5
	MSR	R13, DBGCLAIMSET_EL1               // cd7810d5
	MRS	DBGDTRRX_EL0, R0                   // 000533d5
	MSR	R29, DBGDTRTX_EL0                  // 1d0513d5
	MRS	DBGDTR_EL0, R27                    // 1b0433d5
	MSR	R30, DBGDTR_EL0                    // 1e0413d5
	MRS	DBGPRCR_EL1, R4                    // 841430d5
	MSR	R0, DBGPRCR_EL1                    // 801410d5
	MRS	DBGWCR0_EL1, R24                   // f80030d5
	MRS	DBGWCR1_EL1, R19                   // f30130d5
	MRS	DBGWCR2_EL1, R25                   // f90230d5
	MRS	DBGWCR3_EL1, R0                    // e00330d5
	MRS	DBGWCR4_EL1, R13                   // ed0430d5
	MRS	DBGWCR5_EL1, R8                    // e80530d5
	MRS	DBGWCR6_EL1, R22                   // f60630d5
	MRS	DBGWCR7_EL1, R11                   // eb0730d5
	MRS	DBGWCR8_EL1, R11                   // eb0830d5
	MRS	DBGWCR9_EL1, R3                    // e30930d5
	MRS	DBGWCR10_EL1, R17                  // f10a30d5
	MRS	DBGWCR11_EL1, R21                  // f50b30d5
	MRS	DBGWCR12_EL1, R10                  // ea0c30d5
	MRS	DBGWCR13_EL1, R22                  // f60d30d5
	MRS	DBGWCR14_EL1, R11                  // eb0e30d5
	MRS	DBGWCR15_EL1, R0                   // e00f30d5
	MSR	R24, DBGWCR0_EL1                   // f80010d5
	MSR	R8, DBGWCR1_EL1                    // e80110d5
	MSR	R17, DBGWCR2_EL1                   // f10210d5
	MSR	R29, DBGWCR3_EL1                   // fd0310d5
	MSR	R13, DBGWCR4_EL1                   // ed0410d5
	MSR	R22, DBGWCR5_EL1                   // f60510d5
	MSR	R3, DBGWCR6_EL1                    // e30610d5
	MSR	R4, DBGWCR7_EL1                    // e40710d5
	MSR	R7, DBGWCR8_EL1                    // e70810d5
	MSR	R29, DBGWCR9_EL1                   // fd0910d5
	MSR	R3, DBGWCR10_EL1                   // e30a10d5
	MSR	R11, DBGWCR11_EL1                  // eb0b10d5
	MSR	R20, DBGWCR12_EL1                  // f40c10d5
	MSR	R6, DBGWCR13_EL1                   // e60d10d5
	MSR	R22, DBGWCR14_EL1                  // f60e10d5
	MSR	R25, DBGWCR15_EL1                  // f90f10d5
	MRS	DBGWVR0_EL1, R14                   // ce0030d5
	MRS	DBGWVR1_EL1, R16                   // d00130d5
	MRS	DBGWVR2_EL1, R15                   // cf0230d5
	MRS	DBGWVR3_EL1, R1                    // c10330d5
	MRS	DBGWVR4_EL1, R26                   // da0430d5
	MRS	DBGWVR5_EL1, R14                   // ce0530d5
	MRS	DBGWVR6_EL1, R17                   // d10630d5
	MRS	DBGWVR7_EL1, R22                   // d60730d5
	MRS	DBGWVR8_EL1, R4                    // c40830d5
	MRS	DBGWVR9_EL1, R3                    // c30930d5
	MRS	DBGWVR10_EL1, R16                  // d00a30d5
	MRS	DBGWVR11_EL1, R2                   // c20b30d5
	MRS	DBGWVR12_EL1, R5                   // c50c30d5
	MRS	DBGWVR13_EL1, R23                  // d70d30d5
	MRS	DBGWVR14_EL1, R5                   // c50e30d5
	MRS	DBGWVR15_EL1, R6                   // c60f30d5
	MSR	R24, DBGWVR0_EL1                   // d80010d5
	MSR	R6, DBGWVR1_EL1                    // c60110d5
	MSR	R1, DBGWVR2_EL1                    // c10210d5
	MSR	R24, DBGWVR3_EL1                   // d80310d5
	MSR	R24, DBGWVR4_EL1                   // d80410d5
	MSR	R0, DBGWVR5_EL1                    // c00510d5
	MSR	R10, DBGWVR6_EL1                   // ca0610d5
	MSR	R17, DBGWVR7_EL1                   // d10710d5
	MSR	R7, DBGWVR8_EL1                    // c70810d5
	MSR	R8, DBGWVR9_EL1                    // c80910d5
	MSR	R15, DBGWVR10_EL1                  // cf0a10d5
	MSR	R8, DBGWVR11_EL1                   // c80b10d5
	MSR	R7, DBGWVR12_EL1                   // c70c10d5
	MSR	R14, DBGWVR13_EL1                  // ce0d10d5
	MSR	R16, DBGWVR14_EL1                  // d00e10d5
	MSR	R5, DBGWVR15_EL1                   // c50f10d5
	MRS	DCZID_EL0, R21                     // f5003bd5
	MRS	DISR_EL1, R8                       // 28c138d5
	MSR	R5, DISR_EL1                       // 25c118d5
	MRS	DIT, R29                           // bd423bd5
	MSR	R22, DIT                           // b6421bd5
	MRS	DLR_EL0, R25                       // 39453bd5
	MSR	R9, DLR_EL0                        // 29451bd5
	MRS	DSPSR_EL0, R3                      // 03453bd5
	MSR	R10, DSPSR_EL0                     // 0a451bd5
	MRS	ELR_EL1, R24                       // 384038d5
	MSR	R3, ELR_EL1                        // 234018d5
	MRS	ELR_EL1, R13                       // 2d4038d5
	MSR	R27, ELR_EL1                       // 3b4018d5
	MRS	ERRIDR_EL1, R30                    // 1e5338d5
	MRS	ERRSELR_EL1, R21                   // 355338d5
	MSR	R22, ERRSELR_EL1                   // 365318d5
	MRS	ERXADDR_EL1, R30                   // 7e5438d5
	MSR	R0, ERXADDR_EL1                    // 605418d5
	MRS	ERXCTLR_EL1, R6                    // 265438d5
	MSR	R9, ERXCTLR_EL1                    // 295418d5
	MRS	ERXFR_EL1, R19                     // 135438d5
	MRS	ERXMISC0_EL1, R20                  // 145538d5
	MSR	R24, ERXMISC0_EL1                  // 185518d5
	MRS	ERXMISC1_EL1, R15                  // 2f5538d5
	MSR	R10, ERXMISC1_EL1                  // 2a5518d5
	MRS	ERXSTATUS_EL1, R30                 // 5e5438d5
	MSR	R3, ERXSTATUS_EL1                  // 435418d5
	MRS	ESR_EL1, R6                        // 065238d5
	MSR	R21, ESR_EL1                       // 155218d5
	MRS	ESR_EL1, R17                       // 115238d5
	MSR	R12, ESR_EL1                       // 0c5218d5
	MRS	FAR_EL1, R3                        // 036038d5
	MSR	R17, FAR_EL1                       // 116018d5
	MRS	FAR_EL1, R9                        // 096038d5
	MSR	R25, FAR_EL1                       // 196018d5
	MRS	FPCR, R1                           // 01443bd5
	MSR	R27, FPCR                          // 1b441bd5
	MRS	FPSR, R5                           // 25443bd5
	MSR	R15, FPSR                          // 2f441bd5
	MRS	ID_AA64AFR0_EL1, R19               // 930538d5
	MRS	ID_AA64AFR1_EL1, R24               // b80538d5
	MRS	ID_AA64DFR0_EL1, R21               // 150538d5
	MRS	ID_AA64DFR1_EL1, R20               // 340538d5
	MRS	ID_AA64ISAR0_EL1, R4               // 040638d5
	MRS	ID_AA64ISAR1_EL1, R6               // 260638d5
	MRS	ID_AA64MMFR0_EL1, R0               // 000738d5
	MRS	ID_AA64MMFR1_EL1, R17              // 310738d5
	MRS	ID_AA64MMFR2_EL1, R23              // 570738d5
	MRS	ID_AA64PFR0_EL1, R20               // 140438d5
	MRS	ID_AA64PFR1_EL1, R26               // 3a0438d5
	MRS	ID_AA64ZFR0_EL1, R26               // 9a0438d5
	MRS	ID_AFR0_EL1, R21                   // 750138d5
	MRS	ID_DFR0_EL1, R15                   // 4f0138d5
	MRS	ID_ISAR0_EL1, R11                  // 0b0238d5
	MRS	ID_ISAR1_EL1, R16                  // 300238d5
	MRS	ID_ISAR2_EL1, R10                  // 4a0238d5
	MRS	ID_ISAR3_EL1, R13                  // 6d0238d5
	MRS	ID_ISAR4_EL1, R24                  // 980238d5
	MRS	ID_ISAR5_EL1, R29                  // bd0238d5
	MRS	ID_MMFR0_EL1, R10                  // 8a0138d5
	MRS	ID_MMFR1_EL1, R29                  // bd0138d5
	MRS	ID_MMFR2_EL1, R16                  // d00138d5
	MRS	ID_MMFR3_EL1, R10                  // ea0138d5
	MRS	ID_MMFR4_EL1, R23                  // d70238d5
	MRS	ID_PFR0_EL1, R4                    // 040138d5
	MRS	ID_PFR1_EL1, R12                   // 2c0138d5
	MRS	ISR_EL1, R24                       // 18c138d5
	MRS	MAIR_EL1, R20                      // 14a238d5
	MSR	R21, MAIR_EL1                      // 15a218d5
	MRS	MAIR_EL1, R20                      // 14a238d5
	MSR	R5, MAIR_EL1                       // 05a218d5
	MRS	MDCCINT_EL1, R23                   // 170230d5
	MSR	R27, MDCCINT_EL1                   // 1b0210d5
	MRS	MDCCSR_EL0, R19                    // 130133d5
	MRS	MDRAR_EL1, R12                     // 0c1030d5
	MRS	MDSCR_EL1, R15                     // 4f0230d5
	MSR	R15, MDSCR_EL1                     // 4f0210d5
	MRS	MIDR_EL1, R26                      // 1a0038d5
	MRS	MPIDR_EL1, R25                     // b90038d5
	MRS	MVFR0_EL1, R29                     // 1d0338d5
	MRS	MVFR1_EL1, R7                      // 270338d5
	MRS	MVFR2_EL1, R19                     // 530338d5
	MRS	NZCV, R11                          // 0b423bd5
	MSR	R10, NZCV                          // 0a421bd5
	MRS	OSDLR_EL1, R16                     // 901330d5
	MSR	R21, OSDLR_EL1                     // 951310d5
	MRS	OSDTRRX_EL1, R5                    // 450030d5
	MSR	R30, OSDTRRX_EL1                   // 5e0010d5
	MRS	OSDTRTX_EL1, R3                    // 430330d5
	MSR	R13, OSDTRTX_EL1                   // 4d0310d5
	MRS	OSECCR_EL1, R2                     // 420630d5
	MSR	R17, OSECCR_EL1                    // 510610d5
	MSR	R3, OSLAR_EL1                      // 831010d5
	MRS	OSLSR_EL1, R15                     // 8f1130d5
	MRS	PAN, R14                           // 6e4238d5
	MSR	R0, PAN                            // 604218d5
	MRS	PAR_EL1, R27                       // 1b7438d5
	MSR	R3, PAR_EL1                        // 037418d5
	MRS	PMCCFILTR_EL0, R10                 // eaef3bd5
	MSR	R16, PMCCFILTR_EL0                 // f0ef1bd5
	MRS	PMCCNTR_EL0, R17                   // 119d3bd5
	MSR	R13, PMCCNTR_EL0                   // 0d9d1bd5
	MRS	PMCEID0_EL0, R8                    // c89c3bd5
	MRS	PMCEID1_EL0, R30                   // fe9c3bd5
	MRS	PMCNTENCLR_EL0, R11                // 4b9c3bd5
	MSR	R21, PMCNTENCLR_EL0                // 559c1bd5
	MRS	PMCNTENSET_EL0, R25                // 399c3bd5
	MSR	R13, PMCNTENSET_EL0                // 2d9c1bd5
	MRS	PMCR_EL0, R23                      // 179c3bd5
	MSR	R11, PMCR_EL0                      // 0b9c1bd5
	MRS	PMEVCNTR0_EL0, R27                 // 1be83bd5
	MRS	PMEVCNTR1_EL0, R23                 // 37e83bd5
	MRS	PMEVCNTR2_EL0, R26                 // 5ae83bd5
	MRS	PMEVCNTR3_EL0, R11                 // 6be83bd5
	MRS	PMEVCNTR4_EL0, R14                 // 8ee83bd5
	MRS	PMEVCNTR5_EL0, R9                  // a9e83bd5
	MRS	PMEVCNTR6_EL0, R30                 // dee83bd5
	MRS	PMEVCNTR7_EL0, R19                 // f3e83bd5
	MRS	PMEVCNTR8_EL0, R5                  // 05e93bd5
	MRS	PMEVCNTR9_EL0, R27                 // 3be93bd5
	MRS	PMEVCNTR10_EL0, R23                // 57e93bd5
	MRS	PMEVCNTR11_EL0, R27                // 7be93bd5
	MRS	PMEVCNTR12_EL0, R0                 // 80e93bd5
	MRS	PMEVCNTR13_EL0, R13                // ade93bd5
	MRS	PMEVCNTR14_EL0, R27                // dbe93bd5
	MRS	PMEVCNTR15_EL0, R16                // f0e93bd5
	MRS	PMEVCNTR16_EL0, R16                // 10ea3bd5
	MRS	PMEVCNTR17_EL0, R14                // 2eea3bd5
	MRS	PMEVCNTR18_EL0, R10                // 4aea3bd5
	MRS	PMEVCNTR19_EL0, R12                // 6cea3bd5
	MRS	PMEVCNTR20_EL0, R5                 // 85ea3bd5
	MRS	PMEVCNTR21_EL0, R26                // baea3bd5
	MRS	PMEVCNTR22_EL0, R19                // d3ea3bd5
	MRS	PMEVCNTR23_EL0, R5                 // e5ea3bd5
	MRS	PMEVCNTR24_EL0, R17                // 11eb3bd5
	MRS	PMEVCNTR25_EL0, R0                 // 20eb3bd5
	MRS	PMEVCNTR26_EL0, R20                // 54eb3bd5
	MRS	PMEVCNTR27_EL0, R12                // 6ceb3bd5
	MRS	PMEVCNTR28_EL0, R29                // 9deb3bd5
	MRS	PMEVCNTR29_EL0, R22                // b6eb3bd5
	MRS	PMEVCNTR30_EL0, R22                // d6eb3bd5
	MSR	R30, PMEVCNTR0_EL0                 // 1ee81bd5
	MSR	R1, PMEVCNTR1_EL0                  // 21e81bd5
	MSR	R20, PMEVCNTR2_EL0                 // 54e81bd5
	MSR	R9, PMEVCNTR3_EL0                  // 69e81bd5
	MSR	R8, PMEVCNTR4_EL0                  // 88e81bd5
	MSR	R2, PMEVCNTR5_EL0                  // a2e81bd5
	MSR	R30, PMEVCNTR6_EL0                 // dee81bd5
	MSR	R14, PMEVCNTR7_EL0                 // eee81bd5
	MSR	R1, PMEVCNTR8_EL0                  // 01e91bd5
	MSR	R15, PMEVCNTR9_EL0                 // 2fe91bd5
	MSR	R15, PMEVCNTR10_EL0                // 4fe91bd5
	MSR	R14, PMEVCNTR11_EL0                // 6ee91bd5
	MSR	R15, PMEVCNTR12_EL0                // 8fe91bd5
	MSR	R25, PMEVCNTR13_EL0                // b9e91bd5
	MSR	R26, PMEVCNTR14_EL0                // dae91bd5
	MSR	R21, PMEVCNTR15_EL0                // f5e91bd5
	MSR	R29, PMEVCNTR16_EL0                // 1dea1bd5
	MSR	R11, PMEVCNTR17_EL0                // 2bea1bd5
	MSR	R16, PMEVCNTR18_EL0                // 50ea1bd5
	MSR	R2, PMEVCNTR19_EL0                 // 62ea1bd5
	MSR	R19, PMEVCNTR20_EL0                // 93ea1bd5
	MSR	R17, PMEVCNTR21_EL0                // b1ea1bd5
	MSR	R7, PMEVCNTR22_EL0                 // c7ea1bd5
	MSR	R23, PMEVCNTR23_EL0                // f7ea1bd5
	MSR	R15, PMEVCNTR24_EL0                // 0feb1bd5
	MSR	R27, PMEVCNTR25_EL0                // 3beb1bd5
	MSR	R13, PMEVCNTR26_EL0                // 4deb1bd5
	MSR	R2, PMEVCNTR27_EL0                 // 62eb1bd5
	MSR	R15, PMEVCNTR28_EL0                // 8feb1bd5
	MSR	R14, PMEVCNTR29_EL0                // aeeb1bd5
	MSR	R23, PMEVCNTR30_EL0                // d7eb1bd5
	MRS	PMEVTYPER0_EL0, R23                // 17ec3bd5
	MRS	PMEVTYPER1_EL0, R30                // 3eec3bd5
	MRS	PMEVTYPER2_EL0, R12                // 4cec3bd5
	MRS	PMEVTYPER3_EL0, R13                // 6dec3bd5
	MRS	PMEVTYPER4_EL0, R25                // 99ec3bd5
	MRS	PMEVTYPER5_EL0, R23                // b7ec3bd5
	MRS	PMEVTYPER6_EL0, R8                 // c8ec3bd5
	MRS	PMEVTYPER7_EL0, R2                 // e2ec3bd5
	MRS	PMEVTYPER8_EL0, R23                // 17ed3bd5
	MRS	PMEVTYPER9_EL0, R25                // 39ed3bd5
	MRS	PMEVTYPER10_EL0, R0                // 40ed3bd5
	MRS	PMEVTYPER11_EL0, R30               // 7eed3bd5
	MRS	PMEVTYPER12_EL0, R0                // 80ed3bd5
	MRS	PMEVTYPER13_EL0, R9                // a9ed3bd5
	MRS	PMEVTYPER14_EL0, R15               // cfed3bd5
	MRS	PMEVTYPER15_EL0, R13               // eded3bd5
	MRS	PMEVTYPER16_EL0, R11               // 0bee3bd5
	MRS	PMEVTYPER17_EL0, R19               // 33ee3bd5
	MRS	PMEVTYPER18_EL0, R3                // 43ee3bd5
	MRS	PMEVTYPER19_EL0, R17               // 71ee3bd5
	MRS	PMEVTYPER20_EL0, R8                // 88ee3bd5
	MRS	PMEVTYPER21_EL0, R2                // a2ee3bd5
	MRS	PMEVTYPER22_EL0, R5                // c5ee3bd5
	MRS	PMEVTYPER23_EL0, R17               // f1ee3bd5
	MRS	PMEVTYPER24_EL0, R22               // 16ef3bd5
	MRS	PMEVTYPER25_EL0, R3                // 23ef3bd5
	MRS	PMEVTYPER26_EL0, R23               // 57ef3bd5
	MRS	PMEVTYPER27_EL0, R19               // 73ef3bd5
	MRS	PMEVTYPER28_EL0, R24               // 98ef3bd5
	MRS	PMEVTYPER29_EL0, R3                // a3ef3bd5
	MRS	PMEVTYPER30_EL0, R1                // c1ef3bd5
	MSR	R20, PMEVTYPER0_EL0                // 14ec1bd5
	MSR	R20, PMEVTYPER1_EL0                // 34ec1bd5
	MSR	R14, PMEVTYPER2_EL0                // 4eec1bd5
	MSR	R26, PMEVTYPER3_EL0                // 7aec1bd5
	MSR	R11, PMEVTYPER4_EL0                // 8bec1bd5
	MSR	R16, PMEVTYPER5_EL0                // b0ec1bd5
	MSR	R29, PMEVTYPER6_EL0                // ddec1bd5
	MSR	R3, PMEVTYPER7_EL0                 // e3ec1bd5
	MSR	R30, PMEVTYPER8_EL0                // 1eed1bd5
	MSR	R17, PMEVTYPER9_EL0                // 31ed1bd5
	MSR	R10, PMEVTYPER10_EL0               // 4aed1bd5
	MSR	R19, PMEVTYPER11_EL0               // 73ed1bd5
	MSR	R13, PMEVTYPER12_EL0               // 8ded1bd5
	MSR	R23, PMEVTYPER13_EL0               // b7ed1bd5
	MSR	R13, PMEVTYPER14_EL0               // cded1bd5
	MSR	R9, PMEVTYPER15_EL0                // e9ed1bd5
	MSR	R1, PMEVTYPER16_EL0                // 01ee1bd5
	MSR	R19, PMEVTYPER17_EL0               // 33ee1bd5
	MSR	R22, PMEVTYPER18_EL0               // 56ee1bd5
	MSR	R23, PMEVTYPER19_EL0               // 77ee1bd5
	MSR	R30, PMEVTYPER20_EL0               // 9eee1bd5
	MSR	R9, PMEVTYPER21_EL0                // a9ee1bd5
	MSR	R3, PMEVTYPER22_EL0                // c3ee1bd5
	MSR	R1, PMEVTYPER23_EL0                // e1ee1bd5
	MSR	R16, PMEVTYPER24_EL0               // 10ef1bd5
	MSR	R12, PMEVTYPER25_EL0               // 2cef1bd5
	MSR	R7, PMEVTYPER26_EL0                // 47ef1bd5
	MSR	R9, PMEVTYPER27_EL0                // 69ef1bd5
	MSR	R10, PMEVTYPER28_EL0               // 8aef1bd5
	MSR	R5, PMEVTYPER29_EL0                // a5ef1bd5
	MSR	R12, PMEVTYPER30_EL0               // ccef1bd5
	MRS	PMINTENCLR_EL1, R24                // 589e38d5
	MSR	R15, PMINTENCLR_EL1                // 4f9e18d5
	MRS	PMINTENSET_EL1, R1                 // 219e38d5
	MSR	R4, PMINTENSET_EL1                 // 249e18d5
	MRS	PMOVSCLR_EL0, R6                   // 669c3bd5
	MSR	R30, PMOVSCLR_EL0                  // 7e9c1bd5
	MRS	PMOVSSET_EL0, R16                  // 709e3bd5
	MSR	R12, PMOVSSET_EL0                  // 6c9e1bd5
	MRS	PMSELR_EL0, R30                    // be9c3bd5
	MSR	R5, PMSELR_EL0                     // a59c1bd5
	MSR	R27, PMSWINC_EL0                   // 9b9c1bd5
	MRS	PMUSERENR_EL0, R8                  // 089e3bd5
	MSR	R6, PMUSERENR_EL0                  // 069e1bd5
	MRS	PMXEVCNTR_EL0, R26                 // 5a9d3bd5
	MSR	R10, PMXEVCNTR_EL0                 // 4a9d1bd5
	MRS	PMXEVTYPER_EL0, R4                 // 249d3bd5
	MSR	R4, PMXEVTYPER_EL0                 // 249d1bd5
	MRS	REVIDR_EL1, R29                    // dd0038d5
	MRS	RMR_EL1, R4                        // 44c038d5
	MSR	R0, RMR_EL1                        // 40c018d5
	MRS	RVBAR_EL1, R7                      // 27c038d5
	MRS	SCTLR_EL1, R8                      // 081038d5
	MSR	R0, SCTLR_EL1                      // 001018d5
	MRS	SCTLR_EL1, R30                     // 1e1038d5
	MSR	R13, SCTLR_EL1                     // 0d1018d5
	MRS	SPSR_EL1, R1                       // 014038d5
	MSR	R2, SPSR_EL1                       // 024018d5
	MRS	SPSR_EL1, R3                       // 034038d5
	MSR	R14, SPSR_EL1                      // 0e4018d5
	MRS	SPSR_abt, R12                      // 2c433cd5
	MSR	R4, SPSR_abt                       // 24431cd5
	MRS	SPSR_fiq, R17                      // 71433cd5
	MSR	R9, SPSR_fiq                       // 69431cd5
	MRS	SPSR_irq, R12                      // 0c433cd5
	MSR	R23, SPSR_irq                      // 17431cd5
	MRS	SPSR_und, R29                      // 5d433cd5
	MSR	R3, SPSR_und                       // 43431cd5
	MRS	SPSel, R29                         // 1d4238d5
	MSR	R1, SPSel                          // 014218d5
	MRS	SP_EL0, R10                        // 0a4138d5
	MSR	R4, SP_EL0                         // 044118d5
	MRS	SP_EL1, R22                        // 16413cd5
	MSR	R17, SP_EL1                        // 11411cd5
	MRS	TCR_EL1, R17                       // 512038d5
	MSR	R23, TCR_EL1                       // 572018d5
	MRS	TCR_EL1, R14                       // 4e2038d5
	MSR	R29, TCR_EL1                       // 5d2018d5
	MRS	TPIDRRO_EL0, R26                   // 7ad03bd5
	MSR	R16, TPIDRRO_EL0                   // 70d01bd5
	MRS	TPIDR_EL0, R23                     // 57d03bd5
	MSR	R5, TPIDR_EL0                      // 45d01bd5
	MRS	TPIDR_EL1, R17                     // 91d038d5
	MSR	R22, TPIDR_EL1                     // 96d018d5
	MRS	TTBR0_EL1, R30                     // 1e2038d5
	MSR	R29, TTBR0_EL1                     // 1d2018d5
	MRS	TTBR0_EL1, R23                     // 172038d5
	MSR	R15, TTBR0_EL1                     // 0f2018d5
	MRS	TTBR1_EL1, R5                      // 252038d5
	MSR	R26, TTBR1_EL1                     // 3a2018d5
	MRS	TTBR1_EL1, R19                     // 332038d5
	MSR	R23, TTBR1_EL1                     // 372018d5
	MRS	UAO, R22                           // 964238d5
	MSR	R4, UAO                            // 844218d5
	MRS	VBAR_EL1, R23                      // 17c038d5
	MSR	R2, VBAR_EL1                       // 02c018d5
	MRS	VBAR_EL1, R6                       // 06c038d5
	MSR	R3, VBAR_EL1                       // 03c018d5
	MRS	DISR_EL1, R12                      // 2cc138d5
	MSR	R24, DISR_EL1                      // 38c118d5
	MRS	MPIDR_EL1, R1                      // a10038d5
	MRS	MIDR_EL1, R13                      // 0d0038d5
	MRS	ZCR_EL1, R24                       // 181238d5
	MSR	R13, ZCR_EL1                       // 0d1218d5
	MRS	ZCR_EL1, R23                       // 171238d5
	MSR	R17, ZCR_EL1                       // 111218d5

// END
//
//	LTYPEE comma
//	{
//		outcode($1, &nullgen, NREG, &nullgen);
//	}
	END
