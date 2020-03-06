// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This input was created by taking the ppc64 testcase and modified
// by hand.

#include "../../../../../runtime/textflag.h"

TEXT foo(SB),DUPOK|NOSPLIT,$0
//
// branch
//
//	LBRA rel
//	{
//		outcode(int($1), &nullgen, 0, &$2);
//	}
	BEQ	R1, 2(PC)
label0:
	JMP	1(PC)		// JMP 1(PC)	// 10000001
	BEQ	R1, 2(PC)
	JMP	label0+0	// JMP 3	// 1000fffd
	BEQ	R1, 2(PC)
	JAL	1(PC)		// CALL 1(PC)	// 0c00000e
	BEQ	R1, 2(PC)
	JAL	label0+0	// CALL 3	// 0c000006

//	LBRA addr
//	{
//		outcode(int($1), &nullgen, 0, &$2);
//	}
	BEQ	R1, 2(PC)
	JMP	0(R1)		// JMP (R1)	// 00200008
	BEQ	R1, 2(PC)
	JMP	foo+0(SB)	// JMP foo(SB)	// 08000018
	BEQ	R1, 2(PC)
	JAL	0(R1)		// CALL (R1)	// 0020f809
	BEQ	R1, 2(PC)
	JAL	foo+0(SB)	// CALL foo(SB)	// 0c000020

//
// BEQ/BNE
//
//	LBRA rreg ',' rel
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
label1:
	BEQ	R1, 1(PC)	// BEQ R1, 1(PC)	// 10200001
	BEQ	R1, label1	// BEQ R1, 18		// 1020fffd

//	LBRA rreg ',' sreg ',' rel
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
label2:
	BEQ	R1, R2, 1(PC)	// BEQ R1, R2, 1(PC)	// 10220001
	BEQ	R1, R2, label2	// BEQ R1, R2, 20	// 1022fffd

//
// other integer conditional branch
//
//	LBRA rreg ',' rel
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
label3:
	BLTZ	R1, 1(PC)	// BLTZ R1, 1(PC)	// 04200001
	BLTZ	R1, label3	// BLTZ R1, 22		// 0420fffd

//
// floating point conditional branch
//
//	LBRA rel
label4:
	BFPT	1(PC)	// BFPT 1(PC)			// 4501000100000000
	BFPT	label4	// BFPT 24			// 4501fffd00000000

//inst:
//
// load ints and bytes
//
//	LMOVV rreg ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVV	R25, R17	// 00198825
	MOVV	R1, R2		// 00011025
	MOVV	LO, R1		// 00000812
	MOVV	HI, R1		// 00000810
	MOVV	R1, LO		// 00200013
	MOVV	R1, HI		// 00200011


//	LMOVW rreg ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVW	R1, R2		// 00011004
	MOVW	LO, R1		// 00000812
	MOVW	HI, R1		// 00000810
	MOVW	R1, LO		// 00200013
	MOVW	R1, HI		// 00200011
	MOVWU	R14, R27	// 000ed83c001bd83e

//	LMOVH rreg ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVH	R16, R27	// 0010dc00001bdc03
	MOVHU	R1, R3		// 3023ffff

//	LMOVB rreg ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVB	R8, R9		// 00084e0000094e03
	MOVBU	R12, R17	// 319100ff

//	LMOVV addr ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVV	foo<>+3(SB), R2
	MOVV	(R5), R18	// dcb20000
	MOVV	8(R16), R4	// de040008
	MOVV	-32(R14), R1	// ddc1ffe0
	LLV	(R1), R2	// d0220000

//	LMOVW addr ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVW	foo<>+3(SB), R2
	MOVW	(R11), R22	// 8d760000
	MOVW	1(R9), R24	// 8d380001
	MOVW	-17(R24), R8	// 8f08ffef
	MOVWU	(R11), R22	// 9d760000
	MOVWU	1(R9), R24	// 9d380001
	MOVWU	-17(R24), R8	// 9f08ffef
	LL	(R1), R2	// c0220000

//	LMOVH addr ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVH	foo<>+3(SB), R2
	MOVH	(R20), R7	// 86870000
	MOVH	54(R11), R26	// 857a0036
	MOVH	-42(R3), R20	// 8474ffd6
	MOVHU	(R20), R7	// 96870000
	MOVHU	54(R11), R26	// 957a0036
	MOVHU	-42(R3), R20	// 9474ffd6

//	LMOVB addr ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVB	foo<>+3(SB), R2
	MOVB	(R4), R21	// 80950000
	MOVB	9(R19), R18	// 82720009
	MOVB	-10(R19), R18	// 8272fff6
	MOVBU	(R4), R21	// 90950000
	MOVBU	9(R19), R18	// 92720009
	MOVBU	-10(R19), R18	// 9272fff6

//
// load floats
//
//	LFMOV addr ',' freg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVD	foo<>+3(SB), F2
	MOVD	16(R1), F2
	MOVD	(R1), F2

//	LFMOV fimm ',' freg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVD	$0.1, F2 // MOVD $(0.10000000000000001), F2

//	LFMOV freg ',' freg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVD	F1, F2

//	LFMOV freg ',' addr
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVD	F2, foo<>+3(SB)
	MOVD	F2, 16(R1)
	MOVD	F2, (R1)

//
// store ints and bytes
//
//	LMOVV rreg ',' addr
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVV	R1, foo<>+3(SB)
	MOVV	R18, (R5)	// fcb20000
	MOVV	R4, 8(R16)	// fe040008
	MOVV	R1, -32(R14)	// fdc1ffe0
	SCV	R1, (R2)	// f0410000

//	LMOVW rreg ',' addr
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVW	R1, foo<>+3(SB)
	MOVW	R8, (R3)	// ac680000
	MOVW	R11, 19(R2)	// ac4b0013
	MOVW	R25, -89(R22)	// aed9ffa7
	MOVWU	R8, (R3)	// ac680000
	MOVWU	R11, 19(R2)	// ac4b0013
	MOVWU	R25, -89(R22)	// aed9ffa7
	SC	R1, (R2)	// e0410000

//	LMOVH rreg ',' addr
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVH	R13, (R7)	// a4ed0000
	MOVH	R10, 61(R23)	// a6ea003d
	MOVH	R8, -33(R12)	// a588ffdf
	MOVHU	R13, (R7)	// a4ed0000
	MOVHU	R10, 61(R23)	// a6ea003d
	MOVHU	R8, -33(R12)	// a588ffdf

//	LMOVB rreg ',' addr
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVB	R1, foo<>+3(SB)
	MOVB	R5, -18(R4)	// a085ffee
	MOVB	R10, 9(R13)	// a1aa0009
	MOVB	R15, (R13)	// a1af0000
	MOVBU	R5, -18(R4)	// a085ffee
	MOVBU	R10, 9(R13)	// a1aa0009
	MOVBU	R15, (R13)	// a1af0000

//
// store floats
//
//	LMOVW freg ',' addr
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVD	F1, foo<>+3(SB)
	MOVD	F1, 16(R2)
	MOVD	F1, (R2)

//
// floating point status
//
//	LMOVW fpscr ',' freg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVW	FCR31, R1 // 4441f800

//	LMOVW freg ','  fpscr
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVW	R1, FCR31 // 44c1f800

//	LMOVW rreg ',' mreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVW	R1, M1 // 40810800
	MOVV	R1, M1 // 40a10800

//	LMOVW mreg ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVW	M1, R1 // 40010800
	MOVV	M1, R1 // 40210800


//
// integer operations
// logical instructions
// shift instructions
// unary instructions
//
//	LADDW rreg ',' sreg ',' rreg
//	{
//		outcode(int($1), &$2, int($4), &$6);
//	}
	ADD	R5, R9, R10	// 01255020
	ADDU	R13, R14, R19	// 01cd9821
	ADDV	R5, R9, R10	// 0125502c
	ADDVU	R13, R14, R19	// 01cd982d

//	LADDW imm ',' sreg ',' rreg
//	{
//		outcode(int($1), &$2, int($4), &$6);
//	}
	ADD	$15176, R14, R9	// 21c93b48
	ADD	$-9, R5, R8	// 20a8fff7
	ADDU	$10, R9, R9	// 2529000a
	ADDV	$15176, R14, R9	// 61c93b48
	ADDV	$-9, R5, R8	// 60a8fff7
	ADDVU	$10, R9, R9	// 6529000a

//	LADDW rreg ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	ADD	R1, R2		// 00411020
	ADDU	R1, R2		// 00411021
	ADDV	R1, R2		// 0041102c
	ADDVU	R1, R2		// 0041102d

//	LADDW imm ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	ADD	$4, R1		// 20210004
	ADDV	$4, R1		// 60210004
	ADDU	$4, R1		// 24210004
	ADDVU	$4, R1		// 64210004
	ADD	$-7193, R24	// 2318e3e7
	ADDV	$-7193, R24	// 6318e3e7

//	LSUBW rreg ',' sreg ',' rreg
//	{
//		outcode(int($1), &$2, int($4), &$6);
//	}
	SUB	R6, R26, R27	// 0346d822
	SUBU	R6, R26, R27	// 0346d823
	SUBV	R16, R17, R26	// 0230d02e
	SUBVU	R16, R17, R26	// 0230d02f

//	LSUBW imm ',' sreg ',' rreg
//	{
//		outcode(int($1), &$2, int($4), &$6);
//	}
	SUB	$-3126, R17, R22	// 22360c36
	SUB	$3126, R17, R22		// 2236f3ca
	SUBU	$16384, R17, R12	// 262cc000
	SUBV	$-6122, R10, R9		// 614917ea
	SUBV	$6122, R10, R9		// 6149e816
	SUBVU	$1203, R17, R12		// 662cfb4d

//	LSUBW rreg ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	SUB	R14, R13	// 01ae6822
	SUBU	R14, R13	// 01ae6823
	SUBV	R4, R3		// 0064182e
	SUBVU	R4, R3		// 0064182f
//	LSUBW imm ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	SUB	$6512, R13	// 21ade690
	SUB	$-6512, R13	// 21ad1970
	SUBU	$6512, R13	// 25ade690
	SUBV	$9531, R16	// 6210dac5
	SUBV	$-9531, R13	// 61ad253b
	SUBVU	$9531, R16	// 6610dac5

//	LMUL rreg ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MUL	R19, R8		// 01130018
	MULU	R21, R13	// 01b50019
	MULV	R19, R8		// 0113001c
	MULVU	R21, R13	// 01b5001d

//	LDIV rreg ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	DIV	R18, R22	// 02d2001a
	DIVU	R14, R9		// 012e001b
	DIVV	R8, R13		// 01a8001e
	DIVVU	R16, R19	// 0270001f

//	LREM rreg ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	REM	R18, R22	// 02d2001a
	REMU	R14, R9		// 012e001b
	REMV	R8, R13		// 01a8001e
	REMVU	R16, R19	// 0270001f

//	LSHW rreg ',' sreg ',' rreg
//	{
//		outcode(int($1), &$2, int($4), &$6);
//	}
	SLL	R1, R2, R3	// 00221804
	SLLV	R10, R22, R21	// 0156a814
	SRL	R27, R6, R17	// 03668806
	SRLV	R27, R6, R17	// 03668816
	SRA	R11, R19, R20	// 0173a007
	SRAV	R20, R19, R19	// 02939817

//	LSHW rreg ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	SLL	R1, R2		// 00221004
	SLLV	R10, R22	// 0156b014
	SRL	R27, R6   	// 03663006
	SRLV	R27, R6   	// 03663016
	SRA	R11, R19	// 01739807
	SRAV	R20, R19	// 02939817

//	LSHW imm ',' sreg ',' rreg
//	{
//		outcode(int($1), &$2, int($4), &$6);
//	}
	SLL	$19, R22, R21	// 0016acc0
	SLLV	$19, R22, R21	// 0016acf8
	SRL	$31, R6, R17	// 00068fc2
	SRLV	$31, R6, R17	// 00068ffa
	SRA	$8, R8, R19	// 00089a03
	SRAV	$19, R8, R7	// 00083cfb

//	LSHW imm ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	SLL	$19, R21	// 0015acc0
	SLLV	$19, R21	// 0015acf8
	SRL	$31, R17	// 00118fc2
	SRLV	$31, R17	// 00118ffa
	SRA	$3, R12		// 000c60c3
	SRAV	$12, R3		// 00031b3b


//	LAND/LXOR/LNOR/LOR rreg ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	AND	R14, R8		// 010e4024
	XOR	R15, R9		// 012f4826
	NOR	R16, R10	// 01505027
	OR	R17, R11	// 01715825

//	LAND/LXOR/LOR imm ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	AND	$11, R17, R7	// 3227000b
	XOR	$341, R1, R23	// 38370155
	OR	$254, R25, R13	// 372d00fe
//
// move immediate: macro for lui+or, addi, addis, and other combinations
//
//	LMOVW imm ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVW	$1, R1
	MOVV	$1, R1

//	LMOVW ximm ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVW	$1, R1
	MOVW	$foo(SB), R1
	MOVV	$1, R1
	MOVV	$foo(SB), R1

//
// floating point operate
//
//	LFCONV freg ',' freg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	ABSD	F1, F2

//	LFADD freg ',' freg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	ADDD	F1, F2

//	LFADD freg ',' freg ',' freg
//	{
//		outcode(int($1), &$2, int($4.Reg), &$6);
//	}
	ADDD	F1, F2, F3

//	LFCMP freg ',' freg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	CMPEQD	F1, F2


//
// WORD
//
	WORD	$1	// 00000001
	NOOP		// 00000000
	SYNC		// 0000000f

//
// NOP
//
//	LNOP comma // asm doesn't support the trailing comma.
//	{
//		outcode(int($1), &nullgen, 0, &nullgen);
//	}
	NOP

//	LNOP rreg comma // asm doesn't support the trailing comma.
//	{
//		outcode(int($1), &$2, 0, &nullgen);
//	}
	NOP R2

//	LNOP freg comma // asm doesn't support the trailing comma.
//	{
//		outcode(int($1), &$2, 0, &nullgen);
//	}
	NOP	F2

//	LNOP ',' rreg // asm doesn't support the leading comma.
//	{
//		outcode(int($1), &nullgen, 0, &$3);
//	}
	NOP	R2

//	LNOP ',' freg // asm doesn't support the leading comma.
//	{
//		outcode(int($1), &nullgen, 0, &$3);
//	}
	NOP	F2

//	LNOP imm
//	{
//		outcode(int($1), &$2, 0, &nullgen);
//	}
	NOP	$4

//
// special
//
	SYSCALL
	BREAK
	// overloaded cache opcode:
	BREAK	R1, (R1)

//
// RET
//
//	LRETRN	comma // asm doesn't support the trailing comma.
//	{
//		outcode(int($1), &nullgen, 0, &nullgen);
//	}
	SYSCALL
	BEQ	R1, 2(PC)
	RET


// More JMP/JAL cases, and canonical names JMP, CALL.

	JAL	foo(SB) // CALL foo(SB)
	BEQ	R1, 2(PC)
	JMP	foo(SB)
	CALL	foo(SB)
	RET	foo(SB)

	NEGW	R1, R2 // 00011023
	NEGV	R1, R2 // 0001102f
	RET

// MSA VMOVI
	VMOVB	$511, W0   // 7b0ff807
	VMOVH	$24, W23   // 7b20c5c7
	VMOVW	$-24, W15  // 7b5f43c7
	VMOVD	$-511, W31 // 7b700fc7

	VMOVB	(R0), W8       // 78000220
	VMOVB	511(R3), W0    // 79ff1820
	VMOVB	-512(R12), W21 // 7a006560
	VMOVH	(R24), W12     // 7800c321
	VMOVH	110(R19), W8   // 78379a21
	VMOVH	-70(R12), W3   // 7bdd60e1
	VMOVW	(R3), W31      // 78001fe2
	VMOVW	64(R20), W16   // 7810a422
	VMOVW	-104(R17), W24 // 7be68e22
	VMOVD	(R3), W2       // 780018a3
	VMOVD	128(R23), W19  // 7810bce3
	VMOVD	-256(R31), W0  // 7be0f823

	VMOVB	W8, (R0)       // 78000224
	VMOVB	W0, 511(R3)    // 79ff1824
	VMOVB	W21, -512(R12) // 7a006564
	VMOVH	W12, (R24)     // 7800c325
	VMOVH	W8, 110(R19)   // 78379a25
	VMOVH	W3, -70(R12)   // 7bdd60e5
	VMOVW	W31, (R3)      // 78001fe6
	VMOVW	W16, 64(R20)   // 7810a426
	VMOVW	W24, -104(R17) // 7be68e26
	VMOVD	W2, (R3)       // 780018a7
	VMOVD	W19, 128(R23)  // 7810bce7
	VMOVD	W0, -256(R31)  // 7be0f827
	RET

// END
//
//	LEND	comma // asm doesn't support the trailing comma.
//	{
//		outcode(int($1), &nullgen, 0, &nullgen);
//	}
	END
