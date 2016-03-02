// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This input was created by taking the instruction productions in
// the old assembler's (7a's) grammar and hand-writing complete
// instructions for each rule, to guarantee we cover the same space.

TEXT	foo(SB), 7, $-8

//
// ADD
//
//	LTYPE1 imsr ',' spreg ',' reg
//	{
//		outcode($1, &$2, $4, &$6);
//	}
// imsr comes from the old 7a, we only support immediates and registers
// at the moment, no shifted registers.
	ADDW	$1, R2, R3
	ADDW	R1, R2, R3
	ADDW	R1, ZR, R3
	ADD	$1, R2, R3
	ADD	R1, R2, R3
	ADD	R1, ZR, R3
	ADD	$1, R2, R3

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

//
// MOVK
//
//		LMOVK imm ',' reg
//	{
//		outcode($1, &$2, NREG, &$4);
//	}
	MOVK	$1, R1

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
	CSET	GT, R1
//
// CSEL/CSINC/CSNEG/CSINV
//
//		LTYPES cond ',' reg ',' reg ',' reg
//	{
//		outgcode($1, &$2, $6.reg, &$4, &$8);
//	}
	CSEL	LT, R1, R2, ZR
	CSINC	GT, R1, ZR, R3
	CSNEG	MI, R1, R2, R3
	CSINV	CS, R1, R2, R3 // CSINV HS, R1, R2, R3

//		LTYPES cond ',' reg ',' reg
//	{
//		outcode($1, &$2, $4.reg, &$6);
//	}
	CSEL	LT, R1, R2
//
// CCMN
//
//		LTYPEU cond ',' imsr ',' reg ',' imm comma
//	{
//		outgcode($1, &$2, $6.reg, &$4, &$8);
//	}
	CCMN	MI, ZR, R1, $4

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
//	FCCMP	LT, F1, F2, $1

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
	JMP	foo(SB)
	CALL	foo(SB)

// END
//
//	LTYPEE comma
//	{
//		outcode($1, &nullgen, NREG, &nullgen);
//	}
	END
