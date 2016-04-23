// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This input was created by taking the instruction productions in
// the old assembler's (5a's) grammar and hand-writing complete
// instructions for each rule, to guarantee we cover the same space.

TEXT	foo(SB), 7, $0

// ADD
//
//	LTYPE1 cond imsr ',' spreg ',' reg
//	{
//		outcode($1, $2, &$3, $5, &$7);
//	}
// Cover some operand space here too.
	ADD	$1, R2, R3
	ADD	R1<<R2, R3, R4
	ADD	R1>>R2, R3, R4
	ADD	R1@>R2, R3, R4
	ADD	R1->R2, R3, R4
	ADD	R1, R2, R3
	ADD	R(1)<<R(2), R(3), R(4) // ADD	R1<<R2, R3, R4

//	LTYPE1 cond imsr ',' spreg ',' // asm doesn't support trailing comma.
//	{
//		outcode($1, $2, &$3, $5, &nullgen);
//	}
//	LTYPE1 cond imsr ',' reg
//	{
//		outcode($1, $2, &$3, 0, &$5);
//	}
	ADD	$1, R2
	ADD	R1<<R2, R3
	ADD	R1>>R2, R3
	ADD	R1@>R2, R3
	ADD	R1->R2, R3
	ADD	R1, R2

//
// MVN
//
//	LTYPE2 cond imsr ',' reg
//	{
//		outcode($1, $2, &$3, 0, &$5);
//	}
	CLZ.S	R1, R2

//
// MOVW
//
//	LTYPE3 cond gen ',' gen
//	{
//		outcode($1, $2, &$3, 0, &$5);
//	}
	MOVW.S	R1, R2
	MOVW.S	$1, R2
	MOVW.S	R1<<R2, R3

//
// B/BL
//
//	LTYPE4 cond comma rel
//	{
//		outcode($1, $2, &nullgen, 0, &$4);
//	}
	B.EQ	1(PC) // BEQ 1(PC)

//	LTYPE4 cond comma nireg
//	{
//		outcode($1, $2, &nullgen, 0, &$4);
//	}
	BEQ	2(PC)
	B	foo(SB) // JMP foo(SB)
	BEQ	2(PC)
	B	bar<>(SB) // JMP bar<>(SB)

//
// BX
//
//	LTYPEBX comma ireg
//	{
//		outcode($1, Always, &nullgen, 0, &$3);
//	}
	BX	(R0)

//
// BEQ
//
//	LTYPE5 comma rel
//	{
//		outcode($1, Always, &nullgen, 0, &$3);
//	}
	BEQ	1(PC)

//
// SWI
//
//	LTYPE6 cond comma gen
//	{
//		outcode($1, $2, &nullgen, 0, &$4);
//	}
	SWI.S	$2
	SWI.S	(R1)
//	SWI.S	foo(SB) - TODO: classifying foo(SB) as C_TLS_LE

//
// CMP
//
//	LTYPE7 cond imsr ',' spreg
//	{
//		outcode($1, $2, &$3, $5, &nullgen);
//	}
	CMP.S	$1, R2
	CMP.S	R1<<R2, R3
	CMP.S	R1, R2

//
// MOVM
//
//	LTYPE8 cond ioreg ',' '[' reglist ']'
//	{
//		var g obj.Addr
//
//		g = nullgen;
//		g.Type = obj.TYPE_CONST;
//		g.Offset = int64($6);
//		outcode($1, $2, &$3, 0, &g);
//	}
	MOVM	0(R1), [R2,R5,R8,g] // MOVM	(R1), [R2,R5,R8,g]
	MOVM	(R1), [R2-R5] // MOVM (R1), [R2,R3,R4,R5]
	MOVM.S	(R1), [R2]

//	LTYPE8 cond '[' reglist ']' ',' ioreg
//	{
//		var g obj.Addr
//
//		g = nullgen;
//		g.Type = obj.TYPE_CONST;
//		g.Offset = int64($4);
//		outcode($1, $2, &g, 0, &$7);
//	}
	MOVM	[R2,R5,R8,g], 0(R1) // MOVM	[R2,R5,R8,g], (R1)
	MOVM	[R2-R5], (R1) // MOVM [R2,R3,R4,R5], (R1)
	MOVM.S	[R2], (R1)

//
// SWAP
//
//	LTYPE9 cond reg ',' ireg ',' reg
//	{
//		outcode($1, $2, &$5, int32($3.Reg), &$7);
//	}
	STREX.S	R1, (R2), R3 // STREX.S (R2), R1, R3

//	LTYPE9 cond reg ',' ireg
//	{
//		outcode($1, $2, &$5, int32($3.Reg), &$3);
//	}
	STREX.S	R1, (R2) // STREX.S (R2), R1, R1

//	LTYPE9 cond comma ireg ',' reg
//	{
//		outcode($1, $2, &$4, int32($6.Reg), &$6);
//	}
	STREX.S	(R2), R3 // STREX.S (R2), R3, R3

//
// word
//
//	LTYPEH comma ximm
//	{
//		outcode($1, Always, &nullgen, 0, &$3);
//	}
	WORD	$1234

//
// floating-point coprocessor
//
//	LTYPEI cond freg ',' freg
//	{
//		outcode($1, $2, &$3, 0, &$5);
//	}
	ABSF.S	F1, F2

//	LTYPEK cond frcon ',' freg
//	{
//		outcode($1, $2, &$3, 0, &$5);
//	}
	ADDD.S	F1, F2
	MOVF	$0.5, F2 // MOVF $(0.5), F2

//	LTYPEK cond frcon ',' LFREG ',' freg
//	{
//		outcode($1, $2, &$3, $5, &$7);
//	}
	ADDD.S	F1, F2, F3

//	LTYPEL cond freg ',' freg
//	{
//		outcode($1, $2, &$3, int32($5.Reg), &nullgen);
//	}
	CMPD.S	F1, F2

//
// MCR MRC
//
//	LTYPEJ cond con ',' expr ',' spreg ',' creg ',' creg oexpr
//	{
//		var g obj.Addr
//
//		g = nullgen;
//		g.Type = obj.TYPE_CONST;
//		g.Offset = int64(
//			(0xe << 24) |		/* opcode */
//			($1 << 20) |		/* MCR/MRC */
//			(($2^C_SCOND_XOR) << 28) |		/* scond */
//			(($3 & 15) << 8) |	/* coprocessor number */
//			(($5 & 7) << 21) |	/* coprocessor operation */
//			(($7 & 15) << 12) |	/* arm register */
//			(($9 & 15) << 16) |	/* Crn */
//			(($11 & 15) << 0) |	/* Crm */
//			(($12 & 7) << 5) |	/* coprocessor information */
//			(1<<4));			/* must be set */
//		outcode(AMRC, Always, &nullgen, 0, &g);
//	}
	MRC.S	4, 6, R1, C2, C3, 7 // MRC $8301712627
	MCR.S	4, 6, R1, C2, C3, 7 // MRC $8300664051

//
// MULL r1,r2,(hi,lo)
//
//	LTYPEM cond reg ',' reg ',' regreg
//	{
//		outcode($1, $2, &$3, int32($5.Reg), &$7);
//	}
	MULL	R1, R2, (R3,R4)

//
// MULA r1,r2,r3,r4: (r1*r2+r3) & 0xffffffff . r4
// MULAW{T,B} r1,r2,r3,r4
//
//	LTYPEN cond reg ',' reg ',' reg ',' spreg
//	{
//		$7.Type = obj.TYPE_REGREG2;
//		$7.Offset = int64($9);
//		outcode($1, $2, &$3, int32($5.Reg), &$7);
//	}
	MULAWT	R1, R2, R3, R4
//
// PLD
//
//	LTYPEPLD oreg
//	{
//		outcode($1, Always, &$2, 0, &nullgen);
//	}
	PLD	(R1)
	PLD	4(R1)

//
// RET
//
//	LTYPEA cond
//	{
//		outcode($1, $2, &nullgen, 0, &nullgen);
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

// CMPF and CMPD are special.
	CMPF F1, F2
	CMPD F1, F2

//
// END
//
//	LTYPEE
//	{
//		outcode($1, Always, &nullgen, 0, &nullgen);
//	}
	END
