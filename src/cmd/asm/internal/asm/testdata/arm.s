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

// AND
	AND	$255, R0, R1         // ff1000e2
	AND	$4278190080, R0, R1  // ff1400e2
	AND.S	$255, R0, R1         // ff1010e2
	AND.S	$4278190080, R0, R1  // ff1410e2
	AND	$255, R0             // ff0000e2
	AND	$4278190080, R0      // ff0400e2
	AND.S	$255, R0             // ff0010e2
	AND.S	$4278190080, R0      // ff0410e2
	AND	R0, R1, R2           // 002001e0
	AND.S	R0, R1, R2           // 002011e0
	AND	R0, R1               // 001001e0
	AND.S	R0, R1               // 001011e0
	AND	R0>>28, R1, R2       // 202e01e0
	AND	R0<<28, R1, R2       // 002e01e0
	AND	R0->28, R1, R2       // 402e01e0
	AND	R0@>28, R1, R2       // 602e01e0
	AND.S	R0>>28, R1, R2       // 202e11e0
	AND.S	R0<<28, R1, R2       // 002e11e0
	AND.S	R0->28, R1, R2       // 402e11e0
	AND.S	R0@>28, R1, R2       // 602e11e0
	AND	R0<<28, R1           // 001e01e0
	AND	R0>>28, R1           // 201e01e0
	AND	R0->28, R1           // 401e01e0
	AND	R0@>28, R1           // 601e01e0
	AND.S	R0<<28, R1           // 001e11e0
	AND.S	R0>>28, R1           // 201e11e0
	AND.S	R0->28, R1           // 401e11e0
	AND.S	R0@>28, R1           // 601e11e0
	AND	R0<<R1, R2, R3       // 103102e0
	AND	R0>>R1, R2, R3       // 303102e0
	AND	R0->R1, R2, R3       // 503102e0
	AND	R0@>R1, R2, R3       // 703102e0
	AND.S	R0<<R1, R2, R3       // 103112e0
	AND.S	R0>>R1, R2, R3       // 303112e0
	AND.S	R0->R1, R2, R3       // 503112e0
	AND.S	R0@>R1, R2, R3       // 703112e0
	AND	R0<<R1, R2           // 102102e0
	AND	R0>>R1, R2           // 302102e0
	AND	R0->R1, R2           // 502102e0
	AND	R0@>R1, R2           // 702102e0
	AND.S	R0<<R1, R2           // 102112e0
	AND.S	R0>>R1, R2           // 302112e0
	AND.S	R0->R1, R2           // 502112e0
	AND.S	R0@>R1, R2           // 702112e0

// EOR
	EOR	$255, R0, R1         // ff1020e2
	EOR	$4278190080, R0, R1  // ff1420e2
	EOR.S	$255, R0, R1         // ff1030e2
	EOR.S	$4278190080, R0, R1  // ff1430e2
	EOR	$255, R0             // ff0020e2
	EOR	$4278190080, R0      // ff0420e2
	EOR.S	$255, R0             // ff0030e2
	EOR.S	$4278190080, R0      // ff0430e2
	EOR	R0, R1, R2           // 002021e0
	EOR.S	R0, R1, R2           // 002031e0
	EOR	R0, R1               // 001021e0
	EOR.S	R0, R1               // 001031e0
	EOR	R0>>28, R1, R2       // 202e21e0
	EOR	R0<<28, R1, R2       // 002e21e0
	EOR	R0->28, R1, R2       // 402e21e0
	EOR	R0@>28, R1, R2       // 602e21e0
	EOR.S	R0>>28, R1, R2       // 202e31e0
	EOR.S	R0<<28, R1, R2       // 002e31e0
	EOR.S	R0->28, R1, R2       // 402e31e0
	EOR.S	R0@>28, R1, R2       // 602e31e0
	EOR	R0<<28, R1           // 001e21e0
	EOR	R0>>28, R1           // 201e21e0
	EOR	R0->28, R1           // 401e21e0
	EOR	R0@>28, R1           // 601e21e0
	EOR.S	R0<<28, R1           // 001e31e0
	EOR.S	R0>>28, R1           // 201e31e0
	EOR.S	R0->28, R1           // 401e31e0
	EOR.S	R0@>28, R1           // 601e31e0
	EOR	R0<<R1, R2, R3       // 103122e0
	EOR	R0>>R1, R2, R3       // 303122e0
	EOR	R0->R1, R2, R3       // 503122e0
	EOR	R0@>R1, R2, R3       // 703122e0
	EOR.S	R0<<R1, R2, R3       // 103132e0
	EOR.S	R0>>R1, R2, R3       // 303132e0
	EOR.S	R0->R1, R2, R3       // 503132e0
	EOR.S	R0@>R1, R2, R3       // 703132e0
	EOR	R0<<R1, R2           // 102122e0
	EOR	R0>>R1, R2           // 302122e0
	EOR	R0->R1, R2           // 502122e0
	EOR	R0@>R1, R2           // 702122e0
	EOR.S	R0<<R1, R2           // 102132e0
	EOR.S	R0>>R1, R2           // 302132e0
	EOR.S	R0->R1, R2           // 502132e0
	EOR.S	R0@>R1, R2           // 702132e0

// ORR
	ORR	$255, R0, R1         // ff1080e3
	ORR	$4278190080, R0, R1  // ff1480e3
	ORR.S	$255, R0, R1         // ff1090e3
	ORR.S	$4278190080, R0, R1  // ff1490e3
	ORR	$255, R0             // ff0080e3
	ORR	$4278190080, R0      // ff0480e3
	ORR.S	$255, R0             // ff0090e3
	ORR.S	$4278190080, R0      // ff0490e3
	ORR	R0, R1, R2           // 002081e1
	ORR.S	R0, R1, R2           // 002091e1
	ORR	R0, R1               // 001081e1
	ORR.S	R0, R1               // 001091e1
	ORR	R0>>28, R1, R2       // 202e81e1
	ORR	R0<<28, R1, R2       // 002e81e1
	ORR	R0->28, R1, R2       // 402e81e1
	ORR	R0@>28, R1, R2       // 602e81e1
	ORR.S	R0>>28, R1, R2       // 202e91e1
	ORR.S	R0<<28, R1, R2       // 002e91e1
	ORR.S	R0->28, R1, R2       // 402e91e1
	ORR.S	R0@>28, R1, R2       // 602e91e1
	ORR	R0<<28, R1           // 001e81e1
	ORR	R0>>28, R1           // 201e81e1
	ORR	R0->28, R1           // 401e81e1
	ORR	R0@>28, R1           // 601e81e1
	ORR.S	R0<<28, R1           // 001e91e1
	ORR.S	R0>>28, R1           // 201e91e1
	ORR.S	R0->28, R1           // 401e91e1
	ORR.S	R0@>28, R1           // 601e91e1
	ORR	R0<<R1, R2, R3       // 103182e1
	ORR	R0>>R1, R2, R3       // 303182e1
	ORR	R0->R1, R2, R3       // 503182e1
	ORR	R0@>R1, R2, R3       // 703182e1
	ORR.S	R0<<R1, R2, R3       // 103192e1
	ORR.S	R0>>R1, R2, R3       // 303192e1
	ORR.S	R0->R1, R2, R3       // 503192e1
	ORR.S	R0@>R1, R2, R3       // 703192e1
	ORR	R0<<R1, R2           // 102182e1
	ORR	R0>>R1, R2           // 302182e1
	ORR	R0->R1, R2           // 502182e1
	ORR	R0@>R1, R2           // 702182e1
	ORR.S	R0<<R1, R2           // 102192e1
	ORR.S	R0>>R1, R2           // 302192e1
	ORR.S	R0->R1, R2           // 502192e1
	ORR.S	R0@>R1, R2           // 702192e1

// SUB
	SUB	$255, R0, R1         // ff1040e2
	SUB	$4278190080, R0, R1  // ff1440e2
	SUB.S	$255, R0, R1         // ff1050e2
	SUB.S	$4278190080, R0, R1  // ff1450e2
	SUB	$255, R0             // ff0040e2
	SUB	$4278190080, R0      // ff0440e2
	SUB.S	$255, R0             // ff0050e2
	SUB.S	$4278190080, R0      // ff0450e2
	SUB	R0, R1, R2           // 002041e0
	SUB.S	R0, R1, R2           // 002051e0
	SUB	R0, R1               // 001041e0
	SUB.S	R0, R1               // 001051e0
	SUB	R0>>28, R1, R2       // 202e41e0
	SUB	R0<<28, R1, R2       // 002e41e0
	SUB	R0->28, R1, R2       // 402e41e0
	SUB	R0@>28, R1, R2       // 602e41e0
	SUB.S	R0>>28, R1, R2       // 202e51e0
	SUB.S	R0<<28, R1, R2       // 002e51e0
	SUB.S	R0->28, R1, R2       // 402e51e0
	SUB.S	R0@>28, R1, R2       // 602e51e0
	SUB	R0<<28, R1           // 001e41e0
	SUB	R0>>28, R1           // 201e41e0
	SUB	R0->28, R1           // 401e41e0
	SUB	R0@>28, R1           // 601e41e0
	SUB.S	R0<<28, R1           // 001e51e0
	SUB.S	R0>>28, R1           // 201e51e0
	SUB.S	R0->28, R1           // 401e51e0
	SUB.S	R0@>28, R1           // 601e51e0
	SUB	R0<<R1, R2, R3       // 103142e0
	SUB	R0>>R1, R2, R3       // 303142e0
	SUB	R0->R1, R2, R3       // 503142e0
	SUB	R0@>R1, R2, R3       // 703142e0
	SUB.S	R0<<R1, R2, R3       // 103152e0
	SUB.S	R0>>R1, R2, R3       // 303152e0
	SUB.S	R0->R1, R2, R3       // 503152e0
	SUB.S	R0@>R1, R2, R3       // 703152e0
	SUB	R0<<R1, R2           // 102142e0
	SUB	R0>>R1, R2           // 302142e0
	SUB	R0->R1, R2           // 502142e0
	SUB	R0@>R1, R2           // 702142e0
	SUB.S	R0<<R1, R2           // 102152e0
	SUB.S	R0>>R1, R2           // 302152e0
	SUB.S	R0->R1, R2           // 502152e0
	SUB.S	R0@>R1, R2           // 702152e0

// SBC
	SBC	$255, R0, R1         // ff10c0e2
	SBC	$4278190080, R0, R1  // ff14c0e2
	SBC.S	$255, R0, R1         // ff10d0e2
	SBC.S	$4278190080, R0, R1  // ff14d0e2
	SBC	$255, R0             // ff00c0e2
	SBC	$4278190080, R0      // ff04c0e2
	SBC.S	$255, R0             // ff00d0e2
	SBC.S	$4278190080, R0      // ff04d0e2
	SBC	R0, R1, R2           // 0020c1e0
	SBC.S	R0, R1, R2           // 0020d1e0
	SBC	R0, R1               // 0010c1e0
	SBC.S	R0, R1               // 0010d1e0
	SBC	R0>>28, R1, R2       // 202ec1e0
	SBC	R0<<28, R1, R2       // 002ec1e0
	SBC	R0->28, R1, R2       // 402ec1e0
	SBC	R0@>28, R1, R2       // 602ec1e0
	SBC.S	R0>>28, R1, R2       // 202ed1e0
	SBC.S	R0<<28, R1, R2       // 002ed1e0
	SBC.S	R0->28, R1, R2       // 402ed1e0
	SBC.S	R0@>28, R1, R2       // 602ed1e0
	SBC	R0<<28, R1           // 001ec1e0
	SBC	R0>>28, R1           // 201ec1e0
	SBC	R0->28, R1           // 401ec1e0
	SBC	R0@>28, R1           // 601ec1e0
	SBC.S	R0<<28, R1           // 001ed1e0
	SBC.S	R0>>28, R1           // 201ed1e0
	SBC.S	R0->28, R1           // 401ed1e0
	SBC.S	R0@>28, R1           // 601ed1e0
	SBC	R0<<R1, R2, R3       // 1031c2e0
	SBC	R0>>R1, R2, R3       // 3031c2e0
	SBC	R0->R1, R2, R3       // 5031c2e0
	SBC	R0@>R1, R2, R3       // 7031c2e0
	SBC.S	R0<<R1, R2, R3       // 1031d2e0
	SBC.S	R0>>R1, R2, R3       // 3031d2e0
	SBC.S	R0->R1, R2, R3       // 5031d2e0
	SBC.S	R0@>R1, R2, R3       // 7031d2e0
	SBC	R0<<R1, R2           // 1021c2e0
	SBC	R0>>R1, R2           // 3021c2e0
	SBC	R0->R1, R2           // 5021c2e0
	SBC	R0@>R1, R2           // 7021c2e0
	SBC.S	R0<<R1, R2           // 1021d2e0
	SBC.S	R0>>R1, R2           // 3021d2e0
	SBC.S	R0->R1, R2           // 5021d2e0
	SBC.S	R0@>R1, R2           // 7021d2e0

// RSB
	RSB	$255, R0, R1         // ff1060e2
	RSB	$4278190080, R0, R1  // ff1460e2
	RSB.S	$255, R0, R1         // ff1070e2
	RSB.S	$4278190080, R0, R1  // ff1470e2
	RSB	$255, R0             // ff0060e2
	RSB	$4278190080, R0      // ff0460e2
	RSB.S	$255, R0             // ff0070e2
	RSB.S	$4278190080, R0      // ff0470e2
	RSB	R0, R1, R2           // 002061e0
	RSB.S	R0, R1, R2           // 002071e0
	RSB	R0, R1               // 001061e0
	RSB.S	R0, R1               // 001071e0
	RSB	R0>>28, R1, R2       // 202e61e0
	RSB	R0<<28, R1, R2       // 002e61e0
	RSB	R0->28, R1, R2       // 402e61e0
	RSB	R0@>28, R1, R2       // 602e61e0
	RSB.S	R0>>28, R1, R2       // 202e71e0
	RSB.S	R0<<28, R1, R2       // 002e71e0
	RSB.S	R0->28, R1, R2       // 402e71e0
	RSB.S	R0@>28, R1, R2       // 602e71e0
	RSB	R0<<28, R1           // 001e61e0
	RSB	R0>>28, R1           // 201e61e0
	RSB	R0->28, R1           // 401e61e0
	RSB	R0@>28, R1           // 601e61e0
	RSB.S	R0<<28, R1           // 001e71e0
	RSB.S	R0>>28, R1           // 201e71e0
	RSB.S	R0->28, R1           // 401e71e0
	RSB.S	R0@>28, R1           // 601e71e0
	RSB	R0<<R1, R2, R3       // 103162e0
	RSB	R0>>R1, R2, R3       // 303162e0
	RSB	R0->R1, R2, R3       // 503162e0
	RSB	R0@>R1, R2, R3       // 703162e0
	RSB.S	R0<<R1, R2, R3       // 103172e0
	RSB.S	R0>>R1, R2, R3       // 303172e0
	RSB.S	R0->R1, R2, R3       // 503172e0
	RSB.S	R0@>R1, R2, R3       // 703172e0
	RSB	R0<<R1, R2           // 102162e0
	RSB	R0>>R1, R2           // 302162e0
	RSB	R0->R1, R2           // 502162e0
	RSB	R0@>R1, R2           // 702162e0
	RSB.S	R0<<R1, R2           // 102172e0
	RSB.S	R0>>R1, R2           // 302172e0
	RSB.S	R0->R1, R2           // 502172e0
	RSB.S	R0@>R1, R2           // 702172e0

// RSC
	RSC	$255, R0, R1         // ff10e0e2
	RSC	$4278190080, R0, R1  // ff14e0e2
	RSC.S	$255, R0, R1         // ff10f0e2
	RSC.S	$4278190080, R0, R1  // ff14f0e2
	RSC	$255, R0             // ff00e0e2
	RSC	$4278190080, R0      // ff04e0e2
	RSC.S	$255, R0             // ff00f0e2
	RSC.S	$4278190080, R0      // ff04f0e2
	RSC	R0, R1, R2           // 0020e1e0
	RSC.S	R0, R1, R2           // 0020f1e0
	RSC	R0, R1               // 0010e1e0
	RSC.S	R0, R1               // 0010f1e0
	RSC	R0>>28, R1, R2       // 202ee1e0
	RSC	R0<<28, R1, R2       // 002ee1e0
	RSC	R0->28, R1, R2       // 402ee1e0
	RSC	R0@>28, R1, R2       // 602ee1e0
	RSC.S	R0>>28, R1, R2       // 202ef1e0
	RSC.S	R0<<28, R1, R2       // 002ef1e0
	RSC.S	R0->28, R1, R2       // 402ef1e0
	RSC.S	R0@>28, R1, R2       // 602ef1e0
	RSC	R0<<28, R1           // 001ee1e0
	RSC	R0>>28, R1           // 201ee1e0
	RSC	R0->28, R1           // 401ee1e0
	RSC	R0@>28, R1           // 601ee1e0
	RSC.S	R0<<28, R1           // 001ef1e0
	RSC.S	R0>>28, R1           // 201ef1e0
	RSC.S	R0->28, R1           // 401ef1e0
	RSC.S	R0@>28, R1           // 601ef1e0
	RSC	R0<<R1, R2, R3       // 1031e2e0
	RSC	R0>>R1, R2, R3       // 3031e2e0
	RSC	R0->R1, R2, R3       // 5031e2e0
	RSC	R0@>R1, R2, R3       // 7031e2e0
	RSC.S	R0<<R1, R2, R3       // 1031f2e0
	RSC.S	R0>>R1, R2, R3       // 3031f2e0
	RSC.S	R0->R1, R2, R3       // 5031f2e0
	RSC.S	R0@>R1, R2, R3       // 7031f2e0
	RSC	R0<<R1, R2           // 1021e2e0
	RSC	R0>>R1, R2           // 3021e2e0
	RSC	R0->R1, R2           // 5021e2e0
	RSC	R0@>R1, R2           // 7021e2e0
	RSC.S	R0<<R1, R2           // 1021f2e0
	RSC.S	R0>>R1, R2           // 3021f2e0
	RSC.S	R0->R1, R2           // 5021f2e0
	RSC.S	R0@>R1, R2           // 7021f2e0

// ADD
	ADD	$255, R0, R1         // ff1080e2
	ADD	$4278190080, R0, R1  // ff1480e2
	ADD.S	$255, R0, R1         // ff1090e2
	ADD.S	$4278190080, R0, R1  // ff1490e2
	ADD	$255, R0             // ff0080e2
	ADD	$4278190080, R0      // ff0480e2
	ADD.S	$255, R0             // ff0090e2
	ADD.S	$4278190080, R0      // ff0490e2
	ADD	R0, R1, R2           // 002081e0
	ADD.S	R0, R1, R2           // 002091e0
	ADD	R0, R1               // 001081e0
	ADD.S	R0, R1               // 001091e0
	ADD	R0>>28, R1, R2       // 202e81e0
	ADD	R0<<28, R1, R2       // 002e81e0
	ADD	R0->28, R1, R2       // 402e81e0
	ADD	R0@>28, R1, R2       // 602e81e0
	ADD.S	R0>>28, R1, R2       // 202e91e0
	ADD.S	R0<<28, R1, R2       // 002e91e0
	ADD.S	R0->28, R1, R2       // 402e91e0
	ADD.S	R0@>28, R1, R2       // 602e91e0
	ADD	R0<<28, R1           // 001e81e0
	ADD	R0>>28, R1           // 201e81e0
	ADD	R0->28, R1           // 401e81e0
	ADD	R0@>28, R1           // 601e81e0
	ADD.S	R0<<28, R1           // 001e91e0
	ADD.S	R0>>28, R1           // 201e91e0
	ADD.S	R0->28, R1           // 401e91e0
	ADD.S	R0@>28, R1           // 601e91e0
	ADD	R0<<R1, R2, R3       // 103182e0
	ADD	R0>>R1, R2, R3       // 303182e0
	ADD	R0->R1, R2, R3       // 503182e0
	ADD	R0@>R1, R2, R3       // 703182e0
	ADD.S	R0<<R1, R2, R3       // 103192e0
	ADD.S	R0>>R1, R2, R3       // 303192e0
	ADD.S	R0->R1, R2, R3       // 503192e0
	ADD.S	R0@>R1, R2, R3       // 703192e0
	ADD	R0<<R1, R2           // 102182e0
	ADD	R0>>R1, R2           // 302182e0
	ADD	R0->R1, R2           // 502182e0
	ADD	R0@>R1, R2           // 702182e0
	ADD.S	R0<<R1, R2           // 102192e0
	ADD.S	R0>>R1, R2           // 302192e0
	ADD.S	R0->R1, R2           // 502192e0
	ADD.S	R0@>R1, R2           // 702192e0

// ADC
	ADC	$255, R0, R1         // ff10a0e2
	ADC	$4278190080, R0, R1  // ff14a0e2
	ADC.S	$255, R0, R1         // ff10b0e2
	ADC.S	$4278190080, R0, R1  // ff14b0e2
	ADC	$255, R0             // ff00a0e2
	ADC	$4278190080, R0      // ff04a0e2
	ADC.S	$255, R0             // ff00b0e2
	ADC.S	$4278190080, R0      // ff04b0e2
	ADC	R0, R1, R2           // 0020a1e0
	ADC.S	R0, R1, R2           // 0020b1e0
	ADC	R0, R1               // 0010a1e0
	ADC.S	R0, R1               // 0010b1e0
	ADC	R0>>28, R1, R2       // 202ea1e0
	ADC	R0<<28, R1, R2       // 002ea1e0
	ADC	R0->28, R1, R2       // 402ea1e0
	ADC	R0@>28, R1, R2       // 602ea1e0
	ADC.S	R0>>28, R1, R2       // 202eb1e0
	ADC.S	R0<<28, R1, R2       // 002eb1e0
	ADC.S	R0->28, R1, R2       // 402eb1e0
	ADC.S	R0@>28, R1, R2       // 602eb1e0
	ADC	R0<<28, R1           // 001ea1e0
	ADC	R0>>28, R1           // 201ea1e0
	ADC	R0->28, R1           // 401ea1e0
	ADC	R0@>28, R1           // 601ea1e0
	ADC.S	R0<<28, R1           // 001eb1e0
	ADC.S	R0>>28, R1           // 201eb1e0
	ADC.S	R0->28, R1           // 401eb1e0
	ADC.S	R0@>28, R1           // 601eb1e0
	ADC	R0<<R1, R2, R3       // 1031a2e0
	ADC	R0>>R1, R2, R3       // 3031a2e0
	ADC	R0->R1, R2, R3       // 5031a2e0
	ADC	R0@>R1, R2, R3       // 7031a2e0
	ADC.S	R0<<R1, R2, R3       // 1031b2e0
	ADC.S	R0>>R1, R2, R3       // 3031b2e0
	ADC.S	R0->R1, R2, R3       // 5031b2e0
	ADC.S	R0@>R1, R2, R3       // 7031b2e0
	ADC	R0<<R1, R2           // 1021a2e0
	ADC	R0>>R1, R2           // 3021a2e0
	ADC	R0->R1, R2           // 5021a2e0
	ADC	R0@>R1, R2           // 7021a2e0
	ADC.S	R0<<R1, R2           // 1021b2e0
	ADC.S	R0>>R1, R2           // 3021b2e0
	ADC.S	R0->R1, R2           // 5021b2e0
	ADC.S	R0@>R1, R2           // 7021b2e0

// TEQ
	TEQ	$255, R7             // ff0037e3
	TEQ	$4278190080, R9      // ff0439e3
	TEQ	R9<<30, R7           // 090f37e1
	TEQ	R9>>30, R7           // 290f37e1
	TEQ	R9->30, R7           // 490f37e1
	TEQ	R9@>30, R7           // 690f37e1
	TEQ	R9<<R8, R7           // 190837e1
	TEQ	R9>>R8, R7           // 390837e1
	TEQ	R9->R8, R7           // 590837e1
	TEQ	R9@>R8, R7           // 790837e1

// TST
	TST	$255, R7             // ff0017e3
	TST	$4278190080, R9      // ff0419e3
	TST	R9<<30, R7           // 090f17e1
	TST	R9>>30, R7           // 290f17e1
	TST	R9->30, R7           // 490f17e1
	TST	R9@>30, R7           // 690f17e1
	TST	R9<<R8, R7           // 190817e1
	TST	R9>>R8, R7           // 390817e1
	TST	R9->R8, R7           // 590817e1
	TST	R9@>R8, R7           // 790817e1

// CMP
	CMP	$255, R7             // ff0057e3
	CMP	$4278190080, R9      // ff0459e3
	CMP	R9<<30, R7           // 090f57e1
	CMP	R9>>30, R7           // 290f57e1
	CMP	R9->30, R7           // 490f57e1
	CMP	R9@>30, R7           // 690f57e1
	CMP	R9<<R8, R7           // 190857e1
	CMP	R9>>R8, R7           // 390857e1
	CMP	R9->R8, R7           // 590857e1
	CMP	R9@>R8, R7           // 790857e1

// CMN
	CMN	$255, R7             // ff0077e3
	CMN	$4278190080, R9      // ff0479e3
	CMN	R9<<30, R7           // 090f77e1
	CMN	R9>>30, R7           // 290f77e1
	CMN	R9->30, R7           // 490f77e1
	CMN	R9@>30, R7           // 690f77e1
	CMN	R9<<R8, R7           // 190877e1
	CMN	R9>>R8, R7           // 390877e1
	CMN	R9->R8, R7           // 590877e1
	CMN	R9@>R8, R7           // 790877e1

// B*
	BEQ	14(PC) // BEQ 14(PC)   // 0c00000a
	BNE	13(PC) // BNE 13(PC)   // 0b00001a
	BCS	12(PC) // BCS 12(PC)   // 0a00002a
	BCC	11(PC) // BCC 11(PC)   // 0900003a
	BMI	10(PC) // BMI 10(PC)   // 0800004a
	BPL	9(PC)  // BPL 9(PC)    // 0700005a
	BVS	8(PC)  // BVS 8(PC)    // 0600006a
	BVC	7(PC)  // BVC 7(PC)    // 0500007a
	BHI	6(PC)  // BHI 6(PC)    // 0400008a
	BLS	5(PC)  // BLS 5(PC)    // 0300009a
	BGE	4(PC)  // BGE 4(PC)    // 020000aa
	BLT	3(PC)  // BLT 3(PC)    // 010000ba
	BGT	2(PC)  // BGT 2(PC)    // 000000ca
	BLE	1(PC)  // BLE 1(PC)    // ffffffda
	ADD	$0, R0, R0
	B	-1(PC) // JMP -1(PC)   // fdffffea
	B	-2(PC) // JMP -2(PC)   // fcffffea
	B	-3(PC) // JMP -3(PC)   // fbffffea
	B	-4(PC) // JMP -4(PC)   // faffffea
	B	-5(PC) // JMP -5(PC)   // f9ffffea
	B	jmp_label_0 // JMP     // 010000ea
	B	jmp_label_0 // JMP     // 000000ea
	B	jmp_label_0 // JMP     // ffffffea
jmp_label_0:
	ADD	$0, R0, R0
	BEQ	jmp_label_0 // BEQ 521 // fdffff0a
	BNE	jmp_label_0 // BNE 521 // fcffff1a
	BCS	jmp_label_0 // BCS 521 // fbffff2a
	BCC	jmp_label_0 // BCC 521 // faffff3a
	BMI	jmp_label_0 // BMI 521 // f9ffff4a
	BPL	jmp_label_0 // BPL 521 // f8ffff5a
	BVS	jmp_label_0 // BVS 521 // f7ffff6a
	BVC	jmp_label_0 // BVC 521 // f6ffff7a
	BHI	jmp_label_0 // BHI 521 // f5ffff8a
	BLS	jmp_label_0 // BLS 521 // f4ffff9a
	BGE	jmp_label_0 // BGE 521 // f3ffffaa
	BLT	jmp_label_0 // BLT 521 // f2ffffba
	BGT	jmp_label_0 // BGT 521 // f1ffffca
	BLE	jmp_label_0 // BLE 521 // f0ffffda
	B	jmp_label_0 // JMP 521 // efffffea
	B	0(PC)    // JMP 0(PC)  // feffffea
jmp_label_1:
	B	jmp_label_1 // JMP     // feffffea

// BL
	BL.EQ	14(PC) // CALL.EQ 14(PC)   // 0c00000b
	BL.NE	13(PC) // CALL.NE 13(PC)   // 0b00001b
	BL.CS	12(PC) // CALL.CS 12(PC)   // 0a00002b
	BL.CC	11(PC) // CALL.CC 11(PC)   // 0900003b
	BL.MI	10(PC) // CALL.MI 10(PC)   // 0800004b
	BL.PL	9(PC)  // CALL.PL 9(PC)    // 0700005b
	BL.VS	8(PC)  // CALL.VS 8(PC)    // 0600006b
	BL.VC	7(PC)  // CALL.VC 7(PC)    // 0500007b
	BL.HI	6(PC)  // CALL.HI 6(PC)    // 0400008b
	BL.LS	5(PC)  // CALL.LS 5(PC)    // 0300009b
	BL.GE	4(PC)  // CALL.GE 4(PC)    // 020000ab
	BL.LT	3(PC)  // CALL.LT 3(PC)    // 010000bb
	BL.GT	2(PC)  // CALL.GT 2(PC)    // 000000cb
	BL.LE	1(PC)  // CALL.LE 1(PC)    // ffffffdb
	ADD	$0, R0, R0
	BL	-1(PC) // CALL -1(PC)      // fdffffeb
	BL	-2(PC) // CALL -2(PC)      // fcffffeb
	BL	-3(PC) // CALL -3(PC)      // fbffffeb
	BL	-4(PC) // CALL -4(PC)      // faffffeb
	BL	-5(PC) // CALL -5(PC)      // f9ffffeb
	BL	jmp_label_2 // CALL        // 010000eb
	BL	jmp_label_2 // CALL        // 000000eb
	BL	jmp_label_2 // CALL        // ffffffeb
jmp_label_2:
	ADD	$0, R0, R0
	BL.EQ	jmp_label_2 // CALL.EQ 562 // fdffff0b
	BL.NE	jmp_label_2 // CALL.NE 562 // fcffff1b
	BL.CS	jmp_label_2 // CALL.CS 562 // fbffff2b
	BL.CC	jmp_label_2 // CALL.CC 562 // faffff3b
	BL.MI	jmp_label_2 // CALL.MI 562 // f9ffff4b
	BL.PL	jmp_label_2 // CALL.PL 562 // f8ffff5b
	BL.VS	jmp_label_2 // CALL.VS 562 // f7ffff6b
	BL.VC	jmp_label_2 // CALL.VC 562 // f6ffff7b
	BL.HI	jmp_label_2 // CALL.HI 562 // f5ffff8b
	BL.LS	jmp_label_2 // CALL.LS 562 // f4ffff9b
	BL.GE	jmp_label_2 // CALL.GE 562 // f3ffffab
	BL.LT	jmp_label_2 // CALL.LT 562 // f2ffffbb
	BL.GT	jmp_label_2 // CALL.GT 562 // f1ffffcb
	BL.LE	jmp_label_2 // CALL.LE 562 // f0ffffdb
	BL	jmp_label_2 // CALL 562    // efffffeb
	BL	0(PC)    // CALL 0(PC)     // feffffeb
jmp_label_3:
	BL	jmp_label_3 // CALL        // feffffeb

// BIC
	BIC	$255, R0, R1         // ff10c0e3
	BIC	$4278190080, R0, R1  // ff14c0e3
	BIC.S	$255, R0, R1         // ff10d0e3
	BIC.S	$4278190080, R0, R1  // ff14d0e3
	BIC	$255, R0             // ff00c0e3
	BIC	$4278190080, R0      // ff04c0e3
	BIC.S	$255, R0             // ff00d0e3
	BIC.S	$4278190080, R0      // ff04d0e3
	BIC	R0, R1, R2           // 0020c1e1
	BIC.S	R0, R1, R2           // 0020d1e1
	BIC	R0, R1               // 0010c1e1
	BIC.S	R0, R1               // 0010d1e1
	BIC	R0>>28, R1, R2       // 202ec1e1
	BIC	R0<<28, R1, R2       // 002ec1e1
	BIC	R0->28, R1, R2       // 402ec1e1
	BIC	R0@>28, R1, R2       // 602ec1e1
	BIC.S	R0>>28, R1, R2       // 202ed1e1
	BIC.S	R0<<28, R1, R2       // 002ed1e1
	BIC.S	R0->28, R1, R2       // 402ed1e1
	BIC.S	R0@>28, R1, R2       // 602ed1e1
	BIC	R0<<28, R1           // 001ec1e1
	BIC	R0>>28, R1           // 201ec1e1
	BIC	R0->28, R1           // 401ec1e1
	BIC	R0@>28, R1           // 601ec1e1
	BIC.S	R0<<28, R1           // 001ed1e1
	BIC.S	R0>>28, R1           // 201ed1e1
	BIC.S	R0->28, R1           // 401ed1e1
	BIC.S	R0@>28, R1           // 601ed1e1
	BIC	R0<<R1, R2, R3       // 1031c2e1
	BIC	R0>>R1, R2, R3       // 3031c2e1
	BIC	R0->R1, R2, R3       // 5031c2e1
	BIC	R0@>R1, R2, R3       // 7031c2e1
	BIC.S	R0<<R1, R2, R3       // 1031d2e1
	BIC.S	R0>>R1, R2, R3       // 3031d2e1
	BIC.S	R0->R1, R2, R3       // 5031d2e1
	BIC.S	R0@>R1, R2, R3       // 7031d2e1
	BIC	R0<<R1, R2           // 1021c2e1
	BIC	R0>>R1, R2           // 3021c2e1
	BIC	R0->R1, R2           // 5021c2e1
	BIC	R0@>R1, R2           // 7021c2e1
	BIC.S	R0<<R1, R2           // 1021d2e1
	BIC.S	R0>>R1, R2           // 3021d2e1
	BIC.S	R0->R1, R2           // 5021d2e1
	BIC.S	R0@>R1, R2           // 7021d2e1

// SRL
	SRL	$14, R5, R6          // 2567a0e1
	SRL	$15, R5, R6          // a567a0e1
	SRL	$30, R5, R6          // 256fa0e1
	SRL	$31, R5, R6          // a56fa0e1
	SRL.S	$14, R5, R6          // 2567b0e1
	SRL.S	$15, R5, R6          // a567b0e1
	SRL.S	$30, R5, R6          // 256fb0e1
	SRL.S	$31, R5, R6          // a56fb0e1
	SRL	$14, R5              // 2557a0e1
	SRL	$15, R5              // a557a0e1
	SRL	$30, R5              // 255fa0e1
	SRL	$31, R5              // a55fa0e1
	SRL.S	$14, R5              // 2557b0e1
	SRL.S	$15, R5              // a557b0e1
	SRL.S	$30, R5              // 255fb0e1
	SRL.S	$31, R5              // a55fb0e1
	SRL	R5, R6, R7           // 3675a0e1
	SRL.S	R5, R6, R7           // 3675b0e1
	SRL	R5, R7               // 3775a0e1
	SRL.S	R5, R7               // 3775b0e1

// SRA
	SRA	$14, R5, R6          // 4567a0e1
	SRA	$15, R5, R6          // c567a0e1
	SRA	$30, R5, R6          // 456fa0e1
	SRA	$31, R5, R6          // c56fa0e1
	SRA.S	$14, R5, R6          // 4567b0e1
	SRA.S	$15, R5, R6          // c567b0e1
	SRA.S	$30, R5, R6          // 456fb0e1
	SRA.S	$31, R5, R6          // c56fb0e1
	SRA	$14, R5              // 4557a0e1
	SRA	$15, R5              // c557a0e1
	SRA	$30, R5              // 455fa0e1
	SRA	$31, R5              // c55fa0e1
	SRA.S	$14, R5              // 4557b0e1
	SRA.S	$15, R5              // c557b0e1
	SRA.S	$30, R5              // 455fb0e1
	SRA.S	$31, R5              // c55fb0e1
	SRA	R5, R6, R7           // 5675a0e1
	SRA.S	R5, R6, R7           // 5675b0e1
	SRA	R5, R7               // 5775a0e1
	SRA.S	R5, R7               // 5775b0e1

// SLL
	SLL	$14, R5, R6          // 0567a0e1
	SLL	$15, R5, R6          // 8567a0e1
	SLL	$30, R5, R6          // 056fa0e1
	SLL	$31, R5, R6          // 856fa0e1
	SLL.S	$14, R5, R6          // 0567b0e1
	SLL.S	$15, R5, R6          // 8567b0e1
	SLL.S	$30, R5, R6          // 056fb0e1
	SLL.S	$31, R5, R6          // 856fb0e1
	SLL	$14, R5              // 0557a0e1
	SLL	$15, R5              // 8557a0e1
	SLL	$30, R5              // 055fa0e1
	SLL	$31, R5              // 855fa0e1
	SLL.S	$14, R5              // 0557b0e1
	SLL.S	$15, R5              // 8557b0e1
	SLL.S	$30, R5              // 055fb0e1
	SLL.S	$31, R5              // 855fb0e1
	SLL	R5, R6, R7           // 1675a0e1
	SLL.S	R5, R6, R7           // 1675b0e1
	SLL	R5, R7               // 1775a0e1
	SLL.S	R5, R7               // 1775b0e1

// MULA / MULS
	MULAWT	R1, R2, R3, R4       // c23124e1
	MULAWB	R1, R2, R3, R4       // 823124e1
	MULS	R1, R2, R3, R4       // 923164e0
	MMULA	R1, R2, R3, R4       // 123154e7
	MMULS	R1, R2, R3, R4       // d23154e7
	MULABB	R1, R2, R3, R4       // 823104e1

// MUL
	MMUL	R1, R2, R3           // 12f153e7
	MULBB	R1, R2, R3           // 82f163e1
	MULWB	R1, R2, R3           // a20123e1
	MULWT	R1, R2, R3           // e20123e1

// REV
	REV	R1, R2               // 312fbfe6
	REV16	R1, R2               // b12fbfe6
	REVSH	R1, R2               // b12fffe6
	RBIT	R1, R2               // 312fffe6

//
// END
//
//	LTYPEE
//	{
//		outcode($1, Always, &nullgen, 0, &nullgen);
//	}
	END
