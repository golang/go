// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This input was created by taking the instruction productions in
// the old assembler's (5a's) grammar and hand-writing complete
// instructions for each rule, to guarantee we cover the same space.

#include "../../../../../runtime/textflag.h"

TEXT	foo(SB), DUPOK|NOSPLIT, $0

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
	CLZ	R1, R2

//
// MOVW
//
//	LTYPE3 cond gen ',' gen
//	{
//		outcode($1, $2, &$3, 0, &$5);
//	}
	MOVW.S	R1, R2
	MOVW	$1, R2
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
	SWI	$2
	SWI	$3
//	SWI	foo(SB) - TODO: classifying foo(SB) as C_TLS_LE

//
// CMP
//
//	LTYPE7 cond imsr ',' spreg
//	{
//		outcode($1, $2, &$3, $5, &nullgen);
//	}
	CMP	$1, R2
	CMP	R1<<R2, R3
	CMP	R1, R2

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
	MOVM	(R1), [R2]

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
	MOVM	[R2], (R1)

//
// SWAP
//
//	LTYPE9 cond reg ',' ireg ',' reg
//	{
//		outcode($1, $2, &$5, int32($3.Reg), &$7);
//	}
	STREX	R1, (R2), R3 // STREX (R2), R1, R3

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
	ABSF	F1, F2

//	LTYPEK cond frcon ',' freg
//	{
//		outcode($1, $2, &$3, 0, &$5);
//	}
	ADDD	F1, F2
	MOVF	$0.5, F2 // MOVF $(0.5), F2

//	LTYPEK cond frcon ',' LFREG ',' freg
//	{
//		outcode($1, $2, &$3, $5, &$7);
//	}
	ADDD	F1, F2, F3

//	LTYPEL cond freg ',' freg
//	{
//		outcode($1, $2, &$3, int32($5.Reg), &nullgen);
//	}
	CMPD	F1, F2

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
	BEQ	jmp_label_0 // BEQ 519 // fdffff0a
	BNE	jmp_label_0 // BNE 519 // fcffff1a
	BCS	jmp_label_0 // BCS 519 // fbffff2a
	BCC	jmp_label_0 // BCC 519 // faffff3a
	BMI	jmp_label_0 // BMI 519 // f9ffff4a
	BPL	jmp_label_0 // BPL 519 // f8ffff5a
	BVS	jmp_label_0 // BVS 519 // f7ffff6a
	BVC	jmp_label_0 // BVC 519 // f6ffff7a
	BHI	jmp_label_0 // BHI 519 // f5ffff8a
	BLS	jmp_label_0 // BLS 519 // f4ffff9a
	BGE	jmp_label_0 // BGE 519 // f3ffffaa
	BLT	jmp_label_0 // BLT 519 // f2ffffba
	BGT	jmp_label_0 // BGT 519 // f1ffffca
	BLE	jmp_label_0 // BLE 519 // f0ffffda
	B	jmp_label_0 // JMP 519 // efffffea
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
	BL.EQ	jmp_label_2 // CALL.EQ 560 // fdffff0b
	BL.NE	jmp_label_2 // CALL.NE 560 // fcffff1b
	BL.CS	jmp_label_2 // CALL.CS 560 // fbffff2b
	BL.CC	jmp_label_2 // CALL.CC 560 // faffff3b
	BL.MI	jmp_label_2 // CALL.MI 560 // f9ffff4b
	BL.PL	jmp_label_2 // CALL.PL 560 // f8ffff5b
	BL.VS	jmp_label_2 // CALL.VS 560 // f7ffff6b
	BL.VC	jmp_label_2 // CALL.VC 560 // f6ffff7b
	BL.HI	jmp_label_2 // CALL.HI 560 // f5ffff8b
	BL.LS	jmp_label_2 // CALL.LS 560 // f4ffff9b
	BL.GE	jmp_label_2 // CALL.GE 560 // f3ffffab
	BL.LT	jmp_label_2 // CALL.LT 560 // f2ffffbb
	BL.GT	jmp_label_2 // CALL.GT 560 // f1ffffcb
	BL.LE	jmp_label_2 // CALL.LE 560 // f0ffffdb
	BL	jmp_label_2 // CALL 560    // efffffeb
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
	SRL	$0, R5, R6           // 0560a0e1
	SRL	$1, R5, R6           // a560a0e1
	SRL	$14, R5, R6          // 2567a0e1
	SRL	$15, R5, R6          // a567a0e1
	SRL	$30, R5, R6          // 256fa0e1
	SRL	$31, R5, R6          // a56fa0e1
	SRL	$32, R5, R6          // 2560a0e1
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
	SRA	$0, R5, R6           // 0560a0e1
	SRA	$1, R5, R6           // c560a0e1
	SRA	$14, R5, R6          // 4567a0e1
	SRA	$15, R5, R6          // c567a0e1
	SRA	$30, R5, R6          // 456fa0e1
	SRA	$31, R5, R6          // c56fa0e1
	SRA	$32, R5, R6          // 4560a0e1
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
	SLL	$0, R5, R6           // 0560a0e1
	SLL	$1, R5, R6           // 8560a0e1
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

// Ops with zero shifts should encode as left shifts
	ADD	R0<<0, R1, R2	     // 002081e0
	ADD	R0>>0, R1, R2	     // 002081e0
	ADD	R0->0, R1, R2	     // 002081e0
	ADD	R0@>0, R1, R2	     // 002081e0
	MOVW	R0<<0(R1), R2        // 002091e7
	MOVW	R0>>0(R1), R2        // 002091e7
	MOVW	R0->0(R1), R2        // 002091e7
	MOVW	R0@>0(R1), R2        // 002091e7
	MOVW	R0, R1<<0(R2)        // 010082e7
	MOVW	R0, R1>>0(R2)        // 010082e7
	MOVW	R0, R1->0(R2)        // 010082e7
	MOVW	R0, R1@>0(R2)        // 010082e7

// MULA / MULS
	MULAWT		R1, R2, R3, R4       // c23124e1
	MULAWB		R1, R2, R3, R4       // 823124e1
	MULS		R1, R2, R3, R4       // 923164e0
	MULA		R1, R2, R3, R4       // 923124e0
	MULA.S		R1, R2, R3, R4       // 923134e0
	MMULA		R1, R2, R3, R4       // 123154e7
	MMULS		R1, R2, R3, R4       // d23154e7
	MULABB		R1, R2, R3, R4       // 823104e1
	MULAL		R1, R2, (R4, R3)     // 9231e4e0
	MULAL.S		R1, R2, (R4, R3)     // 9231f4e0
	MULALU		R1, R2, (R4, R3)     // 9231a4e0
	MULALU.S	R1, R2, (R4, R3)     // 9231b4e0

// MUL
	MUL	R2, R3, R4           // 930204e0
	MUL	R2, R4               // 940204e0
	MUL	R2, R4, R4           // 940204e0
	MUL.S	R2, R3, R4           // 930214e0
	MUL.S	R2, R4               // 940214e0
	MUL.S	R2, R4, R4           // 940214e0
	MULU	R5, R6, R7           // 960507e0
	MULU	R5, R7               // 970507e0
	MULU	R5, R7, R7           // 970507e0
	MULU.S	R5, R6, R7           // 960517e0
	MULU.S	R5, R7               // 970517e0
	MULU.S	R5, R7, R7           // 970517e0
	MULLU	R1, R2, (R4, R3)     // 923184e0
	MULLU.S	R1, R2, (R4, R3)     // 923194e0
	MULL	R1, R2, (R4, R3)     // 9231c4e0
	MULL.S	R1, R2, (R4, R3)     // 9231d4e0
	MMUL	R1, R2, R3           // 12f153e7
	MULBB	R1, R2, R3           // 820163e1
	MULWB	R1, R2, R3           // a20123e1
	MULWT	R1, R2, R3           // e20123e1

// REV
	REV	R1, R2               // 312fbfe6
	REV16	R1, R2               // b12fbfe6
	REVSH	R1, R2               // b12fffe6
	RBIT	R1, R2               // 312fffe6

// XTAB/XTAH/XTABU/XTAHU
	XTAB	R2@>0, R8            // 7280a8e6
	XTAB	R2@>8, R8            // 7284a8e6
	XTAB	R2@>16, R8           // 7288a8e6
	XTAB	R2@>24, R8           // 728ca8e6
	XTAH	R3@>0, R9            // 7390b9e6
	XTAH	R3@>8, R9            // 7394b9e6
	XTAH	R3@>16, R9           // 7398b9e6
	XTAH	R3@>24, R9           // 739cb9e6
	XTABU	R4@>0, R7            // 7470e7e6
	XTABU	R4@>8, R7            // 7474e7e6
	XTABU	R4@>16, R7           // 7478e7e6
	XTABU	R4@>24, R7           // 747ce7e6
	XTAHU	R5@>0, R1            // 7510f1e6
	XTAHU	R5@>8, R1            // 7514f1e6
	XTAHU	R5@>16, R1           // 7518f1e6
	XTAHU	R5@>24, R1           // 751cf1e6
	XTAB	R2@>0, R4, R8        // 7280a4e6
	XTAB	R2@>8, R4, R8        // 7284a4e6
	XTAB	R2@>16, R4, R8       // 7288a4e6
	XTAB	R2@>24, R4, R8       // 728ca4e6
	XTAH	R3@>0, R4, R9        // 7390b4e6
	XTAH	R3@>8, R4, R9        // 7394b4e6
	XTAH	R3@>16, R4, R9       // 7398b4e6
	XTAH	R3@>24, R4, R9       // 739cb4e6
	XTABU	R4@>0, R0, R7        // 7470e0e6
	XTABU	R4@>8, R0, R7        // 7474e0e6
	XTABU	R4@>16, R0, R7       // 7478e0e6
	XTABU	R4@>24, R0, R7       // 747ce0e6
	XTAHU	R5@>0, R9, R1        // 7510f9e6
	XTAHU	R5@>8, R9, R1        // 7514f9e6
	XTAHU	R5@>16, R9, R1       // 7518f9e6
	XTAHU	R5@>24, R9, R1       // 751cf9e6

// DIVHW R0, R1, R2: R1 / R0 -> R2
	DIVHW	R0, R1, R2           // 11f012e7
	DIVUHW	R0, R1, R2           // 11f032e7
// DIVHW R0, R1: R1 / R0 -> R1
	DIVHW	R0, R1               // 11f011e7
	DIVUHW	R0, R1               // 11f031e7

// misc
	CLZ	R1, R2         // 112f6fe1
	WORD	$0             // 00000000
	WORD	$4294967295    // ffffffff
	WORD	$2863311530    // aaaaaaaa
	WORD	$1431655765    // 55555555
	PLD	4080(R6)       // f0ffd6f5
	PLD	-4080(R9)      // f0ff59f5
	RFE	               // 0080fde8
	SWPW	R3, (R7), R9   // SWPW  (R7), R3, R9 // 939007e1
	SWPBU	R4, (R2), R8   // SWPBU (R2), R4, R8 // 948042e1
	SWI	$0             // 000000ef
	SWI	$65535         // ffff00ef
	SWI	               // 000000ef

// BFX/BFXU/BFC/BFI
	BFX	$16, $8, R1, R2 // BFX $16, R1, $8, R2   // 5124afe7
	BFX	$29, $2, R8                              // 5881bce7
	BFXU	$16, $8, R1, R2 // BFXU $16, R1, $8, R2  // 5124efe7
	BFXU	$29, $2, R8                              // 5881fce7
	BFC	$29, $2, R8                              // 1f81dee7
	BFI	$29, $2, R8                              // 1881dee7
	BFI	$16, $8, R1, R2 // BFI $16, R1, $8, R2   // 1124d7e7

// synthetic arithmetic
	ADD	$0xffffffaa, R2, R3 // ADD $4294967210, R2, R3   // 55b0e0e30b3082e0
	ADD	$0xffffff55, R5     // ADD $4294967125, R5       // aab0e0e30b5085e0
	ADD.S	$0xffffffab, R2, R3 // ADD.S $4294967211, R2, R3 // 54b0e0e30b3092e0
	ADD.S	$0xffffff54, R5     // ADD.S $4294967124, R5     // abb0e0e30b5095e0
	ADC	$0xffffffac, R2, R3 // ADC $4294967212, R2, R3   // 53b0e0e30b30a2e0
	ADC	$0xffffff53, R5     // ADC $4294967123, R5       // acb0e0e30b50a5e0
	ADC.S	$0xffffffad, R2, R3 // ADC.S $4294967213, R2, R3 // 52b0e0e30b30b2e0
	ADC.S	$0xffffff52, R5     // ADC.S $4294967122, R5     // adb0e0e30b50b5e0
	SUB	$0xffffffae, R2, R3 // SUB $4294967214, R2, R3   // 51b0e0e30b3042e0
	SUB	$0xffffff51, R5     // SUB $4294967121, R5       // aeb0e0e30b5045e0
	SUB.S	$0xffffffaf, R2, R3 // SUB.S $4294967215, R2, R3 // 50b0e0e30b3052e0
	SUB.S	$0xffffff50, R5     // SUB.S $4294967120, R5     // afb0e0e30b5055e0
	SBC	$0xffffffb0, R2, R3 // SBC $4294967216, R2, R3   // 4fb0e0e30b30c2e0
	SBC	$0xffffff4f, R5     // SBC $4294967119, R5       // b0b0e0e30b50c5e0
	SBC.S	$0xffffffb1, R2, R3 // SBC.S $4294967217, R2, R3 // 4eb0e0e30b30d2e0
	SBC.S	$0xffffff4e, R5     // SBC.S $4294967118, R5     // b1b0e0e30b50d5e0
	RSB	$0xffffffb2, R2, R3 // RSB $4294967218, R2, R3   // 4db0e0e30b3062e0
	RSB	$0xffffff4d, R5     // RSB $4294967117, R5       // b2b0e0e30b5065e0
	RSB.S	$0xffffffb3, R2, R3 // RSB.S $4294967219, R2, R3 // 4cb0e0e30b3072e0
	RSB.S	$0xffffff4c, R5     // RSB.S $4294967116, R5     // b3b0e0e30b5075e0
	RSC	$0xffffffb4, R2, R3 // RSC $4294967220, R2, R3   // 4bb0e0e30b30e2e0
	RSC	$0xffffff4b, R5     // RSC $4294967115, R5       // b4b0e0e30b50e5e0
	RSC.S	$0xffffffb5, R2, R3 // RSC.S $4294967221, R2, R3 // 4ab0e0e30b30f2e0
	RSC.S	$0xffffff4a, R5     // RSC.S $4294967114, R5     // b5b0e0e30b50f5e0
	AND	$0xffffffaa, R2, R3 // AND $4294967210, R2, R3   // 55b0e0e30b3002e0
	AND	$0xffffff55, R5     // AND $4294967125, R5       // aab0e0e30b5005e0
	AND.S	$0xffffffab, R2, R3 // AND.S $4294967211, R2, R3 // 54b0e0e30b3012e0
	AND.S	$0xffffff54, R5     // AND.S $4294967124, R5     // abb0e0e30b5015e0
	ORR	$0xffffffaa, R2, R3 // ORR $4294967210, R2, R3   // 55b0e0e30b3082e1
	ORR	$0xffffff55, R5     // ORR $4294967125, R5       // aab0e0e30b5085e1
	ORR.S	$0xffffffab, R2, R3 // ORR.S $4294967211, R2, R3 // 54b0e0e30b3092e1
	ORR.S	$0xffffff54, R5     // ORR.S $4294967124, R5     // abb0e0e30b5095e1
	EOR	$0xffffffaa, R2, R3 // EOR $4294967210, R2, R3   // 55b0e0e30b3022e0
	EOR	$0xffffff55, R5     // EOR $4294967125, R5       // aab0e0e30b5025e0
	EOR.S	$0xffffffab, R2, R3 // EOR.S $4294967211, R2, R3 // 54b0e0e30b3032e0
	EOR.S	$0xffffff54, R5     // EOR.S $4294967124, R5     // abb0e0e30b5035e0
	BIC	$0xffffffaa, R2, R3 // BIC $4294967210, R2, R3   // 55b0e0e30b30c2e1
	BIC	$0xffffff55, R5     // BIC $4294967125, R5       // aab0e0e30b50c5e1
	BIC.S	$0xffffffab, R2, R3 // BIC.S $4294967211, R2, R3 // 54b0e0e30b30d2e1
	BIC.S	$0xffffff54, R5     // BIC.S $4294967124, R5     // abb0e0e30b50d5e1
	CMP	$0xffffffab, R2     // CMP $4294967211, R2       // 54b0e0e30b0052e1
	CMN	$0xffffffac, R3     // CMN $4294967212, R3       // 53b0e0e30b0073e1
	TST	$0xffffffad, R4     // TST $4294967213, R4       // 52b0e0e30b0014e1
	TEQ	$0xffffffae, R5     // TEQ $4294967214, R5       // 51b0e0e30b0035e1

// immediate decomposition
	ADD	$0xff0000ff, R0, R1 // ADD $4278190335, R0, R1 // ff1080e2ff1481e2
	EOR	$0xff0000ff, R0, R1 // EOR $4278190335, R0, R1 // ff1020e2ff1421e2
	ORR	$0xff0000ff, R0, R1 // ORR $4278190335, R0, R1 // ff1080e3ff1481e3
	SUB	$0xff0000ff, R0, R1 // SUB $4278190335, R0, R1 // ff1040e2ff1441e2
	BIC	$0xff0000ff, R0, R1 // BIC $4278190335, R0, R1 // ff10c0e3ff14c1e3
	RSB	$0xff0000ff, R0, R1 // RSB $4278190335, R0, R1 // ff1060e2ff1481e2
	ADC	$0xff0000ff, R0, R1 // ADC $4278190335, R0, R1 // ff10a0e2ff1481e2
	SBC	$0xff0000ff, R0, R1 // SBC $4278190335, R0, R1 // ff10c0e2ff1441e2
	RSC	$0xff0000ff, R0, R1 // RSC $4278190335, R0, R1 // ff10e0e2ff1481e2
	ADD	$0x000fffff, R0, R1 // ADD $1048575, R0, R1    // 011680e2011041e2
	ADC	$0x000fffff, R0, R1 // ADC $1048575, R0, R1    // 0116a0e2011041e2
	SUB	$0x000fffff, R0, R1 // SUB $1048575, R0, R1    // 011640e2011081e2
	SBC	$0x000fffff, R0, R1 // SBC $1048575, R0, R1    // 0116c0e2011081e2
	RSB	$0x000fffff, R0, R1 // RSB $1048575, R0, R1    // 011660e2011041e2
	RSC	$0x000fffff, R0, R1 // RSC $1048575, R0, R1    // 0116e0e2011041e2
	ADD	$0xff0000ff, R1     // ADD $4278190335, R1     // ff1081e2ff1481e2
	EOR	$0xff0000ff, R1     // EOR $4278190335, R1     // ff1021e2ff1421e2
	ORR	$0xff0000ff, R1     // ORR $4278190335, R1     // ff1081e3ff1481e3
	SUB	$0xff0000ff, R1     // SUB $4278190335, R1     // ff1041e2ff1441e2
	BIC	$0xff0000ff, R1     // BIC $4278190335, R1     // ff10c1e3ff14c1e3
	RSB	$0xff0000ff, R1     // RSB $4278190335, R1     // ff1061e2ff1481e2
	ADC	$0xff0000ff, R1     // ADC $4278190335, R1     // ff10a1e2ff1481e2
	SBC	$0xff0000ff, R1     // SBC $4278190335, R1     // ff10c1e2ff1441e2
	RSC	$0xff0000ff, R1     // RSC $4278190335, R1     // ff10e1e2ff1481e2
	ADD	$0x000fffff, R1     // ADD $1048575, R1        // 011681e2011041e2
	ADC	$0x000fffff, R1     // ADC $1048575, R1        // 0116a1e2011041e2
	SUB	$0x000fffff, R1     // SUB $1048575, R1        // 011641e2011081e2
	SBC	$0x000fffff, R1     // SBC $1048575, R1        // 0116c1e2011081e2
	RSB	$0x000fffff, R1     // RSB $1048575, R1        // 011661e2011041e2
	RSC	$0x000fffff, R1     // RSC $1048575, R1        // 0116e1e2011041e2

// MVN
	MVN	$0xff, R1        // MVN $255, R1          // ff10e0e3
	MVN	$0xff000000, R1  // MVN $4278190080, R1   // ff14e0e3
	MVN	R9<<30, R7       // 097fe0e1
	MVN	R9>>30, R7       // 297fe0e1
	MVN	R9->30, R7       // 497fe0e1
	MVN	R9@>30, R7       // 697fe0e1
	MVN.S	R9<<30, R7       // 097ff0e1
	MVN.S	R9>>30, R7       // 297ff0e1
	MVN.S	R9->30, R7       // 497ff0e1
	MVN.S	R9@>30, R7       // 697ff0e1
	MVN	R9<<R8, R7       // 1978e0e1
	MVN	R9>>R8, R7       // 3978e0e1
	MVN	R9->R8, R7       // 5978e0e1
	MVN	R9@>R8, R7       // 7978e0e1
	MVN.S	R9<<R8, R7       // 1978f0e1
	MVN.S	R9>>R8, R7       // 3978f0e1
	MVN.S	R9->R8, R7       // 5978f0e1
	MVN.S	R9@>R8, R7       // 7978f0e1
	MVN	$0xffffffbe, R5  // MVN $4294967230, R5   // 4150a0e3

// MOVM
	MOVM.IA   [R0,R2,R4,R6], (R1)        // MOVM.U [R0,R2,R4,R6], (R1)                      // 550081e8
	MOVM.IA   [R0-R4,R6,R8,R9-R11], (R1) // MOVM.U [R0,R1,R2,R3,R4,R6,R8,R9,g,R11], (R1)    // 5f0f81e8
	MOVM.IA.W [R0,R2,R4,R6], (R1)        // MOVM.W.U [R0,R2,R4,R6], (R1)                    // 5500a1e8
	MOVM.IA.W [R0-R4,R6,R8,R9-R11], (R1) // MOVM.W.U [R0,R1,R2,R3,R4,R6,R8,R9,g,R11], (R1)  // 5f0fa1e8
	MOVM.IA   (R1), [R0,R2,R4,R6]        // MOVM.U (R1), [R0,R2,R4,R6]                      // 550091e8
	MOVM.IA   (R1), [R0-R4,R6,R8,R9-R11] // MOVM.U (R1), [R0,R1,R2,R3,R4,R6,R8,R9,g,R11]    // 5f0f91e8
	MOVM.IA.W (R1), [R0,R2,R4,R6]        // MOVM.W.U (R1), [R0,R2,R4,R6]                    // 5500b1e8
	MOVM.IA.W (R1), [R0-R4,R6,R8,R9-R11] // MOVM.W.U (R1), [R0,R1,R2,R3,R4,R6,R8,R9,g,R11]  // 5f0fb1e8
	MOVM.DA   [R0,R2,R4,R6], (R1)        // MOVM [R0,R2,R4,R6], (R1)                        // 550001e8
	MOVM.DA   [R0-R4,R6,R8,R9-R11], (R1) // MOVM [R0,R1,R2,R3,R4,R6,R8,R9,g,R11], (R1)      // 5f0f01e8
	MOVM.DA.W [R0,R2,R4,R6], (R1)        // MOVM.W [R0,R2,R4,R6], (R1)                      // 550021e8
	MOVM.DA.W [R0-R4,R6,R8,R9-R11], (R1) // MOVM.W [R0,R1,R2,R3,R4,R6,R8,R9,g,R11], (R1)    // 5f0f21e8
	MOVM.DA   (R1), [R0,R2,R4,R6]        // MOVM (R1), [R0,R2,R4,R6]                        // 550011e8
	MOVM.DA   (R1), [R0-R4,R6,R8,R9-R11] // MOVM (R1), [R0,R1,R2,R3,R4,R6,R8,R9,g,R11]      // 5f0f11e8
	MOVM.DA.W (R1), [R0,R2,R4,R6]        // MOVM.W (R1), [R0,R2,R4,R6]                      // 550031e8
	MOVM.DA.W (R1), [R0-R4,R6,R8,R9-R11] // MOVM.W (R1), [R0,R1,R2,R3,R4,R6,R8,R9,g,R11]    // 5f0f31e8
	MOVM.DB   [R0,R2,R4,R6], (R1)        // MOVM.P [R0,R2,R4,R6], (R1)                      // 550001e9
	MOVM.DB   [R0-R4,R6,R8,R9-R11], (R1) // MOVM.P [R0,R1,R2,R3,R4,R6,R8,R9,g,R11], (R1)    // 5f0f01e9
	MOVM.DB.W [R0,R2,R4,R6], (R1)        // MOVM.P.W [R0,R2,R4,R6], (R1)                    // 550021e9
	MOVM.DB.W [R0-R4,R6,R8,R9-R11], (R1) // MOVM.P.W [R0,R1,R2,R3,R4,R6,R8,R9,g,R11], (R1)  // 5f0f21e9
	MOVM.DB   (R1), [R0,R2,R4,R6]        // MOVM.P (R1), [R0,R2,R4,R6]                      // 550011e9
	MOVM.DB   (R1), [R0-R4,R6,R8,R9-R11] // MOVM.P (R1), [R0,R1,R2,R3,R4,R6,R8,R9,g,R11]    // 5f0f11e9
	MOVM.DB.W (R1), [R0,R2,R4,R6]        // MOVM.P.W (R1), [R0,R2,R4,R6]                    // 550031e9
	MOVM.DB.W (R1), [R0-R4,R6,R8,R9-R11] // MOVM.P.W (R1), [R0,R1,R2,R3,R4,R6,R8,R9,g,R11]  // 5f0f31e9
	MOVM.IB   [R0,R2,R4,R6], (g)         // MOVM.P.U [R0,R2,R4,R6], (g)                     // 55008ae9
	MOVM.IB   [R0-R4,R6,R8,R9-R11], (g)  // MOVM.P.U [R0,R1,R2,R3,R4,R6,R8,R9,g,R11], (g)   // 5f0f8ae9
	MOVM.IB.W [R0,R2,R4,R6], (g)         // MOVM.P.W.U [R0,R2,R4,R6], (g)                   // 5500aae9
	MOVM.IB.W [R0-R4,R6,R8,R9-R11], (g)  // MOVM.P.W.U [R0,R1,R2,R3,R4,R6,R8,R9,g,R11], (g) // 5f0faae9
	MOVM.IB   (g), [R0,R2,R4,R6]         // MOVM.P.U (g), [R0,R2,R4,R6]                     // 55009ae9
	MOVM.IB   (g), [R0-R4,R6,R8,R9-R11]  // MOVM.P.U (g), [R0,R1,R2,R3,R4,R6,R8,R9,g,R11]   // 5f0f9ae9
	MOVM.IB.W (g), [R0,R2,R4,R6]         // MOVM.P.W.U (g), [R0,R2,R4,R6]                   // 5500bae9
	MOVM.IB.W (g), [R0-R4,R6,R8,R9-R11]  // MOVM.P.W.U (g), [R0,R1,R2,R3,R4,R6,R8,R9,g,R11] // 5f0fbae9

// MOVW
	MOVW	R3, R4                                            // 0340a0e1
	MOVW.S	R3, R4                                            // 0340b0e1
	MOVW	R9, R2                                            // 0920a0e1
	MOVW.S	R9, R2                                            // 0920b0e1
	MOVW	R5>>1, R2                                         // a520a0e1
	MOVW.S	R5>>1, R2                                         // a520b0e1
	MOVW	R5<<1, R2                                         // 8520a0e1
	MOVW.S	R5<<1, R2                                         // 8520b0e1
	MOVW	R5->1, R2                                         // c520a0e1
	MOVW.S	R5->1, R2                                         // c520b0e1
	MOVW	R5@>1, R2                                         // e520a0e1
	MOVW.S	R5@>1, R2                                         // e520b0e1
	MOVW	$0xff, R9            // MOVW $255, R9             // ff90a0e3
	MOVW	$0xff000000, R9      // MOVW $4278190080, R9      // ff94a0e3
	MOVW	$0xff(R0), R1        // MOVW $255(R0), R1         // ff1080e2
	MOVW.S	$0xff(R0), R1        // MOVW.S $255(R0), R1       // ff1090e2
	MOVW	$-0xff(R0), R1       // MOVW $-255(R0), R1        // ff1040e2
	MOVW.S	$-0xff(R0), R1       // MOVW.S $-255(R0), R1      // ff1050e2
	MOVW	$0xffffffae, R1      // MOVW $4294967214, R1      // 5110e0e3
	MOVW	$0xaaaaaaaa, R1      // MOVW $2863311530, R1
	MOVW	R1, (R2)                                          // 001082e5
	MOVW.P	R1, (R2)                                          // 001082e4
	MOVW.W	R1, (R2)                                          // 0010a2e5
	MOVW	R1, 0x20(R2)         // MOVW R1, 32(R2)           // 201082e5
	MOVW.P	R1, 0x20(R2)         // MOVW.P R1, 32(R2)         // 201082e4
	MOVW.W	R1, 0x20(R2)         // MOVW.W R1, 32(R2)         // 2010a2e5
	MOVW	R1, -0x20(R2)        // MOVW R1, -32(R2)          // 201002e5
	MOVW.P	R1, -0x20(R2)        // MOVW.P R1, -32(R2)        // 201002e4
	MOVW.W	R1, -0x20(R2)        // MOVW.W R1, -32(R2)        // 201022e5
	MOVW	(R2), R1                                          // 001092e5
	MOVW.P	(R2), R1                                          // 001092e4
	MOVW.W	(R2), R1                                          // 0010b2e5
	MOVW	0x20(R2), R1         // MOVW 32(R2), R1           // 201092e5
	MOVW.P	0x20(R2), R1         // MOVW.P 32(R2), R1         // 201092e4
	MOVW.W	0x20(R2), R1         // MOVW.W 32(R2), R1         // 2010b2e5
	MOVW	-0x20(R2), R1        // MOVW -32(R2), R1          // 201012e5
	MOVW.P	-0x20(R2), R1        // MOVW.P -32(R2), R1        // 201012e4
	MOVW.W	-0x20(R2), R1        // MOVW.W -32(R2), R1        // 201032e5
	MOVW	R1, 0x00ffffff(R2)   // MOVW R1, 16777215(R2)
	MOVW	0x00ffffff(R2), R1   // MOVW 16777215(R2), R1
	MOVW	CPSR, R1                                          // 00100fe1
	MOVW	R1, CPSR                                          // 01f02ce1
	MOVW	$0xff, CPSR          // MOVW $255, CPSR           // fff02ce3
	MOVW	$0xff000000, CPSR    // MOVW $4278190080, CPSR    // fff42ce3
	MOVW	FPSR, R9                                          // 109af1ee
	MOVW	FPSR, g                                           // 10aaf1ee
	MOVW	R9, FPSR                                          // 109ae1ee
	MOVW	g, FPSR                                           // 10aae1ee
	MOVW	R0>>28(R1), R2                                    // 202e91e7
	MOVW	R0<<28(R1), R2                                    // 002e91e7
	MOVW	R0->28(R1), R2                                    // 402e91e7
	MOVW	R0@>28(R1), R2                                    // 602e91e7
	MOVW.U	R0>>28(R1), R2                                    // 202e11e7
	MOVW.U	R0<<28(R1), R2                                    // 002e11e7
	MOVW.U	R0->28(R1), R2                                    // 402e11e7
	MOVW.U	R0@>28(R1), R2                                    // 602e11e7
	MOVW.W	R0>>28(R1), R2                                    // 202eb1e7
	MOVW.W	R0<<28(R1), R2                                    // 002eb1e7
	MOVW.W	R0->28(R1), R2                                    // 402eb1e7
	MOVW.W	R0@>28(R1), R2                                    // 602eb1e7
	MOVW.P	R0>>28(g), R2                                     // 202e9ae6
	MOVW.P	R0<<28(g), R2                                     // 002e9ae6
	MOVW.P	R0->28(g), R2                                     // 402e9ae6
	MOVW.P	R0@>28(g), R2                                     // 602e9ae6
	MOVW	R2, R0>>28(R1)                                    // 202e81e7
	MOVW	R2, R0<<28(R1)                                    // 002e81e7
	MOVW	R2, R0->28(R1)                                    // 402e81e7
	MOVW	R2, R0@>28(R1)                                    // 602e81e7
	MOVW.U	R2, R0>>28(R1)                                    // 202e01e7
	MOVW.U	R2, R0<<28(R1)                                    // 002e01e7
	MOVW.U	R2, R0->28(R1)                                    // 402e01e7
	MOVW.U	R2, R0@>28(R1)                                    // 602e01e7
	MOVW.W	R2, R0>>28(R1)                                    // 202ea1e7
	MOVW.W	R2, R0<<28(R1)                                    // 002ea1e7
	MOVW.W	R2, R0->28(R1)                                    // 402ea1e7
	MOVW.W	R2, R0@>28(R1)                                    // 602ea1e7
	MOVW.P	R2, R0>>28(R5)                                    // 202e85e6
	MOVW.P	R2, R0<<28(R5)                                    // 002e85e6
	MOVW.P	R2, R0->28(R5)                                    // 402e85e6
	MOVW.P	R2, R0@>28(R5)                                    // 602e85e6
	MOVW	R0, math·Exp(SB)     // MOVW R0, math.Exp(SB)
	MOVW	math·Exp(SB), R0     // MOVW math.Exp(SB), R0

// MOVB
	MOVB	R3, R4                                            // 0340a0e1
	MOVB	R9, R2                                            // 0920a0e1
	MOVBU	R0, R1                                            // ff1000e2
	MOVBS	R5, R6                                            // 056ca0e1466ca0e1
	MOVB	R1, (R2)                                          // 0010c2e5
	MOVB.P	R1, (R2)                                          // 0010c2e4
	MOVB.W	R1, (R2)                                          // 0010e2e5
	MOVB	R1, 0x20(R2)         // MOVB R1, 32(R2)           // 2010c2e5
	MOVB.P	R1, 0x20(R2)         // MOVB.P R1, 32(R2)         // 2010c2e4
	MOVB.W	R1, 0x20(R2)         // MOVB.W R1, 32(R2)         // 2010e2e5
	MOVB	R1, -0x20(R2)        // MOVB R1, -32(R2)          // 201042e5
	MOVB.P	R1, -0x20(R2)        // MOVB.P R1, -32(R2)        // 201042e4
	MOVB.W	R1, -0x20(R2)        // MOVB.W R1, -32(R2)        // 201062e5
	MOVBS	R1, (R2)                                          // 0010c2e5
	MOVBS.P	R1, (R2)                                          // 0010c2e4
	MOVBS.W	R1, (R2)                                          // 0010e2e5
	MOVBS	R1, 0x20(R2)         // MOVBS R1, 32(R2)          // 2010c2e5
	MOVBS.P	R1, 0x20(R2)         // MOVBS.P R1, 32(R2)        // 2010c2e4
	MOVBS.W	R1, 0x20(R2)         // MOVBS.W R1, 32(R2)        // 2010e2e5
	MOVBS	R1, -0x20(R2)        // MOVBS R1, -32(R2)         // 201042e5
	MOVBS.P	R1, -0x20(R2)        // MOVBS.P R1, -32(R2)       // 201042e4
	MOVBS.W	R1, -0x20(R2)        // MOVBS.W R1, -32(R2)       // 201062e5
	MOVBU	R1, (R2)                                          // 0010c2e5
	MOVBU.P	R1, (R2)                                          // 0010c2e4
	MOVBU.W	R1, (R2)                                          // 0010e2e5
	MOVBU	R1, 0x20(R2)         // MOVBU R1, 32(R2)          // 2010c2e5
	MOVBU.P	R1, 0x20(R2)         // MOVBU.P R1, 32(R2)        // 2010c2e4
	MOVBU.W	R1, 0x20(R2)         // MOVBU.W R1, 32(R2)        // 2010e2e5
	MOVBU	R1, -0x20(R2)        // MOVBU R1, -32(R2)         // 201042e5
	MOVBU.P	R1, -0x20(R2)        // MOVBU.P R1, -32(R2)       // 201042e4
	MOVBU.W	R1, -0x20(R2)        // MOVBU.W R1, -32(R2)       // 201062e5
	MOVB	(R2), R1                                          // d010d2e1
	MOVB.P	(R2), R1                                          // d010d2e0
	MOVB.W	(R2), R1                                          // d010f2e1
	MOVB	0x20(R2), R1         // MOVB 32(R2), R1           // d012d2e1
	MOVB.P	0x20(R2), R1         // MOVB.P 32(R2), R1         // d012d2e0
	MOVB.W	0x20(R2), R1         // MOVB.W 32(R2), R1         // d012f2e1
	MOVB	-0x20(R2), R1        // MOVB -32(R2), R1          // d01252e1
	MOVB.P	-0x20(R2), R1        // MOVB.P -32(R2), R1        // d01252e0
	MOVB.W	-0x20(R2), R1        // MOVB.W -32(R2), R1        // d01272e1
	MOVBS	(R2), R1                                          // d010d2e1
	MOVBS.P	(R2), R1                                          // d010d2e0
	MOVBS.W	(R2), R1                                          // d010f2e1
	MOVBS	0x20(R2), R1         // MOVBS 32(R2), R1          // d012d2e1
	MOVBS.P	0x20(R2), R1         // MOVBS.P 32(R2), R1        // d012d2e0
	MOVBS.W	0x20(R2), R1         // MOVBS.W 32(R2), R1        // d012f2e1
	MOVBS	-0x20(R2), R1        // MOVBS -32(R2), R1         // d01252e1
	MOVBS.P	-0x20(R2), R1        // MOVBS.P -32(R2), R1       // d01252e0
	MOVBS.W	-0x20(R2), R1        // MOVBS.W -32(R2), R1       // d01272e1
	MOVBU	(R2), R1                                          // 0010d2e5
	MOVBU.P	(R2), R1                                          // 0010d2e4
	MOVBU.W	(R2), R1                                          // 0010f2e5
	MOVBU	0x20(R2), R1         // MOVBU 32(R2), R1          // 2010d2e5
	MOVBU.P	0x20(R2), R1         // MOVBU.P 32(R2), R1        // 2010d2e4
	MOVBU.W	0x20(R2), R1         // MOVBU.W 32(R2), R1        // 2010f2e5
	MOVBU	-0x20(R2), R1        // MOVBU -32(R2), R1         // 201052e5
	MOVBU.P	-0x20(R2), R1        // MOVBU.P -32(R2), R1       // 201052e4
	MOVBU.W	-0x20(R2), R1        // MOVBU.W -32(R2), R1       // 201072e5
	MOVB	R1, 0x00ffffff(R2)   // MOVB R1, 16777215(R2)
	MOVB.W	R1, 0x00ffffff(R2)   // MOVB.W R1, 16777215(R2)
	MOVB.P	R1, 0x00ffffff(R2)   // MOVB.P R1, 16777215(R2)
	MOVB	R1, -0x00ffffff(R2)  // MOVB R1, -16777215(R2)
	MOVB.W	R1, -0x00ffffff(R2)  // MOVB.W R1, -16777215(R2)
	MOVB.P	R1, -0x00ffffff(R2)  // MOVB.P R1, -16777215(R2)
	MOVB	0x00ffffff(R2), R1   // MOVB 16777215(R2), R1
	MOVB.P	0x00ffffff(R2), R1   // MOVB.P 16777215(R2), R1
	MOVB.W	0x00ffffff(R2), R1   // MOVB.W 16777215(R2), R1
	MOVB	-0x00ffffff(R2), R1  // MOVB -16777215(R2), R1
	MOVB.P	-0x00ffffff(R2), R1  // MOVB.P -16777215(R2), R1
	MOVB.W	-0x00ffffff(R2), R1  // MOVB.W -16777215(R2), R1
	MOVBS	R1, 0x00ffffff(R2)   // MOVBS R1, 16777215(R2)
	MOVBS.W	R1, 0x00ffffff(R2)   // MOVBS.W R1, 16777215(R2)
	MOVBS.P	R1, 0x00ffffff(R2)   // MOVBS.P R1, 16777215(R2)
	MOVBS	R1, -0x00ffffff(R2)  // MOVBS R1, -16777215(R2)
	MOVBS.W	R1, -0x00ffffff(R2)  // MOVBS.W R1, -16777215(R2)
	MOVBS.P	R1, -0x00ffffff(R2)  // MOVBS.P R1, -16777215(R2)
	MOVBS	0x00ffffff(R2), R1   // MOVBS 16777215(R2), R1
	MOVBS.P	0x00ffffff(R2), R1   // MOVBS.P 16777215(R2), R1
	MOVBS.W	0x00ffffff(R2), R1   // MOVBS.W 16777215(R2), R1
	MOVBS	-0x00ffffff(R2), R1  // MOVBS -16777215(R2), R1
	MOVBS.P	-0x00ffffff(R2), R1  // MOVBS.P -16777215(R2), R1
	MOVBS.W	-0x00ffffff(R2), R1  // MOVBS.W -16777215(R2), R1
	MOVBU	R1, 0x00ffffff(R2)   // MOVBU R1, 16777215(R2)
	MOVBU.W	R1, 0x00ffffff(R2)   // MOVBU.W R1, 16777215(R2)
	MOVBU.P	R1, 0x00ffffff(R2)   // MOVBU.P R1, 16777215(R2)
	MOVBU	R1, -0x00ffffff(R2)  // MOVBU R1, -16777215(R2)
	MOVBU.W	R1, -0x00ffffff(R2)  // MOVBU.W R1, -16777215(R2)
	MOVBU.P	R1, -0x00ffffff(R2)  // MOVBU.P R1, -16777215(R2)
	MOVBU	0x00ffffff(R2), R1   // MOVBU 16777215(R2), R1
	MOVBU.P	0x00ffffff(R2), R1   // MOVBU.P 16777215(R2), R1
	MOVBU.W	0x00ffffff(R2), R1   // MOVBU.W 16777215(R2), R1
	MOVBU	-0x00ffffff(R2), R1  // MOVBU -16777215(R2), R1
	MOVBU.P	-0x00ffffff(R2), R1  // MOVBU.P -16777215(R2), R1
	MOVBU.W	-0x00ffffff(R2), R1  // MOVBU.W -16777215(R2), R1
	MOVB	R0, math·Exp(SB)     // MOVB R0, math.Exp(SB)
	MOVB	math·Exp(SB), R0     // MOVB math.Exp(SB), R0
	MOVBS	R0, math·Exp(SB)     // MOVBS R0, math.Exp(SB)
	MOVBS	math·Exp(SB), R0     // MOVBS math.Exp(SB), R0
	MOVBU	R0, math·Exp(SB)     // MOVBU R0, math.Exp(SB)
	MOVBU	math·Exp(SB), R0     // MOVBU math.Exp(SB), R0
	MOVB	R2, R0>>28(R1)                                    // 202ec1e7
	MOVB	R2, R0<<28(R1)                                    // 002ec1e7
	MOVB	R2, R0->28(R1)                                    // 402ec1e7
	MOVB	R2, R0@>28(R1)                                    // 602ec1e7
	MOVB.U	R2, R0>>28(R1)                                    // 202e41e7
	MOVB.U	R2, R0<<28(R1)                                    // 002e41e7
	MOVB.U	R2, R0->28(R1)                                    // 402e41e7
	MOVB.U	R2, R0@>28(R1)                                    // 602e41e7
	MOVB.W	R2, R0>>28(R1)                                    // 202ee1e7
	MOVB.W	R2, R0<<28(R1)                                    // 002ee1e7
	MOVB.W	R2, R0->28(R1)                                    // 402ee1e7
	MOVB.W	R2, R0@>28(R1)                                    // 602ee1e7
	MOVB.P	R2, R0>>28(R5)                                    // 202ec5e6
	MOVB.P	R2, R0<<28(R5)                                    // 002ec5e6
	MOVB.P	R2, R0->28(R5)                                    // 402ec5e6
	MOVB.P	R2, R0@>28(R5)                                    // 602ec5e6
	MOVBS	R2, R0>>28(R1)                                    // 202ec1e7
	MOVBS	R2, R0<<28(R1)                                    // 002ec1e7
	MOVBS	R2, R0->28(R1)                                    // 402ec1e7
	MOVBS	R2, R0@>28(R1)                                    // 602ec1e7
	MOVBS.U	R2, R0>>28(R1)                                    // 202e41e7
	MOVBS.U	R2, R0<<28(R1)                                    // 002e41e7
	MOVBS.U	R2, R0->28(R1)                                    // 402e41e7
	MOVBS.U	R2, R0@>28(R1)                                    // 602e41e7
	MOVBS.W	R2, R0>>28(R1)                                    // 202ee1e7
	MOVBS.W	R2, R0<<28(R1)                                    // 002ee1e7
	MOVBS.W	R2, R0->28(R1)                                    // 402ee1e7
	MOVBS.W	R2, R0@>28(R1)                                    // 602ee1e7
	MOVBS.P	R2, R0>>28(R5)                                    // 202ec5e6
	MOVBS.P	R2, R0<<28(R5)                                    // 002ec5e6
	MOVBS.P	R2, R0->28(R5)                                    // 402ec5e6
	MOVBS.P	R2, R0@>28(R5)                                    // 602ec5e6
	MOVBU	R2, R0>>28(R1)                                    // 202ec1e7
	MOVBU	R2, R0<<28(R1)                                    // 002ec1e7
	MOVBU	R2, R0->28(R1)                                    // 402ec1e7
	MOVBU	R2, R0@>28(R1)                                    // 602ec1e7
	MOVBU.U	R2, R0>>28(R1)                                    // 202e41e7
	MOVBU.U	R2, R0<<28(R1)                                    // 002e41e7
	MOVBU.U	R2, R0->28(R1)                                    // 402e41e7
	MOVBU.U	R2, R0@>28(R1)                                    // 602e41e7
	MOVBU.W	R2, R0>>28(R1)                                    // 202ee1e7
	MOVBU.W	R2, R0<<28(R1)                                    // 002ee1e7
	MOVBU.W	R2, R0->28(R1)                                    // 402ee1e7
	MOVBU.W	R2, R0@>28(R1)                                    // 602ee1e7
	MOVBU.P	R2, R0>>28(R5)                                    // 202ec5e6
	MOVBU.P	R2, R0<<28(R5)                                    // 002ec5e6
	MOVBU.P	R2, R0->28(R5)                                    // 402ec5e6
	MOVBU.P	R2, R0@>28(R5)                                    // 602ec5e6
	MOVBU	R0>>28(R1), R2                                    // 202ed1e7
	MOVBU	R0<<28(R1), R2                                    // 002ed1e7
	MOVBU	R0->28(R1), R2                                    // 402ed1e7
	MOVBU	R0@>28(R1), R2                                    // 602ed1e7
	MOVBU.U	R0>>28(R1), R2                                    // 202e51e7
	MOVBU.U	R0<<28(R1), R2                                    // 002e51e7
	MOVBU.U	R0->28(R1), R2                                    // 402e51e7
	MOVBU.U	R0@>28(R1), R2                                    // 602e51e7
	MOVBU.W	R0>>28(R1), R2                                    // 202ef1e7
	MOVBU.W	R0<<28(R1), R2                                    // 002ef1e7
	MOVBU.W	R0->28(R1), R2                                    // 402ef1e7
	MOVBU.W	R0@>28(R1), R2                                    // 602ef1e7
	MOVBU.P	R0>>28(g), R2                                     // 202edae6
	MOVBU.P	R0<<28(g), R2                                     // 002edae6
	MOVBU.P	R0->28(g), R2                                     // 402edae6
	MOVBU.P	R0@>28(g), R2                                     // 602edae6
	MOVBS	R0<<0(R1), R2                                     // d02091e1
	MOVBS.U	R0<<0(R1), R2                                     // d02011e1
	MOVBS.W	R0<<0(R1), R2                                     // d020b1e1
	MOVBS.P	R0<<0(R1), R2                                     // d02091e0
	MOVB	R0<<0(R1), R2                                     // d02091e1
	MOVB.U	R0<<0(R1), R2                                     // d02011e1
	MOVB.W	R0<<0(R1), R2                                     // d020b1e1
	MOVB.P	R0<<0(R1), R2                                     // d02091e0
	MOVBS	R2@>0, R8                                         // 7280afe6
	MOVBS	R2@>8, R8                                         // 7284afe6
	MOVBS	R2@>16, R8                                        // 7288afe6
	MOVBS	R2@>24, R8                                        // 728cafe6
	MOVB	R2@>0, R8                                         // 7280afe6
	MOVB	R2@>8, R8                                         // 7284afe6
	MOVB	R2@>16, R8                                        // 7288afe6
	MOVB	R2@>24, R8                                        // 728cafe6
	MOVBU	R4@>0, R7                                         // 7470efe6
	MOVBU	R4@>8, R7                                         // 7474efe6
	MOVBU	R4@>16, R7                                        // 7478efe6
	MOVBU	R4@>24, R7                                        // 747cefe6

// MOVH
	MOVH	R3, R4                                            // 0340a0e1
	MOVH	R9, R2                                            // 0920a0e1
	MOVHS	R5, R6                                            // 0568a0e14668a0e1
	MOVHU	R5, R6                                            // 0568a0e12668a0e1
	MOVH	R4, (R3)                                          // b040c3e1
	MOVHS.W	R4, (R3)                                          // b040e3e1
	MOVHS.P	R4, (R3)                                          // b040c3e0
	MOVHS	R4, (R3)                                          // b040c3e1
	MOVHS.W	R4, (R3)                                          // b040e3e1
	MOVHS.P	R4, (R3)                                          // b040c3e0
	MOVHU	R4, (R3)                                          // b040c3e1
	MOVHU.W	R4, (R3)                                          // b040e3e1
	MOVHU.P	R4, (R3)                                          // b040c3e0
	MOVH	R3, 0x20(R4)         // MOVH R3, 32(R4)           // b032c4e1
	MOVH.W	R3, 0x20(R4)         // MOVH.W R3, 32(R4)         // b032e4e1
	MOVH.P	R3, 0x20(R4)         // MOVH.P R3, 32(R4)         // b032c4e0
	MOVHS	R3, 0x20(R4)         // MOVHS R3, 32(R4)          // b032c4e1
	MOVHS.W	R3, 0x20(R4)         // MOVHS.W R3, 32(R4)        // b032e4e1
	MOVHS.P	R3, 0x20(R4)         // MOVHS.P R3, 32(R4)        // b032c4e0
	MOVHU	R3, 0x20(R4)         // MOVHU R3, 32(R4)          // b032c4e1
	MOVHU.W	R3, 0x20(R4)         // MOVHU.W R3, 32(R4)        // b032e4e1
	MOVHU.P	R3, 0x20(R4)         // MOVHU.P R3, 32(R4)        // b032c4e0
	MOVH	R3, -0x20(R4)        // MOVH R3, -32(R4)          // b03244e1
	MOVH.W	R3, -0x20(R4)        // MOVH.W R3, -32(R4)        // b03264e1
	MOVH.P	R3, -0x20(R4)        // MOVH.P R3, -32(R4)        // b03244e0
	MOVHS	R3, -0x20(R4)        // MOVHS R3, -32(R4)         // b03244e1
	MOVHS.W	R3, -0x20(R4)        // MOVHS.W R3, -32(R4)       // b03264e1
	MOVHS.P	R3, -0x20(R4)        // MOVHS.P R3, -32(R4)       // b03244e0
	MOVHU	R3, -0x20(R4)        // MOVHU R3, -32(R4)         // b03244e1
	MOVHU.W	R3, -0x20(R4)        // MOVHU.W R3, -32(R4)       // b03264e1
	MOVHU.P	R3, -0x20(R4)        // MOVHU.P R3, -32(R4)       // b03244e0
	MOVHU	(R9), R8                                          // b080d9e1
	MOVHU.W	(R9), R8                                          // b080f9e1
	MOVHU.P	(R9), R8                                          // b080d9e0
	MOVH	(R9), R8                                          // f080d9e1
	MOVH.W	(R9), R8                                          // f080f9e1
	MOVH.P	(R9), R8                                          // f080d9e0
	MOVHS	(R9), R8                                          // f080d9e1
	MOVHS.W	(R9), R8                                          // f080f9e1
	MOVHS.P	(R9), R8                                          // f080d9e0
	MOVHU	0x22(R9), R8         // MOVHU 34(R9), R8          // b282d9e1
	MOVHU.W	0x22(R9), R8         // MOVHU.W 34(R9), R8        // b282f9e1
	MOVHU.P	0x22(R9), R8         // MOVHU.P 34(R9), R8        // b282d9e0
	MOVH	0x22(R9), R8         // MOVH 34(R9), R8           // f282d9e1
	MOVH.W	0x22(R9), R8         // MOVH.W 34(R9), R8         // f282f9e1
	MOVH.P	0x22(R9), R8         // MOVH.P 34(R9), R8         // f282d9e0
	MOVHS	0x22(R9), R8         // MOVHS 34(R9), R8          // f282d9e1
	MOVHS.W	0x22(R9), R8         // MOVHS.W 34(R9), R8        // f282f9e1
	MOVHS.P	0x22(R9), R8         // MOVHS.P 34(R9), R8        // f282d9e0
	MOVHU	-0x24(R9), R8        // MOVHU -36(R9), R8         // b48259e1
	MOVHU.W	-0x24(R9), R8        // MOVHU.W -36(R9), R8       // b48279e1
	MOVHU.P	-0x24(R9), R8        // MOVHU.P -36(R9), R8       // b48259e0
	MOVH	-0x24(R9), R8        // MOVH -36(R9), R8          // f48259e1
	MOVH.W	-0x24(R9), R8        // MOVH.W -36(R9), R8        // f48279e1
	MOVH.P	-0x24(R9), R8        // MOVH.P -36(R9), R8        // f48259e0
	MOVHS	-0x24(R9), R8        // MOVHS -36(R9), R8         // f48259e1
	MOVHS.W	-0x24(R9), R8        // MOVHS.W -36(R9), R8       // f48279e1
	MOVHS.P	-0x24(R9), R8        // MOVHS.P -36(R9), R8       // f48259e0
	MOVH	R1, 0x00ffffff(R2)   // MOVH R1, 16777215(R2)
	MOVH.W	R1, 0x00ffffff(R2)   // MOVH.W R1, 16777215(R2)
	MOVH.P	R1, 0x00ffffff(R2)   // MOVH.P R1, 16777215(R2)
	MOVH	R1, -0x00ffffff(R2)  // MOVH R1, -16777215(R2)
	MOVH.W	R1, -0x00ffffff(R2)  // MOVH.W R1, -16777215(R2)
	MOVH.P	R1, -0x00ffffff(R2)  // MOVH.P R1, -16777215(R2)
	MOVH	0x00ffffff(R2), R1   // MOVH 16777215(R2), R1
	MOVH.P	0x00ffffff(R2), R1   // MOVH.P 16777215(R2), R1
	MOVH.W	0x00ffffff(R2), R1   // MOVH.W 16777215(R2), R1
	MOVH	-0x00ffffff(R2), R1  // MOVH -16777215(R2), R1
	MOVH.P	-0x00ffffff(R2), R1  // MOVH.P -16777215(R2), R1
	MOVH.W	-0x00ffffff(R2), R1  // MOVH.W -16777215(R2), R1
	MOVHS	R1, 0x00ffffff(R2)   // MOVHS R1, 16777215(R2)
	MOVHS.W	R1, 0x00ffffff(R2)   // MOVHS.W R1, 16777215(R2)
	MOVHS.P	R1, 0x00ffffff(R2)   // MOVHS.P R1, 16777215(R2)
	MOVHS	R1, -0x00ffffff(R2)  // MOVHS R1, -16777215(R2)
	MOVHS.W	R1, -0x00ffffff(R2)  // MOVHS.W R1, -16777215(R2)
	MOVHS.P	R1, -0x00ffffff(R2)  // MOVHS.P R1, -16777215(R2)
	MOVHS	0x00ffffff(R2), R1   // MOVHS 16777215(R2), R1
	MOVHS.P	0x00ffffff(R2), R1   // MOVHS.P 16777215(R2), R1
	MOVHS.W	0x00ffffff(R2), R1   // MOVHS.W 16777215(R2), R1
	MOVHS	-0x00ffffff(R2), R1  // MOVHS -16777215(R2), R1
	MOVHS.P	-0x00ffffff(R2), R1  // MOVHS.P -16777215(R2), R1
	MOVHS.W	-0x00ffffff(R2), R1  // MOVHS.W -16777215(R2), R1
	MOVHU	R1, 0x00ffffff(R2)   // MOVHU R1, 16777215(R2)
	MOVHU.W	R1, 0x00ffffff(R2)   // MOVHU.W R1, 16777215(R2)
	MOVHU.P	R1, 0x00ffffff(R2)   // MOVHU.P R1, 16777215(R2)
	MOVHU	R1, -0x00ffffff(R2)  // MOVHU R1, -16777215(R2)
	MOVHU.W	R1, -0x00ffffff(R2)  // MOVHU.W R1, -16777215(R2)
	MOVHU.P	R1, -0x00ffffff(R2)  // MOVHU.P R1, -16777215(R2)
	MOVHU	0x00ffffff(R2), R1   // MOVHU 16777215(R2), R1
	MOVHU.P	0x00ffffff(R2), R1   // MOVHU.P 16777215(R2), R1
	MOVHU.W	0x00ffffff(R2), R1   // MOVHU.W 16777215(R2), R1
	MOVHU	-0x00ffffff(R2), R1  // MOVHU -16777215(R2), R1
	MOVHU.P	-0x00ffffff(R2), R1  // MOVHU.P -16777215(R2), R1
	MOVHU.W	-0x00ffffff(R2), R1  // MOVHU.W -16777215(R2), R1
	MOVH	R0, math·Exp(SB)     // MOVH R0, math.Exp(SB)
	MOVH	math·Exp(SB), R0     // MOVH math.Exp(SB), R0
	MOVHS	R0, math·Exp(SB)     // MOVHS R0, math.Exp(SB)
	MOVHS	math·Exp(SB), R0     // MOVHS math.Exp(SB), R0
	MOVHU	R0, math·Exp(SB)     // MOVHU R0, math.Exp(SB)
	MOVHU	math·Exp(SB), R0     // MOVHU math.Exp(SB), R0
	MOVHS	R0<<0(R1), R2                                     // f02091e1
	MOVHS.U	R0<<0(R1), R2                                     // f02011e1
	MOVHS.W	R0<<0(R1), R2                                     // f020b1e1
	MOVHS.P	R0<<0(R1), R2                                     // f02091e0
	MOVH	R0<<0(R1), R2                                     // f02091e1
	MOVH.U	R0<<0(R1), R2                                     // f02011e1
	MOVH.W	R0<<0(R1), R2                                     // f020b1e1
	MOVH.P	R0<<0(R1), R2                                     // f02091e0
	MOVHU	R0<<0(R1), R2                                     // b02091e1
	MOVHU.U	R0<<0(R1), R2                                     // b02011e1
	MOVHU.W	R0<<0(R1), R2                                     // b020b1e1
	MOVHU.P	R0<<0(R1), R2                                     // b02091e0
	MOVHS	R2, R5<<0(R1)                                     // b52081e1
	MOVHS.U	R2, R5<<0(R1)                                     // b52001e1
	MOVHS.W	R2, R5<<0(R1)                                     // b520a1e1
	MOVHS.P	R2, R5<<0(R1)                                     // b52081e0
	MOVH	R2, R5<<0(R1)                                     // b52081e1
	MOVH.U	R2, R5<<0(R1)                                     // b52001e1
	MOVH.W	R2, R5<<0(R1)                                     // b520a1e1
	MOVH.P	R2, R5<<0(R1)                                     // b52081e0
	MOVHU	R2, R5<<0(R1)                                     // b52081e1
	MOVHU.U	R2, R5<<0(R1)                                     // b52001e1
	MOVHU.W	R2, R5<<0(R1)                                     // b520a1e1
	MOVHU.P	R2, R5<<0(R1)                                     // b52081e0
	MOVHS	R3@>0, R9                                         // 7390bfe6
	MOVHS	R3@>8, R9                                         // 7394bfe6
	MOVHS	R3@>16, R9                                        // 7398bfe6
	MOVHS	R3@>24, R9                                        // 739cbfe6
	MOVH	R3@>0, R9                                         // 7390bfe6
	MOVH	R3@>8, R9                                         // 7394bfe6
	MOVH	R3@>16, R9                                        // 7398bfe6
	MOVH	R3@>24, R9                                        // 739cbfe6
	MOVHU	R5@>0, R1                                         // 7510ffe6
	MOVHU	R5@>8, R1                                         // 7514ffe6
	MOVHU	R5@>16, R1                                        // 7518ffe6
	MOVHU	R5@>24, R1                                        // 751cffe6

	RET	foo(SB)

//
// END
//
//	LTYPEE
//	{
//		outcode($1, Always, &nullgen, 0, &nullgen);
//	}
	END
