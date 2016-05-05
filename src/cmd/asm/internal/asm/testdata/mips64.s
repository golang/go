// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This input was created by taking the ppc64 testcase and modified
// by hand.

TEXT foo(SB),7,$0

//inst:
//
// load ints and bytes
//
//	LMOVW rreg ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVW	R1, R2
	MOVW	LO, R1
	MOVW	HI, R1
	MOVW	R1, LO
	MOVW	R1, HI
	MOVV	R1, R2
	MOVV	LO, R1
	MOVV	HI, R1
	MOVV	R1, LO
	MOVV	R1, HI

//	LMOVW addr ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVW	foo<>+3(SB), R2
	MOVW	16(R1), R2
	MOVW	(R1), R2
	MOVV	foo<>+3(SB), R2
	MOVV	16(R1), R2
	MOVV	(R1), R2

//	LMOVB rreg ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVB	R1, R2

//	LMOVB addr ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVB	foo<>+3(SB), R2
	MOVB	16(R1), R2
	MOVB	(R1), R2

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
//	LMOVW rreg ',' addr
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVW	R1, foo<>+3(SB)
	MOVW	R1, 16(R2)
	MOVW	R1, (R2)
	MOVV	R1, foo<>+3(SB)
	MOVV	R1, 16(R2)
	MOVV	R1, (R2)

//	LMOVB rreg ',' addr
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVB	R1, foo<>+3(SB)
	MOVB	R1, 16(R2)
	MOVB	R1, (R2)

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
	MOVW	FCR0, R1

//	LMOVW freg ','  fpscr
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVW	R1, FCR0

//	LMOVW rreg ',' mreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVW	R1, M1
	MOVV	R1, M1

//	LMOVW mreg ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MOVW	M1, R1
	MOVV	M1, R1


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
	ADD	R1, R2, R3

//	LADDW imm ',' sreg ',' rreg
//	{
//		outcode(int($1), &$2, int($4), &$6);
//	}
	ADD	$1, R2, R3

//	LADDW rreg ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	ADD	R1, R2

//	LADDW imm ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	ADD	$4, R1

//	LMUL rreg ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	MUL	R1, R2

//	LSHW rreg ',' sreg ',' rreg
//	{
//		outcode(int($1), &$2, int($4), &$6);
//	}
	SLL	R1, R2, R3

//	LSHW rreg ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	SLL	R1, R2

//	LSHW imm ',' sreg ',' rreg
//	{
//		outcode(int($1), &$2, int($4), &$6);
//	}
	SLL	$4, R1, R2

//	LSHW imm ',' rreg
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
	SLL	$4, R1

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
// branch
//
//	LBRA rel
//	{
//		outcode(int($1), &nullgen, 0, &$2);
//	}
	BEQ	R1, 2(PC)
label0:
	JMP	1(PC)
	BEQ	R1, 2(PC)
	JMP	label0+0 // JMP 64
	BEQ	R1, 2(PC)
	JAL	1(PC) // CALL 1(PC)
	BEQ	R1, 2(PC)
	JAL	label0+0 // CALL 64

//	LBRA addr
//	{
//		outcode(int($1), &nullgen, 0, &$2);
//	}
	BEQ	R1, 2(PC)
	JMP	0(R1) // JMP (R1)
	BEQ	R1, 2(PC)
	JMP	foo+0(SB) // JMP foo(SB)
	BEQ	R1, 2(PC)
	JAL	0(R1) // CALL (R1)
	BEQ	R1, 2(PC)
	JAL	foo+0(SB) // CALL foo(SB)

//
// BEQ/BNE
//
//	LBRA rreg ',' rel
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
label1:
	BEQ	R1, 1(PC)
	BEQ	R1, label1 // BEQ R1, 79

//	LBRA rreg ',' sreg ',' rel
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
label2:
	BEQ	R1, R2, 1(PC)
	BEQ	R1, R2, label2 // BEQ R1, R2, 81

//
// other integer conditional branch
//
//	LBRA rreg ',' rel
//	{
//		outcode(int($1), &$2, 0, &$4);
//	}
label3:
	BLTZ	R1, 1(PC)
	BLTZ	R1, label3 // BLTZ R1, 83

//
// floating point conditional branch
//
//	LBRA rel
label4:
	BFPT	1(PC)
	BFPT	label4 // BFPT 85


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
	WORD	$1

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

// END
//
//	LEND	comma // asm doesn't support the trailing comma.
//	{
//		outcode(int($1), &nullgen, 0, &nullgen);
//	}
	END
