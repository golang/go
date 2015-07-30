// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This input was created by taking the instruction productions in
// the old assembler's (6a's) grammar and hand-writing complete
// instructions for each rule, to guarantee we cover the same space.

TEXT	foo(SB), 0, $0

// LTYPE1 nonrem	{ outcode($1, &$2); }
	NEGQ	R11
	NEGQ	4(R11)
	NEGQ	foo+4(SB)

// LTYPE2 rimnon	{ outcode($1, &$2); }
	INT	$4
	DIVB	R11
	DIVB	4(R11)
	DIVB	foo+4(SB)

// LTYPE3 rimrem	{ outcode($1, &$2); }
	SUBQ $4, DI
	SUBQ R11, DI
	SUBQ 4(R11), DI
	SUBQ foo+4(SB), DI
	SUBQ $4, 8(R12)
	SUBQ R11, 8(R12)
	SUBQ R11, foo+4(SB)

// LTYPE4 remrim	{ outcode($1, &$2); }
	CMPB	CX, $4

// LTYPER nonrel	{ outcode($1, &$2); }
label:
	JB	-4(PC)
	JB	label

// LTYPEC spec3	{ outcode($1, &$2); }
	JMP	-4(PC)
	JMP	label
	JMP	foo+4(SB)
	JMP	bar<>+4(SB)
	JMP	bar<>+4(SB)(R11*4)
	JMP	*4(SP)
	JMP	*(R12)
	JMP	*(R12*4)
	JMP	*(R12)(R13*4)
	JMP	*(AX)
	JMP	*(SP)
	JMP	*(AX*4)
	JMP	*(AX)(AX*4)
	JMP	4(SP)
	JMP	(R12)
	JMP	(R12*4)
	JMP	(R12)(R13*4)
	JMP	(AX)
	JMP	(SP)
	JMP	(AX*4)
	JMP	(AX)(AX*4)
	JMP	R13

// LTYPEN spec4	{ outcode($1, &$2); }
	NOP
	NOP	AX
	NOP	foo+4(SB)

// LTYPES spec5	{ outcode($1, &$2); }
	SHLL	R11, R12
	SHLL	R11, foo+4(SB)
	SHLL	R11, R11:AX // Old syntax, still accepted.

// LTYPEM spec6	{ outcode($1, &$2); }
	MOVL	AX, R11
	MOVL	$4, R11
	MOVL	AX, AX:CS

// LTYPEI spec7	{ outcode($1, &$2); }
	IMULB	$4
	IMULB	R11
	IMULB	$4, R11
	IMULB	R11, R12
	IMULB	R11, foo+4(SB)

// LTYPEXC spec8	{ outcode($1, &$2); }
	CMPPD	R11, R12, 4
	CMPPD	R11, foo+4(SB), 4

// LTYPEX spec9	{ outcode($1, &$2); }
	PINSRW	$4, R11, AX
	PINSRW	$4, foo+4(SB), AX

// LTYPERT spec10	{ outcode($1, &$2); }
	RETFL	$4

// Was bug: LOOP is a branch instruction.
loop:
	LOOP	loop

// LTYPE0 nonnon	{ outcode($1, &$2); }
	RET
