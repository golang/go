// This input was created by taking the instruction productions in
// the old assembler's (8a's) grammar and hand-writing complete
// instructions for each rule, to guarantee we cover the same space.

TEXT foo(SB), 7, $0

// LTYPE1 nonrem	{ outcode(int($1), &$2); }
	SETCC	AX
	SETCC	foo+4(SB)

// LTYPE2 rimnon	{ outcode(int($1), &$2); }
	DIVB	AX
	DIVB	foo+4(SB)
	PUSHL	$foo+4(SB)
	POPL		AX

// LTYPE3 rimrem	{ outcode(int($1), &$2); }
	SUBB	$1, AX
	SUBB	$1, foo+4(SB)
	SUBB	BX, AX
	SUBB	BX, foo+4(SB)

// LTYPE4 remrim	{ outcode(int($1), &$2); }
	CMPB	AX, $1
	CMPB	foo+4(SB), $4
	CMPB	BX, AX
	CMPB	foo+4(SB), BX

// LTYPER nonrel	{ outcode(int($1), &$2); }
label:
	JC	label // JCS
	JC	-1(PC) // JCS -1(PC)

// LTYPEC spec3	{ outcode(int($1), &$2); }
	CALL	AX
	JCS	2(PC)
	JMP	*AX // JMP AX
	CALL	*foo(SB)
	JCS	2(PC)
	JMP	$4
	JCS	2(PC)
	JMP	label // JMP 16
	CALL	foo(SB)
//	CALL	(AX*4) // TODO: This line is silently dropped on the floor!
	CALL	foo+4(SB)(AX*4)
	CALL	*4(SP) // CALL 4(SP)
	CALL	*(AX) // CALL (AX)
	CALL	*(SP) // CALL (SP)
//	CALL	*(AX*4) // TODO: This line is silently dropped on the floor!
	CALL	*(AX)(AX*4) // CALL (AX)(AX*4)
	CALL	4(SP)
	CALL	(AX)
	CALL	(SP)
//	CALL	(AX*4) // TODO: This line is silently dropped on the floor!
	JCS	2(PC)
	JMP	(AX)(AX*4)

// LTYPEN spec4	{ outcode(int($1), &$2); }
	NOP
	NOP	AX
	NOP	foo+4(SB)

// LTYPES spec5	{ outcode(int($1), &$2); }
	SHLL	$4, BX
	SHLL	$4, foo+4(SB)
	SHLL	$4, foo+4(SB):AX // SHLL $4, AX, foo+4(SB)

// LTYPEM spec6	{ outcode(int($1), &$2); }
	MOVL	AX, BX
	MOVL	$4, BX
	
// LTYPEI spec7	{ outcode(int($1), &$2); }
	IMULL	AX
	IMULL	$4, CX
	IMULL	AX, BX

// LTYPEXC spec9	{ outcode(int($1), &$2); }
	CMPPD	X0, X1, 4
	CMPPD	foo+4(SB), X1, 4

// LTYPEX spec10	{ outcode(int($1), &$2); }
	PINSRD	$1, (AX), X0
	PINSRD	$2, foo+4(FP), X0

// Was bug: LOOP is a branch instruction.
	JCS	2(PC)
loop:
	LOOP	loop // LOOP

// LTYPE0 nonnon	{ outcode(int($1), &$2); }
	RET
