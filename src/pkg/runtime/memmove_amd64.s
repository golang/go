	TEXT	memmove(SB), $0

	MOVQ	to+0(FP), DI
	MOVQ	fr+8(FP), SI
	MOVLQSX	n+16(FP), BX
	JLT	fault

/*
 * check and set for backwards
 * should we look closer for overlap?
 */
	CMPQ	SI, DI
	JLS	back

/*
 * foreward copy loop
 */
	MOVQ	BX, CX
	SHRQ	$3, CX
	ANDQ	$7, BX

	REP;	MOVSQ
	MOVQ	BX, CX
	REP;	MOVSB

	MOVQ	to+0(FP),AX
	RET
/*
 * whole thing backwards has
 * adjusted addresses
 */
back:
	ADDQ	BX, DI
	ADDQ	BX, SI
	STD

/*
 * copy
 */
	MOVQ	BX, CX
	SHRQ	$3, CX
	ANDQ	$7, BX

	SUBQ	$8, DI
	SUBQ	$8, SI
	REP;	MOVSQ

	ADDQ	$7, DI
	ADDQ	$7, SI
	MOVQ	BX, CX
	REP;	MOVSB

	CLD
	MOVQ	to+0(FP),AX
	RET

/*
 * if called with negative count,
 * treat as error rather than
 * rotating all of memory
 */
fault:
	MOVQ	$0,SI
	MOVQ	0(SI), AX
	RET
