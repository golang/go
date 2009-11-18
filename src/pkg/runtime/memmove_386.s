	TEXT	memmove(SB), $0

	MOVL	to+0(FP), DI
	MOVL	fr+4(FP), SI
	MOVL	n+8(FP), BX
	JLT	fault

/*
 * check and set for backwards
 * should we look closer for overlap?
 */
	CMPL	SI, DI
	JLS	back

/*
 * foreward copy loop
 */
	MOVL	BX, CX
	SHRL	$2, CX
	ANDL	$3, BX

	REP;	MOVSL
	MOVL	BX, CX
	REP;	MOVSB

	MOVL	to+0(FP),AX
	RET
/*
 * whole thing backwards has
 * adjusted addresses
 */
back:
	ADDL	BX, DI
	ADDL	BX, SI
	STD

/*
 * copy
 */
	MOVL	BX, CX
	SHRL	$2, CX
	ANDL	$3, BX

	SUBL	$4, DI
	SUBL	$4, SI
	REP;	MOVSL

	ADDL	$3, DI
	ADDL	$3, SI
	MOVL	BX, CX
	REP;	MOVSB

	CLD
	MOVL	to+0(FP),AX
	RET

/*
 * if called with negative count,
 * treat as error rather than
 * rotating all of memory
 */
fault:
	MOVL	$0,SI
	MOVL	0(SI), AX
	RET
