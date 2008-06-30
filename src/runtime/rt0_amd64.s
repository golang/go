// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.


TEXT	_rt0_amd64(SB),7,$-8

	// copy arguments forward on an even stack

	MOVQ	0(SP), AX		// argc
	LEAQ	8(SP), BX		// argv
	SUBQ	$(4*8+7), SP		// 2args 2auto
	ANDQ	$~7, SP
	MOVQ	AX, 16(SP)
	MOVQ	BX, 24(SP)

	// allocate the per-user block

	LEAQ	peruser<>(SB), R15	// dedicated u. register

	LEAQ	(-4096+104+4*8)(SP), AX
	MOVQ	AX, 0(R15)		// 0(R15) is stack limit (w 104b guard)

	MOVL	$1024, AX
	MOVL	AX, 0(SP)
	CALL	mal(SB)

	LEAQ	104(AX), BX
	MOVQ	BX, 16(R15)		// 16(R15) is limit of istack (w 104b guard)

	ADDQ	0(SP), AX
	LEAQ	(-4*8)(AX), BX
	MOVQ	BX, 24(R15)		// 24(R15) is base of istack (w auto*4)

	CALL	check(SB)

	// process the arguments

	MOVL	16(SP), AX		// copy argc
	MOVL	AX, 0(SP)
	MOVQ	24(SP), AX		// copy argv
	MOVQ	AX, 8(SP)
	CALL	args(SB)

	CALL	main·main(SB)

	MOVQ	$0, AX
	MOVQ	AX, 0(SP)		// exit status
	CALL	sys·exit(SB)

	CALL	notok(SB)		// fault
	RET

//
// the calling sequence for a routine that
// needs N bytes stack, A args.
//
//	N1 = (N+160 > 4096)? N+160: 0
//	A1 = A
//
// if N <= 75
//	CMPQ	SP, 0(R15)
//	JHI	3(PC)
//	MOVQ	$(N1<<0) | (A1<<32)), AX
//	CALL	_morestack
//
// if N > 75
//	LEAQ	(-N-75)(SP), AX
//	CMPQ	AX, 0(R15)
//	JHI	3(PC)
//	MOVQ	$(N1<<0) | (A1<<32)), AX
//	CALL	_morestack
//

TEXT	_morestack(SB), 7, $0
	// save stuff on interrupt stack

	MOVQ	24(R15), BX		// istack
	MOVQ	SP, 8(BX)		// old SP
	MOVQ	AX, 16(BX)		// magic number
	MOVQ	0(R15), AX		// old limit
	MOVQ	AX, 24(BX)

	// switch and set up new limit

	MOVQ	BX, SP
	MOVQ	16(R15), AX		// istack limit
	MOVQ	AX, 0(R15)

	// allocate a new stack max of request and 4k

	MOVL	16(SP), AX		// magic number
	CMPL	AX, $4096
	JHI	2(PC)
	MOVL	$4096, AX
	MOVL	AX, 0(SP)
	CALL	mal(SB)

	// switch to new stack

	MOVQ	SP, BX			// istack
	ADDQ	$104, AX		// new stack limit
	MOVQ	AX, 0(R15)
	ADDQ	0(SP), AX
	LEAQ	(-104-4*8)(AX), SP	// new SP
	MOVQ	8(R15), AX
	MOVQ	AX, 0(SP)		// old base
	MOVQ	SP, 8(R15)		// new base

	// copy needed stuff from istack to new stack

	MOVQ	16(BX), AX		// magic number
	MOVQ	AX, 16(SP)
	MOVQ	24(BX), AX		// old limit
	MOVQ	AX, 24(SP)
	MOVQ	8(BX), AX		// old SP
	MOVQ	AX, 8(SP)

// are there parameters

	MOVL	20(SP), CX		// copy count
	CMPL	CX, $0
	JEQ	easy

// copy in

	LEAQ	16(AX), SI
	SUBQ	CX, SP
	MOVQ	SP, DI
	SHRL	$3, CX
	CLD
	REP
	MOVSQ

	// call the intended
	CALL	0(AX)

// copy out

	MOVQ	SP, SI
	MOVQ	8(R15), BX		// new base
	MOVQ	8(BX), AX		// old SP
	LEAQ	16(AX), DI
	MOVL	20(BX), CX		// copy count
	SHRL	$3, CX
	CLD
	REP
	MOVSQ

	// restore old SP and limit
	MOVQ	8(R15), SP		// new base
	MOVQ	24(SP), AX		// old limit
	MOVQ	AX, 0(R15)
	MOVQ	0(SP), AX
	MOVQ	AX, 8(R15)		// old base
	MOVQ	8(SP), AX		// old SP
	MOVQ	AX, SP

	// and return to the call behind mine
	ADDQ	$8, SP
	RET

easy:
	CALL	0(AX)

	// restore old SP and limit
	MOVQ	24(SP), AX		// old limit
	MOVQ	AX, 0(R15)
	MOVQ	0(SP), AX
	MOVQ	AX, 8(R15)		// old base
	MOVQ	8(SP), AX		// old SP
	MOVQ	AX, SP

	// and return to the call behind mine
	ADDQ	$8, SP
	RET

// marker.  must be here; used by traceback() to discover calls to _morestack
TEXT _endmorestack(SB), 7, $-8
	RET

TEXT	FLUSH(SB),7,$-8
	RET

TEXT	getu(SB),7,$-8
	MOVQ	R15, AX
	RET


GLOBL	peruser<>(SB),$64
