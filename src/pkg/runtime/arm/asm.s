// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

TEXT _rt0_arm(SB),7,$0
	MOVW $setR12(SB), R12

	// copy arguments forward on an even stack
	MOVW	0(SP), R0		// argc
	MOVW	4(SP), R1		// argv
	SUB	$128, SP		// plenty of scratch
	AND	$~7, SP
	MOVW	R0, 120(SP)		// save argc, argv away
	MOVW	R1, 124(SP)

	// set up m and g registers
	// g is R10, m is R9
	MOVW	$g0(SB), R10
	MOVW	$m0(SB), R9

	// save m->g0 = g0
	MOVW	R10, 0(R9)

	// create istack out of the OS stack
	MOVW	$(-8192+104)(SP), R0
	MOVW	R0, 0(R10)	// 0(g) is stack limit (w 104b guard)
	MOVW	SP, 4(R10)	// 4(g) is base
	BL	emptyfunc(SB)	// fault if stack check is wrong

	BL	check(SB)

	// saved argc, argv
	MOVW	120(SP), R0
	MOVW	R0, 0(SP)
	MOVW	124(SP), R0
	MOVW	R0, 4(SP)
	BL	args(SB)
	BL	osinit(SB)
	BL	schedinit(SB)

	// create a new goroutine to start program
	MOVW	$mainstart(SB), R0
	MOVW.W	R0, -4(SP)
	MOVW	$8, R0
	MOVW.W	R0, -4(SP)
	MOVW	$0, R0
	MOVW.W	R0, -4(SP)	// push $0 as guard
	BL	sys·newproc(SB)
	MOVW	$12(SP), SP	// pop args and LR

	// start this M
	BL	mstart(SB)

	MOVW	$0, R0
	SWI	$0x00900001
	B	_dep_dummy(SB)	// Never reached


TEXT mainstart(SB),7,$0
	BL	main·init(SB)
	BL	initdone(SB)
	BL	main·main(SB)
	MOVW	$0, R0
	MOVW.W	R0, -4(SP)
	MOVW.W	R14, -4(SP)	// Push link as well
	BL	exit(SB)
	MOVW	$8(SP), SP	// pop args and LR
	RET

// TODO(kaib): remove these once linker works properly
// pull in dummy dependencies
TEXT _dep_dummy(SB),7,$0
	BL	sys·morestack(SB)
	BL	sys·morestackx(SB)
	BL	_div(SB)
	BL	_divu(SB)
	BL	_mod(SB)
	BL	_modu(SB)
	BL	_modu(SB)


TEXT	breakpoint(SB),7,$0
	BL	abort(SB)
//	BYTE $0xcc
//	RET

// go-routine
TEXT	gogo(SB), 7, $0
	BL	abort(SB)
//	MOVL	4(SP), AX	// gobuf
//	MOVL	0(AX), SP	// restore SP
//	MOVL	4(AX), AX
//	MOVL	AX, 0(SP)	// put PC on the stack
//	MOVL	$1, AX
//	RET

TEXT gosave(SB), 7, $0
	BL	abort(SB)
//	MOVL	4(SP), AX	// gobuf
//	MOVL	SP, 0(AX)	// save SP
//	MOVL	0(SP), BX
//	MOVL	BX, 4(AX)	// save PC
//	MOVL	$0, AX	// return 0
//	RET

// support for morestack

// return point when leaving new stack.
// save R0, jmp to lesstack to switch back
TEXT	retfromnewstack(SB),7,$0
	MOVW	R0,12(R9)	// m->cret
	B	lessstack(SB)

// gogo, returning 2nd arg instead of 1
TEXT gogoret(SB), 7, $0
	MOVW	8(SP), R0	// return 2nd arg
	MOVW	4(SP), R1	// gobuf
	MOVW	0(R1), SP	// restore SP
	MOVW	4(R1), PC	// restore PC

TEXT setspgoto(SB), 7, $0
	MOVW	4(SP), R0	// SP
	MOVW	8(SP), R1	// fn to call
	MOVW	12(SP), R2	// fn to return into
	MOVW	R2, R14		// restore LR
	MOVW	R0, SP
	MOVW	R1, PC		// goto

// Optimization to make inline stack splitting code smaller
// R0 is original first argument
// R1 is arg_num << 24 | autosize >> 3
TEXT sys·morestackx(SB), 7, $0
	MOVW	R0, 4(SP)	// Save arg0
	MOVW	R1<<8, R2
	MOVW	R2>>5, R2
	MOVW	R2, 4(R10)	// autooffset into g
	MOVW	R1>>24, R2
	MOVW	R2<<3, R2
	MOVW	R2, 8(R10)	// argsize into g
	B	sys·morestack(SB)

// bool cas(int32 *val, int32 old, int32 new)
// Atomically:
//	if(*val == old){
//		*val = new;
//		return 1;
//	}else
//		return 0;
#define	LDREX(a,r)	WORD	$(0xe<<28|0x01900f9f | (a)<<16 | (r)<<12)
#define	STREX(a,v,r)	WORD	$(0xe<<28|0x01800f90 | (a)<<16 | (r)<<12 | (v)<<0)

TEXT	cas+0(SB),0,$12		/* r0 holds p */
	MOVW	ov+4(FP), R1
	MOVW	nv+8(FP), R2
spin:
/*	LDREX	0(R0),R3	*/
	LDREX(0,3)
	CMP.S	R3, R1
	BNE	fail
/*	STREX	0(R0),R2,R4	*/
	STREX(0,2,4)
	CMP.S	$0, R4
	BNE	spin
	MOVW	$1, R0
	RET
fail:
	MOVW	$0, R0
	RET

// void jmpdefer(fn, sp);
// called from deferreturn.
// 1. pop the caller
// 2. sub 5 bytes from the callers return
// 3. jmp to the argument
TEXT jmpdefer(SB), 7, $0
	BL	abort(SB)
//	MOVL	4(SP), AX	// fn
//	MOVL	8(SP), BX	// caller sp
//	LEAL	-4(BX), SP	// caller sp after CALL
//	SUBL	$5, (SP)	// return to CALL again
//	JMP	AX	// but first run the deferred function

TEXT	sys·memclr(SB),7,$0
	BL	abort(SB)
//	MOVL	4(SP), DI		// arg 1 addr
//	MOVL	8(SP), CX		// arg 2 count
//	ADDL	$3, CX
//	SHRL	$2, CX
//	MOVL	$0, AX
//	CLD
//	REP
//	STOSL
//	RET

TEXT	sys·getcallerpc+0(SB),7,$0
	BL	abort(SB)
//	MOVL	x+0(FP),AX		// addr of first arg
//	MOVL	-4(AX),AX		// get calling pc
//	RET

TEXT	sys·setcallerpc+0(SB),7,$0
	BL	abort(SB)
//	MOVL	x+0(FP),AX		// addr of first arg
//	MOVL	x+4(FP), BX
//	MOVL	BX, -4(AX)		// set calling pc
//	RET

TEXT emptyfunc(SB),0,$0
	RET

TEXT abort(SB),7,$0
	MOVW	$0, R0
	MOVW	(R0), R1

