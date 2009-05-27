// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

TEXT _rt0_386(SB),7,$0
	// copy arguments forward on an even stack
	MOVL	0(SP), AX		// argc
	LEAL	4(SP), BX		// argv
	SUBL	$128, SP		// plenty of scratch
	ANDL	$~7, SP
	MOVL	AX, 120(SP)		// save argc, argv away
	MOVL	BX, 124(SP)

/*
	// write "go386\n"
	PUSHL	$6
	PUSHL	$hello(SB)
	PUSHL	$1
	CALL	sys·write(SB)
	POPL	AX
	POPL	AX
	POPL	AX
*/

	CALL	ldt0setup(SB)

	// set up %fs to refer to that ldt entry
	MOVL	$(7*8+7), AX
	MOVW	AX, FS

	// store through it, to make sure it works
	MOVL	$0x123, 0(FS)
	MOVL	tls0(SB), AX
	CMPL	AX, $0x123
	JEQ	ok
	MOVL	AX, 0
ok:

	// set up m and g "registers"
	// g is 0(FS), m is 4(FS)
	LEAL	g0(SB), CX
	MOVL	CX, 0(FS)
	LEAL	m0(SB), AX
	MOVL	AX, 4(FS)

	// save m->g0 = g0
	MOVL	CX, 0(AX)

	// create istack out of the OS stack
	LEAL	(-8192+104)(SP), AX	// TODO: 104?
	MOVL	AX, 0(CX)	// 8(g) is stack limit (w 104b guard)
	MOVL	SP, 4(CX)	// 12(g) is base
	CALL	emptyfunc(SB)	// fault if stack check is wrong

	// convention is D is always cleared
	CLD

	CALL	check(SB)

	// saved argc, argv
	MOVL	120(SP), AX
	MOVL	AX, 0(SP)
	MOVL	124(SP), AX
	MOVL	AX, 4(SP)
	CALL	args(SB)
	CALL	osinit(SB)
	CALL	schedinit(SB)

	// create a new goroutine to start program
	PUSHL	$mainstart(SB)	// entry
	PUSHL	$8	// arg size
	CALL	sys·newproc(SB)
	POPL	AX
	POPL	AX

	// start this M
	CALL	mstart(SB)

	INT $3
	RET

TEXT mainstart(SB),7,$0
	CALL	main·init(SB)
	CALL	initdone(SB)
	CALL	main·main(SB)
	PUSHL	$0
	CALL	exit(SB)
	POPL	AX
	INT $3
	RET

TEXT	breakpoint(SB),7,$0
	BYTE $0xcc
	RET

// go-routine
TEXT	gogo(SB), 7, $0
	MOVL	4(SP), AX	// gobuf
	MOVL	0(AX), SP	// restore SP
	MOVL	4(AX), AX
	MOVL	AX, 0(SP)	// put PC on the stack
	MOVL	$1, AX
	RET

TEXT gosave(SB), 7, $0
	MOVL	4(SP), AX	// gobuf
	MOVL	SP, 0(AX)	// save SP
	MOVL	0(SP), BX
	MOVL	BX, 4(AX)	// save PC
	MOVL	$0, AX	// return 0
	RET

// support for morestack

// return point when leaving new stack.
// save AX, jmp to lesstack to switch back
TEXT	retfromnewstack(SB),7,$0
	MOVL	4(FS), BX	// m
	MOVL	AX, 8(BX)	// save AX in m->cret
	JMP	lessstack(SB)

// gogo, returning 2nd arg instead of 1
TEXT gogoret(SB), 7, $0
	MOVL	8(SP), AX	// return 2nd arg
	MOVL	4(SP), BX	// gobuf
	MOVL	0(BX), SP	// restore SP
	MOVL	4(BX), BX
	MOVL	BX, 0(SP)	// put PC on the stack
	RET

TEXT setspgoto(SB), 7, $0
	MOVL	4(SP), AX	// SP
	MOVL	8(SP), BX	// fn to call
	MOVL	12(SP), CX	// fn to return
	MOVL	AX, SP
	PUSHL	CX
	JMP	BX
	POPL	AX	// not reached
	RET

// bool cas(int32 *val, int32 old, int32 new)
// Atomically:
//	if(*val == old){
//		*val = new;
//		return 1;
//	}else
//		return 0;
TEXT cas(SB), 7, $0
	MOVL	4(SP), BX
	MOVL	8(SP), AX
	MOVL	12(SP), CX
	LOCK
	CMPXCHGL	CX, 0(BX)
	JZ 3(PC)
	MOVL	$0, AX
	RET
	MOVL	$1, AX
	RET

// void jmpdefer(byte*);
// 1. pop the caller
// 2. sub 5 bytes from the callers return
// 3. jmp to the argument
TEXT jmpdefer(SB), 7, $0
	MOVL	4(SP), AX	// function
	ADDL	$(4+56), SP	// pop saved PC and callers frame
	SUBL	$5, (SP)	// reposition his return address
	JMP	AX		// and goto function

TEXT	sys·memclr(SB),7,$0
	MOVL	4(SP), DI		// arg 1 addr
	MOVL	8(SP), CX		// arg 2 count
	ADDL	$3, CX
	SHRL	$2, CX
	MOVL	$0, AX
	CLD
	REP
	STOSL
	RET

TEXT	sys·getcallerpc+0(SB),7,$0
	MOVL	x+0(FP),AX		// addr of first arg
	MOVL	-4(AX),AX		// get calling pc
	RET

TEXT	sys·setcallerpc+0(SB),7,$0
	MOVL	x+0(FP),AX		// addr of first arg
	MOVL	x+4(FP), BX
	MOVL	BX, -4(AX)		// set calling pc
	RET

TEXT ldt0setup(SB),7,$16
	// set up ldt 7 to point at tls0
	// ldt 1 would be fine on Linux, but on OS X, 7 is as low as we can go.
	MOVL	$7, 0(SP)
	LEAL	tls0(SB), AX
	MOVL	AX, 4(SP)
	MOVL	$32, 8(SP)	// sizeof(tls array)
	CALL	setldt(SB)
	RET

GLOBL m0+0(SB), $1024
GLOBL g0+0(SB), $1024

GLOBL tls0+0(SB), $32

TEXT emptyfunc(SB),0,$0
	RET

TEXT	abort(SB),7,$0
	INT $0x3

DATA hello+0(SB)/8, $"go386\n\z\z"
GLOBL hello+0(SB), $8

