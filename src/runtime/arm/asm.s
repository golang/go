// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

TEXT _rt0_arm(SB),7,$0
	// copy arguments forward on an even stack
    //      	MOVW	$0(SP), R0
    //	MOVL	0(SP), R1		// argc
//	LEAL	4(SP), R1		// argv
//	SUBL	$128, SP		// plenty of scratch
//	ANDL	$~7, SP
//	MOVL	AX, 120(SP)		// save argc, argv away
//	MOVL	BX, 124(SP)


// 	// write "go386\n"
// 	PUSHL	$6
// 	PUSHL	$hello(SB)
// 	PUSHL	$1
// 	CALL	sys·write(SB)
// 	POPL	AX
// 	POPL	AX
// 	POPL	AX


// 	CALL	ldt0setup(SB)

	// set up %fs to refer to that ldt entry
// 	MOVL	$(7*8+7), AX
// 	MOVW	AX, FS

// 	// store through it, to make sure it works
// 	MOVL	$0x123, 0(FS)
// 	MOVL	tls0(SB), AX
// 	CMPL	AX, $0x123
// 	JEQ	ok
// 	MOVL	AX, 0
// ok:

// 	// set up m and g "registers"
// 	// g is 0(FS), m is 4(FS)
// 	LEAL	g0(SB), CX
// 	MOVL	CX, 0(FS)
// 	LEAL	m0(SB), AX
// 	MOVL	AX, 4(FS)

// 	// save m->g0 = g0
// 	MOVL	CX, 0(AX)

// 	// create istack out of the OS stack
// 	LEAL	(-8192+104)(SP), AX	// TODO: 104?
// 	MOVL	AX, 0(CX)	// 8(g) is stack limit (w 104b guard)
// 	MOVL	SP, 4(CX)	// 12(g) is base
// 	CALL	emptyfunc(SB)	// fault if stack check is wrong

// 	// convention is D is always cleared
// 	CLD

// 	CALL	check(SB)

// 	// saved argc, argv
// 	MOVL	120(SP), AX
// 	MOVL	AX, 0(SP)
// 	MOVL	124(SP), AX
// 	MOVL	AX, 4(SP)
// 	CALL	args(SB)
// 	CALL	osinit(SB)
// 	CALL	schedinit(SB)

// 	// create a new goroutine to start program
// 	PUSHL	$mainstart(SB)	// entry
// 	PUSHL	$8	// arg size
// 	CALL	sys·newproc(SB)
// 	POPL	AX
// 	POPL	AX

// 	// start this M
// 	CALL	mstart(SB)

	BL	main�main(SB)
	MOVW	$99, R0
	SWI	$0x00900001

