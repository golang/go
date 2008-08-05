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

	// set the per-goroutine and per-mach registers

	LEAQ	m0(SB), R14		// dedicated m. register
	LEAQ	g0(SB), R15		// dedicated g. register
	MOVQ	R15, 0(R14)		// m has pointer to its g0

	// create istack out of the given (operating system) stack

	LEAQ	(-1024+104)(SP), AX
	MOVQ	AX, 0(R15)		// 0(R15) is stack limit (w 104b guard)
	MOVQ	SP, 8(R15)		// 8(R15) is base

	CALL	check(SB)

	MOVL	16(SP), AX		// copy argc
	MOVL	AX, 0(SP)
	MOVQ	24(SP), AX		// copy argv
	MOVQ	AX, 8(SP)
	CALL	args(SB)
	CALL	schedinit(SB)
	CALL	main·init_function(SB) // initialization

	// create a new goroutine to start program

	PUSHQ	$main·main(SB)		// entry
	PUSHQ	$16			// arg size
	CALL	sys·newproc(SB)
	CALL	m0init(SB)
	POPQ	AX
	POPQ	AX

	CALL	notok(SB)		// never returns
	RET

TEXT	sys·breakpoint(SB),7,$0
	BYTE	$0xcc
	RET

TEXT	FLUSH(SB),7,$0
	RET

/*
 *  go-routine
 */
TEXT gogo(SB), 7, $0
	MOVQ	8(SP), AX		// gobuf
	MOVQ	0(AX), SP		// restore SP
	MOVQ	8(AX), AX
	MOVQ	AX, 0(SP)		// put PC on the stack
	MOVL	$1, AX			// return 1
	RET

TEXT gosave(SB), 7, $0
	MOVQ	8(SP), AX		// gobuf
	MOVQ	SP, 0(AX)		// save SP
	MOVQ	0(SP), BX
	MOVQ	BX, 8(AX)		// save PC
	MOVL	$0, AX			// return 0
	RET

/*
 * support for morestack
 */

// return point when leaving new stack.  save AX, jmp to oldstack to switch back
TEXT retfromnewstack(SB), 7, $0
	MOVQ	AX, 16(R14)	// save AX in m->cret
	MOVQ	$oldstack(SB), AX
	JMP	AX

// gogo, returning 2nd arg instead of 1
TEXT gogoret(SB), 7, $0
	MOVQ	16(SP), AX			// return 2nd arg
	MOVQ	8(SP), BX		// gobuf
	MOVQ	0(BX), SP		// restore SP
	MOVQ	8(BX), BX
	MOVQ	BX, 0(SP)		// put PC on the stack
	RET

TEXT setspgoto(SB), 7, $0
	MOVQ	8(SP), AX		// SP
	MOVQ	16(SP), BX		// fn to call
	MOVQ	24(SP), CX		// fn to return
	MOVQ	AX, SP
	PUSHQ	CX
	JMP	BX
	POPQ	AX
	RET

// bool cas(int32 *val, int32 old, int32 new)
// Atomically:
//	if(*val == old){
//		*val = new;
//		return 1;
//	}else
//		return 0;
TEXT cas(SB), 7, $0
	MOVQ	8(SP), BX
	MOVL	16(SP), AX
	MOVL	20(SP), CX
	LOCK
	CMPXCHGL	CX, 0(BX)
	JZ 3(PC)
	MOVL	$0, AX
	RET
	MOVL	$1, AX
	RET

