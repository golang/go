// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "defs.h"
#include "386/asm.h"

// setldt(int entry, int address, int limit)
TEXT setldt(SB),7,$0
	RET

TEXT write(SB),7,$0
	MOVL    $20, AX
	INT     $64
	RET

TEXT exits(SB),7,$0
	MOVL    $8, AX
	INT     $64
	RET

TEXT brk_(SB),7,$0
	MOVL    $24, AX
	INT     $64
	RET

TEXT plan9_semacquire(SB),7,$0
	MOVL	$37, AX
	INT	$64
	RET
	
TEXT plan9_semrelease(SB),7,$0
	MOVL	$38, AX
	INT	$64
	RET
	
TEXT rfork(SB),7,$0
	MOVL    $19, AX // rfork
	INT     $64

	// In parent, return.
	CMPL	AX, $0
	JEQ	2(PC)
	RET

	// In child on old stack.
	MOVL	mm+12(SP), BX	// m
	MOVL	gg+16(SP), DX	// g
	MOVL	fn+20(SP), SI	// fn

	// set SP to be on the new child stack
	MOVL	stack+8(SP), CX
	MOVL	CX, SP

	// Initialize m, g.
	get_tls(AX)
	MOVL	DX, g(AX)
	MOVL	BX, m(AX)

	// Initialize AX from _tos->pid
	MOVL	0xdfffeff8, AX
	MOVL	AX, m_procid(BX)	// save pid as m->procid

	CALL	stackcheck(SB)	// smashes AX, CX
	
	MOVL	0(DX), DX	// paranoia; check they are not nil
	MOVL	0(BX), BX
	
	// more paranoia; check that stack splitting code works
	PUSHAL
	CALL	emptyfunc(SB)
	POPAL
	
	CALL	SI	// fn()
	CALL	exit(SB)
	RET
