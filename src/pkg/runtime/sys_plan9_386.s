// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "defs_GOOS_GOARCH.h"
#include "asm_386.h"

// setldt(int entry, int address, int limit)
TEXT runtime·setldt(SB),7,$0
	RET

TEXT runtime·open(SB),7,$0
	MOVL    $14, AX
	INT     $64
	RET

TEXT runtime·pread(SB),7,$0
	MOVL    $50, AX
	INT     $64
	RET

TEXT runtime·pwrite(SB),7,$0
	MOVL    $51, AX
	INT     $64
	RET

TEXT runtime·close(SB),7,$0
	MOVL	$4, AX
	INT		$64
	RET

TEXT runtime·exits(SB),7,$0
	MOVL    $8, AX
	INT     $64
	RET

TEXT runtime·brk_(SB),7,$0
	MOVL    $24, AX
	INT     $64
	RET

TEXT runtime·sleep(SB),7,$0
	MOVL    $17, AX
	INT     $64
	RET

TEXT runtime·plan9_semacquire(SB),7,$0
	MOVL	$37, AX
	INT	$64
	RET
	
TEXT runtime·plan9_semrelease(SB),7,$0
	MOVL	$38, AX
	INT	$64
	RET
	
TEXT runtime·rfork(SB),7,$0
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
	MOVL	_tos(SB), AX
	MOVL	tos_pid(AX), AX
	MOVL	AX, m_procid(BX)	// save pid as m->procid
	
	CALL	runtime·stackcheck(SB)	// smashes AX, CX
	
	MOVL	0(DX), DX	// paranoia; check they are not nil
	MOVL	0(BX), BX
	
	// more paranoia; check that stack splitting code works
	PUSHL	SI
	CALL	runtime·emptyfunc(SB)
	POPL	SI
	
	CALL	SI	// fn()
	CALL	runtime·exit(SB)
	RET
