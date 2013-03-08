// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

TEXT _rt0_386_plan9(SB),7, $0
	MOVL	AX, _tos(SB)
	
	// move arguments down to make room for
	// m and g at top of stack, right before Tos.
	MOVL	SP, SI
	SUBL	$8, SP
	MOVL	SP, DI
		
	MOVL	AX, CX
	SUBL	SI, CX
	CLD
	REP; MOVSB
	
	// adjust argv
	SUBL	SI, DI
	MOVL	newargc+0(SP), CX
	LEAL	newargv+4(SP), BP
argv_fix:
	ADDL	DI, 0(BP)
	ADDL	$4, BP
	LOOP	argv_fix
	
	CALL	runtime·asminit(SB)

	MOVL	0(SP), AX
	LEAL	4(SP), BX
	PUSHL	BX
	PUSHL	AX
	PUSHL	$-1

	JMP	_rt0_386(SB)

DATA  runtime·isplan9(SB)/4, $1
GLOBL runtime·isplan9(SB), $4
GLOBL _tos(SB), $4
