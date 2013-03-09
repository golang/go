// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "zasm_GOOS_GOARCH.h"

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

TEXT runtime·seek(SB),7,$0
	MOVL	$39, AX
	INT	$64
	CMPL	AX, $-1
	JNE	4(PC)
	MOVL	a+0(FP), CX
	MOVL	AX, 0(CX)
	MOVL	AX, 4(CX)
	RET

TEXT runtime·close(SB),7,$0
	MOVL	$4, AX
	INT	$64
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

TEXT runtime·plan9_tsemacquire(SB),7,$0
	MOVL	$52, AX
	INT	$64
	RET

TEXT runtime·notify(SB),7,$0
	MOVL	$28, AX
	INT	$64
	RET

TEXT runtime·noted(SB),7,$0
	MOVL	$29, AX
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

	// Initialize AX from TOS struct.
	MOVL	procid(AX), AX
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

// void sigtramp(void *ureg, int8 *note)
TEXT runtime·sigtramp(SB),7,$0
	get_tls(AX)

	// check that m exists
	MOVL	m(AX), BX
	CMPL	BX, $0
	JNE	3(PC)
	CALL	runtime·badsignal(SB) // will exit
	RET

	// save args
	MOVL	ureg+4(SP), CX
	MOVL	note+8(SP), DX

	// change stack
	MOVL	m_gsignal(BX), BP
	MOVL	g_stackbase(BP), BP
	MOVL	BP, SP

	// make room for args and g
	SUBL	$16, SP

	// save g
	MOVL	g(AX), BP
	MOVL	BP, 12(SP)

	// g = m->gsignal
	MOVL	m_gsignal(BX), DI
	MOVL	DI, g(AX)

	// load args and call sighandler
	MOVL	CX, 0(SP)
	MOVL	DX, 4(SP)
	MOVL	BP, 8(SP)

	CALL	runtime·sighandler(SB)

	// restore g
	get_tls(BX)
	MOVL	12(SP), BP
	MOVL	BP, g(BX)

	// call noted(AX)
	MOVL	AX, 0(SP)
	CALL	runtime·noted(SB)
	RET

// Only used by the 64-bit runtime.
TEXT runtime·setfpmasks(SB),7,$0
	RET

#define ERRMAX 128	/* from os_plan9.h */

// func errstr() String
// Only used by package syscall.
// Grab error string due to a syscall made
// in entersyscall mode, without going
// through the allocator (issue 4994).
// See ../syscall/asm_plan9_386.s:/·Syscall/
TEXT runtime·errstr(SB),7,$0
	get_tls(AX)
	MOVL	m(AX), BX
	MOVL	m_errstr(BX), CX
	MOVL	CX, 4(SP)
	MOVL	$ERRMAX, 8(SP)
	MOVL	$41, AX
	INT	$64

	// syscall requires caller-save
	MOVL	4(SP), CX

	// push the argument
	PUSHL	CX
	CALL	runtime·findnull(SB)
	POPL	CX
	MOVL	AX, 8(SP)
	RET
