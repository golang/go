// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "zasm_GOOS_GOARCH.h"

// setldt(int entry, int address, int limit)
TEXT runtime·setldt(SB),7,$0
	RET

TEXT runtime·open(SB),7,$0
	MOVQ	$0x8000, AX
	MOVQ	$14, BP
	SYSCALL
	RET

TEXT runtime·pread(SB),7,$0
	MOVQ	$0x8000, AX
	MOVQ	$50, BP
	SYSCALL
	RET

TEXT runtime·pwrite(SB),7,$0
	MOVQ	$0x8000, AX
	MOVQ	$51, BP
	SYSCALL
	RET

// int32 _seek(int64*, int32, int64, int32)
TEXT _seek<>(SB),7,$0
	MOVQ	$0x8000, AX
	MOVQ	$39, BP
	SYSCALL
	RET

// int64 seek(int32, int64, int32)
TEXT runtime·seek(SB),7,$56
	LEAQ	new+48(SP), CX
	MOVQ	CX, 0(SP)
	MOVQ	fd+0(FP), CX
	MOVQ	CX, 8(SP)
	MOVQ	off+8(FP), CX
	MOVQ	CX, 16(SP)
	MOVQ	whence+16(FP), CX
	MOVQ	CX, 24(SP)
	CALL	_seek<>(SB)
	CMPL	AX, $0
	JGE	2(PC)
	MOVQ	$-1, new+48(SP)
	MOVQ	new+48(SP), AX
	RET

TEXT runtime·close(SB),7,$0
	MOVQ	$0x8000, AX
	MOVQ	$4, BP
	SYSCALL
	RET

TEXT runtime·exits(SB),7,$0
	MOVQ	$0x8000, AX
	MOVQ	$8, BP
	SYSCALL
	RET

TEXT runtime·brk_(SB),7,$0
	MOVQ	$0x8000, AX
	MOVQ	$24, BP
	SYSCALL
	RET

TEXT runtime·sleep(SB),7,$0
	MOVQ	$0x8000, AX
	MOVQ	$17, BP
	SYSCALL
	RET

TEXT runtime·plan9_semacquire(SB),7,$0
	MOVQ	$0x8000, AX
	MOVQ	$37, BP
	SYSCALL
	RET

TEXT runtime·plan9_tsemacquire(SB),7,$0
	MOVQ	$0x8000, AX
	MOVQ	$52, BP
	SYSCALL
	RET

TEXT runtime·notify(SB),7,$0
	MOVQ	$0x8000, AX
	MOVQ	$28, BP
	SYSCALL
	RET

TEXT runtime·noted(SB),7,$0
	MOVQ	$0x8000, AX
	MOVQ	$29, BP
	SYSCALL
	RET
	
TEXT runtime·plan9_semrelease(SB),7,$0
	MOVQ	$0x8000, AX
	MOVQ	$38, BP
	SYSCALL
	RET

TEXT runtime·nanotime(SB),7,$0
	MOVQ	$0x8000, AX
	MOVQ	$60, BP
	SYSCALL
	RET

TEXT runtime·rfork(SB),7,$0
	MOVQ	$0x8000, AX
	MOVQ	$19, BP // rfork
	SYSCALL

	// In parent, return.
	CMPQ	AX, $0
	JEQ	2(PC)
	RET

	// In child on forked stack.
	MOVQ	mm+24(SP), BX	// m
	MOVQ	gg+32(SP), DX	// g
	MOVQ	fn+40(SP), SI	// fn

	// set SP to be on the new child stack
	MOVQ	stack+16(SP), CX
	MOVQ	CX, SP

	// Initialize m, g.
	get_tls(AX)
	MOVQ	DX, g(AX)
	MOVQ	BX, m(AX)

	// Initialize AX from pid in TLS.
	MOVQ	procid(AX), AX
	MOVQ	AX, m_procid(BX)	// save pid as m->procid
	
	CALL	runtime·stackcheck(SB)	// smashes AX, CX
	
	MOVQ	0(DX), DX	// paranoia; check they are not nil
	MOVQ	0(BX), BX
	
	CALL	SI	// fn()
	CALL	runtime·exit(SB)
	RET

// This is needed by asm_amd64.s
TEXT runtime·settls(SB),7,$0
	RET

// void sigtramp(void *ureg, int8 *note)
TEXT runtime·sigtramp(SB),7,$0
	get_tls(AX)

	// check that m exists
	MOVQ	m(AX), BX
	CMPQ	BX, $0
	JNE	3(PC)
	CALL	runtime·badsignal(SB) // will exit
	RET

	// save args
	MOVQ	ureg+8(SP), CX
	MOVQ	note+16(SP), DX

	// change stack
	MOVQ	m_gsignal(BX), R10
	MOVQ	g_stackbase(R10), BP
	MOVQ	BP, SP

	// make room for args and g
	SUBQ	$32, SP

	// save g
	MOVQ	g(AX), BP
	MOVQ	BP, 24(SP)

	// g = m->gsignal
	MOVQ	R10, g(AX)

	// load args and call sighandler
	MOVQ	CX, 0(SP)
	MOVQ	DX, 8(SP)
	MOVQ	BP, 16(SP)

	CALL	runtime·sighandler(SB)

	// restore g
	get_tls(BX)
	MOVQ	24(SP), R10
	MOVQ	R10, g(BX)

	// call noted(AX)
	MOVQ	AX, 0(SP)
	CALL	runtime·noted(SB)
	RET

TEXT runtime·setfpmasks(SB),7,$8
	STMXCSR	0(SP)
	MOVL	0(SP), AX
	ANDL	$~0x3F, AX
	ORL	$(0x3F<<7), AX
	MOVL	AX, 0(SP)
	LDMXCSR	0(SP)
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
	MOVQ	m(AX), BX
	MOVQ	m_errstr(BX), CX
	MOVQ	CX, 8(SP)
	MOVQ	$ERRMAX, 16(SP)
	MOVQ	$0x8000, AX
	MOVQ	$41, BP
	SYSCALL

	// syscall requires caller-save
	MOVQ	8(SP), CX

	// push the argument
	PUSHQ	CX
	CALL	runtime·findnull(SB)
	POPQ	CX
	MOVQ	AX, 16(SP)
	RET
