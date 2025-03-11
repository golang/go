// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

// setldt(int entry, int address, int limit)
TEXT runtime·setldt(SB),NOSPLIT,$0
	RET

TEXT runtime·open(SB),NOSPLIT,$0
	MOVL    $14, AX
	INT     $64
	MOVL	AX, ret+12(FP)
	RET

TEXT runtime·pread(SB),NOSPLIT,$0
	MOVL    $50, AX
	INT     $64
	MOVL	AX, ret+20(FP)
	RET

TEXT runtime·pwrite(SB),NOSPLIT,$0
	MOVL    $51, AX
	INT     $64
	MOVL	AX, ret+20(FP)
	RET

// int32 _seek(int64*, int32, int64, int32)
TEXT _seek<>(SB),NOSPLIT,$0
	MOVL	$39, AX
	INT	$64
	RET

TEXT runtime·seek(SB),NOSPLIT,$24
	LEAL	ret+16(FP), AX
	MOVL	fd+0(FP), BX
	MOVL	offset_lo+4(FP), CX
	MOVL	offset_hi+8(FP), DX
	MOVL	whence+12(FP), SI
	MOVL	AX, 0(SP)
	MOVL	BX, 4(SP)
	MOVL	CX, 8(SP)
	MOVL	DX, 12(SP)
	MOVL	SI, 16(SP)
	CALL	_seek<>(SB)
	CMPL	AX, $0
	JGE	3(PC)
	MOVL	$-1, ret_lo+16(FP)
	MOVL	$-1, ret_hi+20(FP)
	RET

TEXT runtime·closefd(SB),NOSPLIT,$0
	MOVL	$4, AX
	INT	$64
	MOVL	AX, ret+4(FP)
	RET

TEXT runtime·dupfd(SB),NOSPLIT,$0
	MOVL	$5, AX
	INT	$64
	MOVL	AX, ret+8(FP)
	RET

TEXT runtime·exits(SB),NOSPLIT,$0
	MOVL    $8, AX
	INT     $64
	RET

TEXT runtime·brk_(SB),NOSPLIT,$0
	MOVL    $24, AX
	INT     $64
	MOVL	AX, ret+4(FP)
	RET

TEXT runtime·sleep(SB),NOSPLIT,$0
	MOVL    $17, AX
	INT     $64
	MOVL	AX, ret+4(FP)
	RET

TEXT runtime·plan9_semacquire(SB),NOSPLIT,$0
	MOVL	$37, AX
	INT	$64
	MOVL	AX, ret+8(FP)
	RET

TEXT runtime·plan9_tsemacquire(SB),NOSPLIT,$0
	MOVL	$52, AX
	INT	$64
	MOVL	AX, ret+8(FP)
	RET

// func timesplit(u uint64) (sec int64, nsec int32)
TEXT runtime·timesplit(SB),NOSPLIT,$0
	MOVL	u_lo+0(FP), AX
	MOVL	u_hi+4(FP), DX
	MOVL	$1000000000, CX
	DIVL	CX
	MOVL	AX, sec_lo+8(FP)
	MOVL	$0, sec_hi+12(FP)
	MOVL	DX, nsec+16(FP)
	RET

TEXT runtime·notify(SB),NOSPLIT,$0
	MOVL	$28, AX
	INT	$64
	MOVL	AX, ret+4(FP)
	RET

TEXT runtime·noted(SB),NOSPLIT,$0
	MOVL	$29, AX
	INT	$64
	MOVL	AX, ret+4(FP)
	RET

TEXT runtime·plan9_semrelease(SB),NOSPLIT,$0
	MOVL	$38, AX
	INT	$64
	MOVL	AX, ret+8(FP)
	RET

TEXT runtime·rfork(SB),NOSPLIT,$0
	MOVL	$19, AX
	INT	$64
	MOVL	AX, ret+4(FP)
	RET

TEXT runtime·tstart_plan9(SB),NOSPLIT,$4
	MOVL	newm+0(FP), CX
	MOVL	m_g0(CX), DX

	// Layout new m scheduler stack on os stack.
	MOVL	SP, AX
	MOVL	AX, (g_stack+stack_hi)(DX)
	SUBL	$(64*1024), AX		// stack size
	MOVL	AX, (g_stack+stack_lo)(DX)
	MOVL	AX, g_stackguard0(DX)
	MOVL	AX, g_stackguard1(DX)

	// Initialize procid from TOS struct.
	MOVL	_tos(SB), AX
	MOVL	48(AX), AX
	MOVL	AX, m_procid(CX)	// save pid as m->procid

	// Finally, initialize g.
	get_tls(BX)
	MOVL	DX, g(BX)

	CALL	runtime·stackcheck(SB)	// smashes AX, CX
	CALL	runtime·mstart(SB)

	// Exit the thread.
	MOVL	$0, 0(SP)
	CALL	runtime·exits(SB)
	JMP	0(PC)

// void sigtramp(void *ureg, int8 *note)
TEXT runtime·sigtramp(SB),NOSPLIT,$0
	get_tls(AX)

	// check that g exists
	MOVL	g(AX), BX
	CMPL	BX, $0
	JNE	3(PC)
	CALL	runtime·badsignal2(SB) // will exit
	RET

	// save args
	MOVL	ureg+0(FP), CX
	MOVL	note+4(FP), DX

	// change stack
	MOVL	g_m(BX), BX
	MOVL	m_gsignal(BX), BP
	MOVL	(g_stack+stack_hi)(BP), BP
	MOVL	BP, SP

	// make room for args and g
	SUBL	$24, SP

	// save g
	MOVL	g(AX), BP
	MOVL	BP, 20(SP)

	// g = m->gsignal
	MOVL	m_gsignal(BX), DI
	MOVL	DI, g(AX)

	// load args and call sighandler
	MOVL	CX, 0(SP)
	MOVL	DX, 4(SP)
	MOVL	BP, 8(SP)

	CALL	runtime·sighandler(SB)
	MOVL	12(SP), AX

	// restore g
	get_tls(BX)
	MOVL	20(SP), BP
	MOVL	BP, g(BX)

	// call noted(AX)
	MOVL	AX, 0(SP)
	CALL	runtime·noted(SB)
	RET

// Only used by the 64-bit runtime.
TEXT runtime·setfpmasks(SB),NOSPLIT,$0
	RET

#define ERRMAX 128	/* from os_plan9.h */

// void errstr(int8 *buf, int32 len)
TEXT errstr<>(SB),NOSPLIT,$0
	MOVL    $41, AX
	INT     $64
	RET

// func errstr() string
// Only used by package syscall.
// Grab error string due to a syscall made
// in entersyscall mode, without going
// through the allocator (issue 4994).
// See ../syscall/asm_plan9_386.s:/·Syscall/
TEXT runtime·errstr(SB),NOSPLIT,$8-8
	get_tls(AX)
	MOVL	g(AX), BX
	MOVL	g_m(BX), BX
	MOVL	(m_mOS+mOS_errstr)(BX), CX
	MOVL	CX, 0(SP)
	MOVL	$ERRMAX, 4(SP)
	CALL	errstr<>(SB)
	CALL	runtime·findnull(SB)
	MOVL	4(SP), AX
	MOVL	AX, ret_len+4(FP)
	MOVL	0(SP), AX
	MOVL	AX, ret_base+0(FP)
	RET

// never called on this platform
TEXT ·sigpanictramp(SB),NOSPLIT,$0-0
	UNDEF
