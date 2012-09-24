// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "defs_GOOS_GOARCH.h"
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
