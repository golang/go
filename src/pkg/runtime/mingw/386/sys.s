// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "386/asm.h"

TEXT get_kernel_module(SB),7,$0
	MOVL	0x30(FS), AX		// get PEB
	MOVL	0x0c(AX), AX		// get PEB_LDR_DATA
	MOVL	0x1c(AX), AX		// get init order module list
	MOVL	(AX), AX		// get next entry (kernel module)
	MOVL	0x08(AX), AX		// get base of module
	RET

// TODO(rsc,hectorchu): Switch to m stack before call.
TEXT stdcall(SB),7,$0
	CALL	·entersyscall(SB)
	get_tls(CX)
	MOVL	m(CX), CX
	POPL	m_return_address(CX)	// save return address
	POPL	AX			// first arg is function pointer
	MOVL	SP, m_stack_pointer(CX)	// save stack pointer
	CALL	AX
	get_tls(CX)
	MOVL	m(CX), CX
	MOVL	m_stack_pointer(CX), SP
	PUSHL	AX
	PUSHL	m_return_address(CX)
	CALL	·exitsyscall(SB)
	MOVL	4(SP), AX
	RET

// TODO(rsc,hectorchu): Switch to m stack before call.
TEXT stdcall_raw(SB),7,$0
	get_tls(CX)
	MOVL	m(CX), CX
	POPL	m_return_address(CX)	// save return address
	POPL	AX			// first arg is function pointer
	MOVL	SP, m_stack_pointer(CX)	// save stack pointer
	CALL	AX
	get_tls(CX)
	MOVL	m(CX), CX
	MOVL	m_stack_pointer(CX), SP
	PUSHL	AX
	PUSHL	m_return_address(CX)
	RET

TEXT threadstart(SB),7,$0
	MOVL	4(SP), AX		// threadstart param
	MOVL	0(AX), BX		// newosproc arg stack
	MOVL	0(BX), CX		// m
	MOVL	4(BX), DX		// g

	// set up tls
	LEAL	m_tls(CX), SI
	MOVL	SI, 0x2c(FS)
	MOVL	CX, m(SI)
	MOVL	DX, g(SI)
	MOVL	SP, m_os_stack_pointer(CX)

	PUSHL	8(BX)			// stk
	PUSHL	12(BX)			// fn
	PUSHL	4(AX)			// event_handle

	// signal that we're done with thread args
	MOVL	SetEvent(SB), BX
	CALL	BX			// SetEvent(event_handle)
	POPL	BX			// fn
	POPL	SP			// stk

	CALL	stackcheck(SB)		// clobbers AX,CX
	CALL	BX			// fn()

	// cleanup stack before returning as we are stdcall
	get_tls(CX)
	MOVL	m(CX), CX
	MOVL	m_os_stack_pointer(CX), SP
	POPL	AX			// return address
	MOVL	AX, (SP)
	XORL	AX, AX
	RET

// setldt(int entry, int address, int limit)
TEXT setldt(SB),7,$0
	MOVL	address+4(FP), CX
	MOVL	CX, 0x2c(FS)
	RET

// for now, return 0,0.  only used for internal performance monitoring.
TEXT gettime(SB),7,$0
	MOVL	sec+0(FP), DI
	MOVL	$0, (DI)
	MOVL	$0, 4(DI)	// zero extend 32 -> 64 bits
	MOVL	usec+4(FP), DI
	MOVL	$0, (DI)
	RET
