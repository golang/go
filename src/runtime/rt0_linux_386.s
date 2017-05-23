// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT _rt0_386_linux(SB),NOSPLIT,$8
	MOVL	8(SP), AX
	LEAL	12(SP), BX
	MOVL	AX, 0(SP)
	MOVL	BX, 4(SP)
	CALL	main(SB)
	INT	$3

// When building with -buildmode=c-shared, this symbol is called when the shared
// library is loaded.
TEXT _rt0_386_linux_lib(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	PUSHL	BX
	PUSHL	SI
	PUSHL	DI

	MOVL	8(BP), AX
	MOVL	AX, _rt0_386_linux_lib_argc<>(SB)
	MOVL	12(BP), AX
	MOVL	AX, _rt0_386_linux_lib_argv<>(SB)

	// Synchronous initialization.
	MOVL	$runtime·libpreinit(SB), AX
	CALL	AX

	SUBL	$8, SP

	// Create a new thread to do the runtime initialization.
	MOVL	_cgo_sys_thread_create(SB), AX
	TESTL	AX, AX
	JZ	nocgo
	MOVL	$_rt0_386_linux_lib_go(SB), BX
	MOVL	BX, 0(SP)
	MOVL	$0, 4(SP)
	CALL	AX
	JMP	restore

nocgo:
	MOVL	$0x800000, 0(SP)                    // stacksize = 8192KB
	MOVL	$_rt0_386_linux_lib_go(SB), AX
	MOVL	AX, 4(SP)                           // fn
	MOVL	$runtime·newosproc0(SB), AX
	CALL	AX

restore:
	ADDL	$8, SP
	POPL	DI
	POPL	SI
	POPL	BX
	POPL	BP
	RET

TEXT _rt0_386_linux_lib_go(SB),NOSPLIT,$12
	MOVL	_rt0_386_linux_lib_argc<>(SB), AX
	MOVL	AX, 0(SP)
	MOVL	_rt0_386_linux_lib_argv<>(SB), AX
	MOVL	AX, 4(SP)
	MOVL	$runtime·rt0_go(SB), AX
	CALL	AX
	RET

DATA _rt0_386_linux_lib_argc<>(SB)/4, $0
GLOBL _rt0_386_linux_lib_argc<>(SB),NOPTR, $4
DATA _rt0_386_linux_lib_argv<>(SB)/4, $0
GLOBL _rt0_386_linux_lib_argv<>(SB),NOPTR, $4

TEXT main(SB),NOSPLIT,$0
	JMP	runtime·rt0_go(SB)
