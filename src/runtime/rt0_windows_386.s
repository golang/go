// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT _rt0_386_windows(SB),NOSPLIT,$12
	MOVL	12(SP), AX
	LEAL	16(SP), BX
	MOVL	AX, 4(SP)
	MOVL	BX, 8(SP)
	MOVL	$-1, 0(SP) // return PC for main
	JMP	_main(SB)

// When building with -buildmode=(c-shared or c-archive), this
// symbol is called. For dynamic libraries it is called when the
// library is loaded. For static libraries it is called when the
// final executable starts, during the C runtime initialization
// phase.
TEXT _rt0_386_windows_lib(SB),NOSPLIT,$0x1C
	MOVL	BP, 0x08(SP)
	MOVL	BX, 0x0C(SP)
	MOVL	AX, 0x10(SP)
	MOVL  CX, 0x14(SP)
	MOVL  DX, 0x18(SP)

	// Create a new thread to do the runtime initialization and return.
	MOVL	_cgo_sys_thread_create(SB), AX
	MOVL	$_rt0_386_windows_lib_go(SB), 0x00(SP)
	MOVL	$0, 0x04(SP)

	 // Top two items on the stack are passed to _cgo_sys_thread_create
	 // as parameters. This is the calling convention on 32-bit Windows.
	CALL	AX

	MOVL	0x08(SP), BP
	MOVL	0x0C(SP), BX
	MOVL	0x10(SP), AX
	MOVL	0x14(SP), CX
	MOVL	0x18(SP), DX
	RET

TEXT _rt0_386_windows_lib_go(SB),NOSPLIT,$0
	MOVL  $0, DI
	MOVL	$0, SI
	MOVL	$runtime·rt0_go(SB), AX
	JMP	AX

TEXT _main(SB),NOSPLIT,$0
	JMP	runtime·rt0_go(SB)
