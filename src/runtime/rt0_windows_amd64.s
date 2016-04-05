// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

TEXT _rt0_amd64_windows(SB),NOSPLIT,$-8
	LEAQ	8(SP), SI // argv
	MOVQ	0(SP), DI // argc
	MOVQ	$main(SB), AX
	JMP	AX

// When building with -buildmode=(c-shared or c-archive), this
// symbol is called. For dynamic libraries it is called when the
// library is loaded. For static libraries it is called when the
// final executable starts, during the C runtime initialization
// phase.
TEXT _rt0_amd64_windows_lib(SB),NOSPLIT,$0x28
	MOVQ	BP, 0x00(SP)
	MOVQ	BX, 0x08(SP)
	MOVQ	AX, 0x10(SP)
	MOVQ  CX, 0x18(SP)
	MOVQ  DX, 0x20(SP)

	// Create a new thread to do the runtime initialization and return.
	MOVQ	_cgo_sys_thread_create(SB), AX
	MOVQ	$_rt0_amd64_windows_lib_go(SB), CX
	MOVQ	$0, DX
	CALL	AX

	MOVQ	0x00(SP), BP
	MOVQ	0x08(SP), BX
	MOVQ	0x10(SP), AX
	MOVQ	0x18(SP), CX
	MOVQ	0x20(SP), DX
	RET

TEXT _rt0_amd64_windows_lib_go(SB),NOSPLIT,$0
	MOVQ  $0, DI
	MOVQ	$0, SI
	MOVQ	$runtime·rt0_go(SB), AX
	JMP	AX

TEXT main(SB),NOSPLIT,$-8
	MOVQ	$runtime·rt0_go(SB), AX
	JMP	AX
