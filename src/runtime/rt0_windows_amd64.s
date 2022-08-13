// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

TEXT _rt0_amd64_windows(SB),NOSPLIT,$-8
	JMP	_rt0_amd64(SB)

// When building with -buildmode=(c-shared or c-archive), this
// symbol is called. For dynamic libraries it is called when the
// library is loaded. For static libraries it is called when the
// final executable starts, during the C runtime initialization
// phase.
// Leave space for four pointers on the stack as required
// by the Windows amd64 calling convention.
TEXT _rt0_amd64_windows_lib(SB),NOSPLIT,$0x20
	// Create a new thread to do the runtime initialization and return.
	MOVQ	_cgo_sys_thread_create(SB), AX
	MOVQ	$_rt0_amd64_windows_lib_go(SB), CX
	MOVQ	$0, DX
	CALL	AX
	RET

TEXT _rt0_amd64_windows_lib_go(SB),NOSPLIT,$0
	MOVQ  $0, DI
	MOVQ	$0, SI
	MOVQ	$runtime·rt0_go(SB), AX
	JMP	AX
