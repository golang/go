// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT _rt0_amd64_linux(SB),NOSPLIT,$-8
	LEAQ	8(SP), SI // argv
	MOVQ	0(SP), DI // argc
	MOVQ	$main(SB), AX
	JMP	AX

// When building with -buildmode=c-shared, this symbol is called when the shared
// library is loaded.
TEXT _rt0_amd64_linux_lib(SB),NOSPLIT,$40
	MOVQ	DI, _rt0_amd64_linux_lib_argc<>(SB)
	MOVQ	SI, _rt0_amd64_linux_lib_argv<>(SB)

	// Create a new thread to do the runtime initialization and return.
	MOVQ	_cgo_sys_thread_create(SB), AX
	TESTQ	AX, AX
	JZ	nocgo
	MOVQ	$_rt0_amd64_linux_lib_go(SB), DI
	MOVQ	$0, SI
	CALL	AX
	RET
nocgo:
	MOVQ	$8388608, 0(SP)                    // stacksize
	MOVQ	$_rt0_amd64_linux_lib_go(SB), AX
	MOVQ	AX, 8(SP)                          // fn
	MOVQ	$0, 16(SP)                         // fnarg
	MOVQ	$runtime·newosproc0(SB), AX
	CALL	AX
	RET

TEXT _rt0_amd64_linux_lib_go(SB),NOSPLIT,$0
	MOVQ	_rt0_amd64_linux_lib_argc<>(SB), DI
	MOVQ	_rt0_amd64_linux_lib_argv<>(SB), SI
	MOVQ	$runtime·rt0_go(SB), AX
	JMP	AX

DATA _rt0_amd64_linux_lib_argc<>(SB)/8, $0
GLOBL _rt0_amd64_linux_lib_argc<>(SB),NOPTR, $8
DATA _rt0_amd64_linux_lib_argv<>(SB)/8, $0
GLOBL _rt0_amd64_linux_lib_argv<>(SB),NOPTR, $8

TEXT main(SB),NOSPLIT,$-8
	MOVQ	$runtime·rt0_go(SB), AX
	JMP	AX
