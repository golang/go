// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT _rt0_amd64_darwin(SB),NOSPLIT,$-8
	LEAQ	8(SP), SI // argv
	MOVQ	0(SP), DI // argc
	MOVQ	$main(SB), AX
	JMP	AX

// When linking with -shared, this symbol is called when the shared library
// is loaded.
TEXT _rt0_amd64_darwin_lib(SB),NOSPLIT,$0x48
	MOVQ	BX, 0x18(SP)
	MOVQ	BP, 0x20(SP)
	MOVQ	R12, 0x28(SP)
	MOVQ	R13, 0x30(SP)
	MOVQ	R14, 0x38(SP)
	MOVQ	R15, 0x40(SP)

	MOVQ	DI, _rt0_amd64_darwin_lib_argc<>(SB)
	MOVQ	SI, _rt0_amd64_darwin_lib_argv<>(SB)

	// Synchronous initialization.
	MOVQ	$runtime·libpreinit(SB), AX
	CALL	AX

	// Create a new thread to do the runtime initialization and return.
	MOVQ	_cgo_sys_thread_create(SB), AX
	TESTQ	AX, AX
	JZ	nocgo
	MOVQ	$_rt0_amd64_darwin_lib_go(SB), DI
	MOVQ	$0, SI
	CALL	AX
	JMP	restore

nocgo:
	MOVQ	$8388608, 0(SP)                    // stacksize
	MOVQ	$_rt0_amd64_darwin_lib_go(SB), AX
	MOVQ	AX, 8(SP)                          // fn
	MOVQ	$0, 16(SP)                         // fnarg
	MOVQ	$runtime·newosproc0(SB), AX
	CALL	AX

restore:
	MOVQ	0x18(SP), BX
	MOVQ	0x20(SP), BP
	MOVQ	0x28(SP), R12
	MOVQ	0x30(SP), R13
	MOVQ	0x38(SP), R14
	MOVQ	0x40(SP), R15
	RET

TEXT _rt0_amd64_darwin_lib_go(SB),NOSPLIT,$0
	MOVQ	_rt0_amd64_darwin_lib_argc<>(SB), DI
	MOVQ	_rt0_amd64_darwin_lib_argv<>(SB), SI
	MOVQ	$runtime·rt0_go(SB), AX
	JMP	AX

DATA _rt0_amd64_darwin_lib_argc<>(SB)/8, $0
GLOBL _rt0_amd64_darwin_lib_argc<>(SB),NOPTR, $8
DATA _rt0_amd64_darwin_lib_argv<>(SB)/8, $0
GLOBL _rt0_amd64_darwin_lib_argv<>(SB),NOPTR, $8

TEXT main(SB),NOSPLIT,$-8
	MOVQ	$runtime·rt0_go(SB), AX
	JMP	AX
