// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"
#include "cgo/abi_arm64.h"

TEXT _rt0_arm64_darwin(SB),NOSPLIT|NOFRAME,$0
	MOVD	$runtime·rt0_go(SB), R2
	BL	(R2)
exit:
	MOVD	$0, R0
	MOVD	$1, R16	// sys_exit
	SVC	$0x80
	B	exit

// When linking with -buildmode=c-archive or -buildmode=c-shared,
// this symbol is called from a global initialization function.
//
// Note that all currently shipping darwin/arm64 platforms require
// cgo and do not support c-shared.
TEXT _rt0_arm64_darwin_lib(SB),NOSPLIT,$152
	// Preserve callee-save registers.
	SAVE_R19_TO_R28(8)
	SAVE_F8_TO_F15(88)

	MOVD  R0, _rt0_arm64_darwin_lib_argc<>(SB)
	MOVD  R1, _rt0_arm64_darwin_lib_argv<>(SB)

	MOVD	$0, g // initialize g to nil

	// Synchronous initialization.
	MOVD	$runtime·libpreinit(SB), R4
	BL	(R4)

	// Create a new thread to do the runtime initialization and return.
	MOVD  _cgo_sys_thread_create(SB), R4
	MOVD  $_rt0_arm64_darwin_lib_go(SB), R0
	MOVD  $0, R1
	SUB   $16, RSP		// reserve 16 bytes for sp-8 where fp may be saved.
	BL    (R4)
	ADD   $16, RSP

	// Restore callee-save registers.
	RESTORE_R19_TO_R28(8)
	RESTORE_F8_TO_F15(88)

	RET

TEXT _rt0_arm64_darwin_lib_go(SB),NOSPLIT,$0
	MOVD  _rt0_arm64_darwin_lib_argc<>(SB), R0
	MOVD  _rt0_arm64_darwin_lib_argv<>(SB), R1
	MOVD  $runtime·rt0_go(SB), R4
	B     (R4)

DATA  _rt0_arm64_darwin_lib_argc<>(SB)/8, $0
GLOBL _rt0_arm64_darwin_lib_argc<>(SB),NOPTR, $8
DATA  _rt0_arm64_darwin_lib_argv<>(SB)/8, $0
GLOBL _rt0_arm64_darwin_lib_argv<>(SB),NOPTR, $8

// external linking entry point.
TEXT main(SB),NOSPLIT|NOFRAME,$0
	JMP	_rt0_arm64_darwin(SB)
