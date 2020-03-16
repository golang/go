// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// No need for _rt0_arm64_darwin as darwin/arm64 only
// supports external linking.
TEXT _rt0_arm64_darwin(SB),NOSPLIT|NOFRAME,$0
	MOVD	$42, R0
	BL  libc_exit(SB)

// When linking with -buildmode=c-archive or -buildmode=c-shared,
// this symbol is called from a global initialization function.
//
// Note that all currently shipping darwin/arm64 platforms require
// cgo and do not support c-shared.
TEXT _rt0_arm64_darwin_lib(SB),NOSPLIT,$168
	// Preserve callee-save registers.
	MOVD R19, 24(RSP)
	MOVD R20, 32(RSP)
	MOVD R21, 40(RSP)
	MOVD R22, 48(RSP)
	MOVD R23, 56(RSP)
	MOVD R24, 64(RSP)
	MOVD R25, 72(RSP)
	MOVD R26, 80(RSP)
	MOVD R27, 88(RSP)
	MOVD g, 96(RSP)
	FMOVD F8, 104(RSP)
	FMOVD F9, 112(RSP)
	FMOVD F10, 120(RSP)
	FMOVD F11, 128(RSP)
	FMOVD F12, 136(RSP)
	FMOVD F13, 144(RSP)
	FMOVD F14, 152(RSP)
	FMOVD F15, 160(RSP)

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
	MOVD 24(RSP), R19
	MOVD 32(RSP), R20
	MOVD 40(RSP), R21
	MOVD 48(RSP), R22
	MOVD 56(RSP), R23
	MOVD 64(RSP), R24
	MOVD 72(RSP), R25
	MOVD 80(RSP), R26
	MOVD 88(RSP), R27
	MOVD 96(RSP), g
	FMOVD 104(RSP), F8
	FMOVD 112(RSP), F9
	FMOVD 120(RSP), F10
	FMOVD 128(RSP), F11
	FMOVD 136(RSP), F12
	FMOVD 144(RSP), F13
	FMOVD 152(RSP), F14
	FMOVD 160(RSP), F15

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

TEXT main(SB),NOSPLIT|NOFRAME,$0
	MOVD	$runtime·rt0_go(SB), R2
	BL	(R2)
exit:
	MOVD	$0, R0
	MOVD	$1, R16	// sys_exit
	SVC	$0x80
	B	exit
