// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// No need for _rt0_arm64_darwin as darwin/arm64 only
// supports external linking.
TEXT _rt0_arm64_darwin(SB),NOSPLIT,$-8
	MOVD	$42, R0
	MOVD	$1, R16	// SYS_exit
	SVC	$0x80

// When linking with -buildmode=c-archive or -buildmode=c-shared,
// this symbol is called from a global initialization function.
//
// Note that all currently shipping darwin/arm64 platforms require
// cgo and do not support c-shared.
TEXT _rt0_arm64_darwin_lib(SB),NOSPLIT,$88
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

	MOVD  R0, _rt0_arm64_darwin_lib_argc<>(SB)
	MOVD  R1, _rt0_arm64_darwin_lib_argv<>(SB)

	// Synchronous initialization.
	MOVD	$runtime·libpreinit(SB), R4
	BL	(R4)

	// Create a new thread to do the runtime initialization and return.
	MOVD  _cgo_sys_thread_create(SB), R4
	MOVD  $_rt0_arm64_darwin_lib_go(SB), R0
	MOVD  $0, R1
	BL    (R4)

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

TEXT main(SB),NOSPLIT,$-8
	MOVD	$runtime·rt0_go(SB), R2
	BL	(R2)
exit:
	MOVD	$0, R0
	MOVD	$1, R16	// sys_exit
	SVC	$0x80
	B	exit
