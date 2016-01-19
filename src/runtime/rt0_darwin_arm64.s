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
TEXT _rt0_arm64_darwin_lib(SB),NOSPLIT,$0
	// R27 is REGTMP, reserved for liblink. It is used below to
	// move R0/R1 into globals. However in the standard ARM64 calling
	// convention, it is a callee-saved register. So we save it to a
	// temporary register.
	MOVD  R27, R7

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

	MOVD  R7, R27
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
