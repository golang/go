// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT _rt0_arm_darwin(SB),7,$-4
	// prepare arguments for main (_rt0_go)
	MOVW	(R13), R0	// argc
	MOVW	$4(R13), R1		// argv
	MOVW	$main(SB), R4
	B		(R4)

// When linking with -buildmode=c-archive or -buildmode=c-shared,
// this symbol is called from a global initialization function.
//
// Note that all currently shipping darwin/arm platforms require
// cgo and do not support c-shared.
TEXT _rt0_arm_darwin_lib(SB),NOSPLIT,$12
	MOVW  R0, _rt0_arm_darwin_lib_argc<>(SB)
	MOVW  R1, _rt0_arm_darwin_lib_argv<>(SB)

	// Create a new thread to do the runtime initialization and return.
	MOVW  _cgo_sys_thread_create(SB), R4
	CMP   $0, R4
	B.EQ  nocgo
	MOVW  $_rt0_arm_darwin_lib_go(SB), R0
	MOVW  $0, R1
	BL    (R4)
	RET
nocgo:
	MOVW  $0x400000, R0
	MOVW  $_rt0_arm_darwin_lib_go(SB), R1
	MOVW  $0, R2
	MOVW  R0,  (R13) // stacksize
	MOVW  R1, 4(R13) // fn
	MOVW  R2, 8(R13) // fnarg
	MOVW  $runtime·newosproc0(SB), R4
	BL    (R4)
	RET

TEXT _rt0_arm_darwin_lib_go(SB),NOSPLIT,$0
	MOVW  _rt0_arm_darwin_lib_argc<>(SB), R0
	MOVW  _rt0_arm_darwin_lib_argv<>(SB), R1
	MOVW  R0,  (R13)
	MOVW  R1, 4(R13)
	MOVW  $runtime·rt0_go(SB), R4
	B     (R4)

DATA  _rt0_arm_darwin_lib_argc<>(SB)/4, $0
GLOBL _rt0_arm_darwin_lib_argc<>(SB),NOPTR, $4
DATA  _rt0_arm_darwin_lib_argv<>(SB)/4, $0
GLOBL _rt0_arm_darwin_lib_argv<>(SB),NOPTR, $4

TEXT main(SB),NOSPLIT,$-8
	// save argc and argv onto stack
	MOVM.DB.W [R0-R1], (R13)
	MOVW	$runtime·rt0_go(SB), R4
	B		(R4)
