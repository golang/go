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
TEXT _rt0_arm_darwin_lib(SB),NOSPLIT,$104
	// Preserve callee-save registers.
	MOVW    R4, 12(R13)
	MOVW    R5, 16(R13)
	MOVW    R6, 20(R13)
	MOVW    R7, 24(R13)
	MOVW    R8, 28(R13)
	MOVW    R11, 32(R13)

	MOVD	F8, (32+8*1)(R13)
	MOVD	F9, (32+8*2)(R13)
	MOVD	F10, (32+8*3)(R13)
	MOVD	F11, (32+8*4)(R13)
	MOVD	F12, (32+8*5)(R13)
	MOVD	F13, (32+8*6)(R13)
	MOVD	F14, (32+8*7)(R13)
	MOVD	F15, (32+8*8)(R13)

	MOVW  R0, _rt0_arm_darwin_lib_argc<>(SB)
	MOVW  R1, _rt0_arm_darwin_lib_argv<>(SB)

	// Synchronous initialization.
	MOVW	$runtime·libpreinit(SB), R3
	CALL	(R3)

	// Create a new thread to do the runtime initialization and return.
	MOVW  _cgo_sys_thread_create(SB), R3
	CMP   $0, R3
	B.EQ  nocgo
	MOVW  $_rt0_arm_darwin_lib_go(SB), R0
	MOVW  $0, R1
	BL    (R3)
	B rr
nocgo:
	MOVW  $0x400000, R0
	MOVW  R0, (R13) // stacksize
	MOVW  $_rt0_arm_darwin_lib_go(SB), R0
	MOVW  R0, 4(R13) // fn
	MOVW  $0, R0
	MOVW  R0, 8(R13) // fnarg
	MOVW  $runtime·newosproc0(SB), R3
	BL    (R3)
rr:
	// Restore callee-save registers and return.
	MOVW    12(R13), R4
	MOVW    16(R13), R5
	MOVW    20(R13), R6
	MOVW    24(R13), R7
	MOVW    28(R13), R8
	MOVW    32(R13), R11
	MOVD	(32+8*1)(R13), F8
	MOVD	(32+8*2)(R13), F9
	MOVD	(32+8*3)(R13), F10
	MOVD	(32+8*4)(R13), F11
	MOVD	(32+8*5)(R13), F12
	MOVD	(32+8*6)(R13), F13
	MOVD	(32+8*7)(R13), F14
	MOVD	(32+8*8)(R13), F15
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
