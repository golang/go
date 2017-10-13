// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT _rt0_arm_linux(SB),NOSPLIT,$-4
	MOVW	(R13), R0	// argc
	MOVW	$4(R13), R1		// argv
	MOVW	$_rt0_arm_linux1(SB), R4
	B		(R4)

// When building with -buildmode=c-shared, this symbol is called when the shared
// library is loaded.
TEXT _rt0_arm_linux_lib(SB),NOSPLIT,$104
	// Preserve callee-save registers. Raspberry Pi's dlopen(), for example,
	// actually cares that R11 is preserved.
	MOVW	R4, 12(R13)
	MOVW	R5, 16(R13)
	MOVW	R6, 20(R13)
	MOVW	R7, 24(R13)
	MOVW	R8, 28(R13)
	MOVW	R11, 32(R13)

	// Skip floating point registers on GOARM < 6.
	MOVB    runtime·goarm(SB), R11
	CMP $6, R11
	BLT skipfpsave
	MOVD	F8, (32+8*1)(R13)
	MOVD	F9, (32+8*2)(R13)
	MOVD	F10, (32+8*3)(R13)
	MOVD	F11, (32+8*4)(R13)
	MOVD	F12, (32+8*5)(R13)
	MOVD	F13, (32+8*6)(R13)
	MOVD	F14, (32+8*7)(R13)
	MOVD	F15, (32+8*8)(R13)
skipfpsave:
	// Save argc/argv.
	MOVW	R0, _rt0_arm_linux_lib_argc<>(SB)
	MOVW	R1, _rt0_arm_linux_lib_argv<>(SB)

	// Synchronous initialization.
	MOVW	$runtime·libpreinit(SB), R2
	CALL	(R2)

	// Create a new thread to do the runtime initialization.
	MOVW	_cgo_sys_thread_create(SB), R2
	CMP	$0, R2
	BEQ	nocgo
	MOVW	$_rt0_arm_linux_lib_go<>(SB), R0
	MOVW	$0, R1
	BL	(R2)
	B	rr
nocgo:
	MOVW	$0x800000, R0                     // stacksize = 8192KB
	MOVW	$_rt0_arm_linux_lib_go<>(SB), R1  // fn
	MOVW	R0, 4(R13)
	MOVW	R1, 8(R13)
	BL	runtime·newosproc0(SB)
rr:
	// Restore callee-save registers and return.
	MOVB    runtime·goarm(SB), R11
	CMP $6, R11
	BLT skipfprest
	MOVD	(32+8*1)(R13), F8
	MOVD	(32+8*2)(R13), F9
	MOVD	(32+8*3)(R13), F10
	MOVD	(32+8*4)(R13), F11
	MOVD	(32+8*5)(R13), F12
	MOVD	(32+8*6)(R13), F13
	MOVD	(32+8*7)(R13), F14
	MOVD	(32+8*8)(R13), F15
skipfprest:
	MOVW	12(R13), R4
	MOVW	16(R13), R5
	MOVW	20(R13), R6
	MOVW	24(R13), R7
	MOVW	28(R13), R8
	MOVW	32(R13), R11
	RET

TEXT _rt0_arm_linux_lib_go<>(SB),NOSPLIT,$8
	MOVW	_rt0_arm_linux_lib_argc<>(SB), R0
	MOVW	_rt0_arm_linux_lib_argv<>(SB), R1
	MOVW	R0, 0(R13)
	MOVW	R1, 4(R13)
	B	runtime·rt0_go(SB)

DATA _rt0_arm_linux_lib_argc<>(SB)/4,$0
GLOBL _rt0_arm_linux_lib_argc<>(SB),NOPTR,$4
DATA _rt0_arm_linux_lib_argv<>(SB)/4,$0
GLOBL _rt0_arm_linux_lib_argv<>(SB),NOPTR,$4

TEXT _rt0_arm_linux1(SB),NOSPLIT,$-4
	// We first need to detect the kernel ABI, and warn the user
	// if the system only supports OABI.
	// The strategy here is to call some EABI syscall to see if
	// SIGILL is received.
	// If you get a SIGILL here, you have the wrong kernel.

	// Save argc and argv
	MOVM.DB.W [R0-R1], (R13)

	// do an EABI syscall
	MOVW	$20, R7 // sys_getpid
	SWI	$0 // this will trigger SIGILL on OABI systems
	
	B	runtime·rt0_go(SB)

TEXT main(SB),NOSPLIT,$-4
	MOVW	$_rt0_arm_linux1(SB), R4
	B		(R4)

