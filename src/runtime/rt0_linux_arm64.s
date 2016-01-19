// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT _rt0_arm64_linux(SB),NOSPLIT,$-8
	MOVD	0(RSP), R0	// argc
	ADD	$8, RSP, R1	// argv
	BL	main(SB)

// When building with -buildmode=c-shared, this symbol is called when the shared
// library is loaded.
TEXT _rt0_arm64_linux_lib(SB),NOSPLIT,$40
	// R27 is REGTMP, reserved for liblink. It is used below to
	// move R0/R1 into globals. However in the standard ARM64 calling
	// convention, it is a callee-saved register.
	MOVD	R27, 24(RSP)

	MOVD	R0, _rt0_arm64_linux_lib_argc<>(SB)
	MOVD	R1, _rt0_arm64_linux_lib_argv<>(SB)

	// Synchronous initialization.
	MOVD	$runtime·libpreinit(SB), R4
	BL	(R4)

	// Create a new thread to do the runtime initialization and return.
	MOVD	_cgo_sys_thread_create(SB), R4
	CMP	$0, R4
	BEQ	nocgo
	MOVD	$_rt0_arm64_linux_lib_go(SB), R0
	MOVD	$0, R1
	BL	(R4)
	B	restore

nocgo:
	MOVD	$0x800000, R0                     // stacksize = 8192KB
	MOVD	$_rt0_arm64_linux_lib_go(SB), R1
	MOVD	R0, 8(RSP)
	MOVD	R1, 16(RSP)
	MOVD	$runtime·newosproc0(SB),R4
	BL	(R4)

restore:
	MOVD	24(RSP), R27
	RET

TEXT _rt0_arm64_linux_lib_go(SB),NOSPLIT,$0
	MOVD	_rt0_arm64_linux_lib_argc<>(SB), R0
	MOVD	_rt0_arm64_linux_lib_argv<>(SB), R1
	MOVD	$runtime·rt0_go(SB),R4
	B       (R4)

DATA _rt0_arm64_linux_lib_argc<>(SB)/8, $0
GLOBL _rt0_arm64_linux_lib_argc<>(SB),NOPTR, $8
DATA _rt0_arm64_linux_lib_argv<>(SB)/8, $0
GLOBL _rt0_arm64_linux_lib_argv<>(SB),NOPTR, $8


TEXT main(SB),NOSPLIT,$-8
	MOVD	$runtime·rt0_go(SB), R2
	BL	(R2)
exit:
	MOVD $0, R0
	MOVD	$94, R8	// sys_exit
	SVC
	B	exit
