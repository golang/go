// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT _rt0_arm64_linux(SB),NOSPLIT|NOFRAME,$0
	MOVD	0(RSP), R0	// argc
	ADD	$8, RSP, R1	// argv
	BL	main(SB)

// When building with -buildmode=c-shared, this symbol is called when the shared
// library is loaded.
TEXT _rt0_arm64_linux_lib(SB),NOSPLIT,$184
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
	FMOVD F8, 96(RSP)
	FMOVD F9, 104(RSP)
	FMOVD F10, 112(RSP)
	FMOVD F11, 120(RSP)
	FMOVD F12, 128(RSP)
	FMOVD F13, 136(RSP)
	FMOVD F14, 144(RSP)
	FMOVD F15, 152(RSP)
	MOVD g, 160(RSP)

	// Initialize g as null in case of using g later e.g. sigaction in cgo_sigaction.go
	MOVD	ZR, g

	MOVD	R0, _rt0_arm64_linux_lib_argc<>(SB)
	MOVD	R1, _rt0_arm64_linux_lib_argv<>(SB)

	// Synchronous initialization.
	MOVD	$runtime·libpreinit(SB), R4
	BL	(R4)

	// Create a new thread to do the runtime initialization and return.
	MOVD	_cgo_sys_thread_create(SB), R4
	CBZ	R4, nocgo
	MOVD	$_rt0_arm64_linux_lib_go(SB), R0
	MOVD	$0, R1
	SUB	$16, RSP		// reserve 16 bytes for sp-8 where fp may be saved.
	BL	(R4)
	ADD	$16, RSP
	B	restore

nocgo:
	MOVD	$0x800000, R0                     // stacksize = 8192KB
	MOVD	$_rt0_arm64_linux_lib_go(SB), R1
	MOVD	R0, 8(RSP)
	MOVD	R1, 16(RSP)
	MOVD	$runtime·newosproc0(SB),R4
	BL	(R4)

restore:
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
	FMOVD 96(RSP), F8
	FMOVD 104(RSP), F9
	FMOVD 112(RSP), F10
	FMOVD 120(RSP), F11
	FMOVD 128(RSP), F12
	FMOVD 136(RSP), F13
	FMOVD 144(RSP), F14
	FMOVD 152(RSP), F15
	MOVD 160(RSP), g
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


TEXT main(SB),NOSPLIT|NOFRAME,$0
	MOVD	$runtime·rt0_go(SB), R2
	BL	(R2)
exit:
	MOVD $0, R0
	MOVD	$94, R8	// sys_exit
	SVC
	B	exit
