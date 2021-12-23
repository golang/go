// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"
#include "cgo/abi_arm64.h"

// On FreeBSD argc/argv are passed in R0, not RSP
TEXT _rt0_arm64_freebsd(SB),NOSPLIT|NOFRAME,$0
	ADD	$8, R0, R1	// argv
	MOVD	0(R0), R0	// argc
	BL	main(SB)

// When building with -buildmode=c-shared, this symbol is called when the shared
// library is loaded.
TEXT _rt0_arm64_freebsd_lib(SB),NOSPLIT,$184
	// Preserve callee-save registers.
	SAVE_R19_TO_R28(24)
	SAVE_F8_TO_F15(104)

	// Initialize g as null in case of using g later e.g. sigaction in cgo_sigaction.go
	MOVD	ZR, g

	MOVD	R0, _rt0_arm64_freebsd_lib_argc<>(SB)
	MOVD	R1, _rt0_arm64_freebsd_lib_argv<>(SB)

	// Synchronous initialization.
	MOVD	$runtime·libpreinit(SB), R4
	BL	(R4)

	// Create a new thread to do the runtime initialization and return.
	MOVD	_cgo_sys_thread_create(SB), R4
	CBZ	R4, nocgo
	MOVD	$_rt0_arm64_freebsd_lib_go(SB), R0
	MOVD	$0, R1
	SUB	$16, RSP	// reserve 16 bytes for sp-8 where fp may be saved.
	BL	(R4)
	ADD	$16, RSP
	B	restore

nocgo:
	MOVD	$0x800000, R0                     // stacksize = 8192KB
	MOVD	$_rt0_arm64_freebsd_lib_go(SB), R1
	MOVD	R0, 8(RSP)
	MOVD	R1, 16(RSP)
	MOVD	$runtime·newosproc0(SB),R4
	BL	(R4)

restore:
	// Restore callee-save registers.
	RESTORE_R19_TO_R28(24)
	RESTORE_F8_TO_F15(104)
	RET

TEXT _rt0_arm64_freebsd_lib_go(SB),NOSPLIT,$0
	MOVD	_rt0_arm64_freebsd_lib_argc<>(SB), R0
	MOVD	_rt0_arm64_freebsd_lib_argv<>(SB), R1
	MOVD	$runtime·rt0_go(SB),R4
	B       (R4)

DATA _rt0_arm64_freebsd_lib_argc<>(SB)/8, $0
GLOBL _rt0_arm64_freebsd_lib_argc<>(SB),NOPTR, $8
DATA _rt0_arm64_freebsd_lib_argv<>(SB)/8, $0
GLOBL _rt0_arm64_freebsd_lib_argv<>(SB),NOPTR, $8


TEXT main(SB),NOSPLIT|NOFRAME,$0
	MOVD	$runtime·rt0_go(SB), R2
	BL	(R2)
exit:
	MOVD	$0, R0
	MOVD	$1, R8	// SYS_exit
	SVC
	B	exit
