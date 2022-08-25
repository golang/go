// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"
#include "cgo/abi_loong64.h"

TEXT _rt0_loong64_linux(SB),NOSPLIT|NOFRAME,$0
	// In a statically linked binary, the stack contains argc,
	// argv as argc string pointers followed by a NULL, envv as a
	// sequence of string pointers followed by a NULL, and auxv.
	// There is no TLS base pointer.
	MOVW	0(R3), R4	// argc
	ADDV	$8, R3, R5	// argv
	JMP	main(SB)

// When building with -buildmode=c-shared, this symbol is called when the shared
// library is loaded.
TEXT _rt0_loong64_linux_lib(SB),NOSPLIT,$168
	// Preserve callee-save registers.
	SAVE_R22_TO_R31(3*8)
	SAVE_F24_TO_F31(13*8)

	// Initialize g as nil in case of using g later e.g. sigaction in cgo_sigaction.go
	MOVV	R0, g

	MOVV	R4, _rt0_loong64_linux_lib_argc<>(SB)
	MOVV	R5, _rt0_loong64_linux_lib_argv<>(SB)

	// Synchronous initialization.
	MOVV	$runtime·libpreinit(SB), R19
	JAL	(R19)

	// Create a new thread to do the runtime initialization and return.
	MOVV	_cgo_sys_thread_create(SB), R19
	BEQ	R19, nocgo
	MOVV	$_rt0_loong64_linux_lib_go(SB), R4
	MOVV	$0, R5
	JAL	(R19)
	JMP	restore

nocgo:
	MOVV	$0x800000, R4                     // stacksize = 8192KB
	MOVV	$_rt0_loong64_linux_lib_go(SB), R5
	MOVV	R4, 8(R3)
	MOVV	R5, 16(R3)
	MOVV	$runtime·newosproc0(SB), R19
	JAL	(R19)

restore:
	// Restore callee-save registers.
	RESTORE_R22_TO_R31(3*8)
	RESTORE_F24_TO_F31(13*8)
	RET

TEXT _rt0_loong64_linux_lib_go(SB),NOSPLIT,$0
	MOVV	_rt0_loong64_linux_lib_argc<>(SB), R4
	MOVV	_rt0_loong64_linux_lib_argv<>(SB), R5
	MOVV	$runtime·rt0_go(SB),R19
	JMP	(R19)

DATA _rt0_loong64_linux_lib_argc<>(SB)/8, $0
GLOBL _rt0_loong64_linux_lib_argc<>(SB),NOPTR, $8
DATA _rt0_loong64_linux_lib_argv<>(SB)/8, $0
GLOBL _rt0_loong64_linux_lib_argv<>(SB),NOPTR, $8

TEXT main(SB),NOSPLIT|NOFRAME,$0
	// in external linking, glibc jumps to main with argc in R4
	// and argv in R5

	MOVV	$runtime·rt0_go(SB), R19
	JMP	(R19)
