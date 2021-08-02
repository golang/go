// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && (mips64 || mips64le)
// +build linux
// +build mips64 mips64le

#include "textflag.h"
#include "cgo/abi_mips64x.h"

TEXT _rt0_mips64_linux(SB),NOSPLIT,$0
	JMP	_main<>(SB)

TEXT _rt0_mips64le_linux(SB),NOSPLIT,$0
	JMP	_main<>(SB)

TEXT _main<>(SB),NOSPLIT|NOFRAME,$0
	// In a statically linked binary, the stack contains argc,
	// argv as argc string pointers followed by a NULL, envv as a
	// sequence of string pointers followed by a NULL, and auxv.
	// There is no TLS base pointer.
#ifdef GOARCH_mips64
	MOVW	4(R29), R4 // argc, big-endian ABI places int32 at offset 4
#else
	MOVW	0(R29), R4 // argc
#endif
	ADDV	$8, R29, R5 // argv
	JMP	main(SB)

// When building with -buildmode=c-shared, this symbol is called when the shared
// library is loaded.
#ifdef GOMIPS64_hardfloat
TEXT _rt0_mips64le_linux_lib(SB),NOSPLIT,$232
#else
TEXT _rt0_mips64le_linux_lib(SB),NOSPLIT,$136
#endif
	// Preserve callee-save registers.
	PUSH_REGS_HOST_TO_ABI0()

	// Initialize g as null in case of using g later e.g. sigaction in cgo_sigaction.go
	MOVV	R0, g

	MOVV	R4, _rt0_mips64le_linux_lib_argc<>(SB)
	MOVV	R5, _rt0_mips64le_linux_lib_argv<>(SB)

	// Synchronous initialization.
	MOVV	$runtime·libpreinit(SB), R1
	JAL	(R1)

	// Create a new thread to do the runtime initialization and return.
	MOVV	_cgo_sys_thread_create(SB), R1
	BEQ	R1, nocgo
	MOVV	$_rt0_mips64le_linux_lib_go(SB), R4
	MOVV	$0, R5
	JAL	(R1)
	JMP	restore

nocgo:
	MOVV	$0x800000, R4    // stacksize = 8192KB
	MOVV	$_rt0_mips64le_linux_lib_go(SB), R5
	MOVV	R4, 8(R29)
	MOVV	R5, 16(R29)
	MOVV	$runtime·newosproc0(SB),R1
	JAL	(R1)

restore:
	// Restore callee-save registers.
	POP_REGS_HOST_TO_ABI0()
	RET

TEXT _rt0_mips64le_linux_lib_go(SB),NOSPLIT,$0
	MOVV	_rt0_mips64le_linux_lib_argc<>(SB), R4
	MOVV	_rt0_mips64le_linux_lib_argv<>(SB), R5
	MOVV	$runtime·rt0_go(SB),R1
	JMP     (R1)

DATA _rt0_mips64le_linux_lib_argc<>(SB)/8, $0
GLOBL _rt0_mips64le_linux_lib_argc<>(SB),NOPTR, $8
DATA _rt0_mips64le_linux_lib_argv<>(SB)/8, $0
GLOBL _rt0_mips64le_linux_lib_argv<>(SB),NOPTR, $8

TEXT main(SB),NOSPLIT|NOFRAME,$0
	// in external linking, glibc jumps to main with argc in R4
	// and argv in R5

	// initialize REGSB = PC&0xffffffff00000000
	BGEZAL	R0, 1(PC)
	SRLV	$32, R31, RSB
	SLLV	$32, RSB

	MOVV	$runtime·rt0_go(SB), R1
	JMP	(R1)
