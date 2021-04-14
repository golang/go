// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux
// +build mips64 mips64le

#include "textflag.h"

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

TEXT main(SB),NOSPLIT|NOFRAME,$0
	// in external linking, glibc jumps to main with argc in R4
	// and argv in R5

	// initialize REGSB = PC&0xffffffff00000000
	BGEZAL	R0, 1(PC)
	SRLV	$32, R31, RSB
	SLLV	$32, RSB

	MOVV	$runtime·rt0_go(SB), R1
	JMP	(R1)
