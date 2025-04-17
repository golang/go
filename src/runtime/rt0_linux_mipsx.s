// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && (mips || mipsle)

#include "textflag.h"

TEXT _rt0_mips_linux(SB),NOSPLIT,$0
	JMP	_main<>(SB)

TEXT _rt0_mipsle_linux(SB),NOSPLIT,$0
	JMP	_main<>(SB)

TEXT _main<>(SB),NOSPLIT|NOFRAME,$0
	// In a statically linked binary, the stack contains argc,
	// argv as argc string pointers followed by a NULL, envv as a
	// sequence of string pointers followed by a NULL, and auxv.
	// There is no TLS base pointer.
	MOVW	0(R29), R4 // argc
	ADD	$4, R29, R5 // argv
	JMP	main(SB)

TEXT main(SB),NOSPLIT|NOFRAME,$0
	// In external linking, libc jumps to main with argc in R4, argv in R5
	MOVW	$runtime·rt0_go(SB), R1
	JMP	(R1)
