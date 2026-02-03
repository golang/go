// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

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
TEXT _rt0_loong64_linux_lib(SB),NOSPLIT,$0
	JMP	_rt0_loong64_lib(SB)

TEXT main(SB),NOSPLIT|NOFRAME,$0
	// in external linking, glibc jumps to main with argc in R4
	// and argv in R5

	MOVV	$runtime·rt0_go(SB), R19
	JMP	(R19)
