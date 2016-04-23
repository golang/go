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

TEXT _main<>(SB),NOSPLIT,$-8
	// In a statically linked binary, the stack contains argc,
	// argv as argc string pointers followed by a NULL, envv as a
	// sequence of string pointers followed by a NULL, and auxv.
	// There is no TLS base pointer.
#ifdef GOARCH_mips64
	MOVW 4(R29), R1 // argc, big-endian ABI places int32 at offset 4
#else
	MOVW 0(R29), R1 // argc
#endif
	ADDV $8, R29, R2 // argv
	JMP main(SB)

TEXT main(SB),NOSPLIT,$-8
	MOVV	$runtime·rt0_go(SB), R4
	JMP	(R4)
