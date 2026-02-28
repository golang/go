// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// actually a function descriptor for _main<>(SB)
TEXT _rt0_ppc64_linux(SB),NOSPLIT,$0
	DWORD $_main<>(SB)
	DWORD $0
	DWORD $0

TEXT main(SB),NOSPLIT,$0
	DWORD $_main<>(SB)
	DWORD $0
	DWORD $0

TEXT _main<>(SB),NOSPLIT,$-8
	// In a statically linked binary, the stack contains argc,
	// argv as argc string pointers followed by a NULL, envv as a
	// sequence of string pointers followed by a NULL, and auxv.
	// There is no TLS base pointer.
	//
	// TODO(austin): Support ABI v1 dynamic linking entry point
	MOVD	$runtime·rt0_go(SB), R12
	MOVD	R12, CTR
	MOVBZ	runtime·iscgo(SB), R5
	CMP	R5, $0
	BEQ	nocgo
	BR	(CTR)
nocgo:
	MOVD	0(R1), R3 // argc
	ADD	$8, R1, R4 // argv
	BR	(CTR)
