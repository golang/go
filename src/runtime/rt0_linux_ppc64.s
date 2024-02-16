// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"
#include "asm_ppc64x.h"

DEFINE_PPC64X_FUNCDESC(_rt0_ppc64_linux, _main<>)
DEFINE_PPC64X_FUNCDESC(main, _main<>)

TEXT _main<>(SB),NOSPLIT,$-8
	// In a statically linked binary, the stack contains argc,
	// argv as argc string pointers followed by a NULL, envv as a
	// sequence of string pointers followed by a NULL, and auxv.
	// There is no TLS base pointer.
	//
	// TODO(austin): Support ABI v1 dynamic linking entry point
	XOR	R0, R0 // Note, newer kernels may not always set R0 to 0.
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
