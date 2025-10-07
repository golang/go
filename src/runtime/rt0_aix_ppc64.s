// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"
#include "asm_ppc64x.h"

// _rt0_ppc64_aix is a function descriptor of the entrypoint function
// __start. This name is needed by cmd/link.
DEFINE_PPC64X_FUNCDESC(_rt0_ppc64_aix, __start<>)

// The starting function must return in the loader to
// initialise some libraries, especially libthread which
// creates the main thread and adds the TLS in R13
// R19 contains a function descriptor to the loader function
// which needs to be called.
// This code is similar to the __start function in C
TEXT __start<>(SB),NOSPLIT,$-8
	XOR R0, R0
	MOVD $libc___n_pthreads(SB), R4
	MOVD 0(R4), R4
	MOVD $libc___mod_init(SB), R5
	MOVD 0(R5), R5
	MOVD 0(R19), R0
	MOVD R2, 40(R1)
	MOVD 8(R19), R2
	MOVD R18, R3
	MOVD R0, CTR
	BL (CTR) // Return to AIX loader

	// Launch rt0_go
	MOVD 40(R1), R2
	MOVD R14, R3 // argc
	MOVD R15, R4 // argv
	BL _main(SB)


DEFINE_PPC64X_FUNCDESC(main, _main)
TEXT _main(SB),NOSPLIT,$-8
	MOVD $runtime·rt0_go(SB), R12
	MOVD R12, CTR
	BR (CTR)

TEXT _rt0_ppc64_aix_lib(SB),NOSPLIT,$0
	MOVD R14, R3 // argc
	MOVD R15, R4 // argv
	JMP _rt0_ppc64x_lib(SB)
