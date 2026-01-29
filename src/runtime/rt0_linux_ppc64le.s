// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"
#include "asm_ppc64x.h"

TEXT _rt0_ppc64le_linux(SB),NOSPLIT,$0
	XOR R0, R0	  // Make sure R0 is zero before _main
	BR _main<>(SB)

TEXT _rt0_ppc64le_linux_lib(SB),NOSPLIT|NOFRAME,$0
	JMP _rt0_ppc64x_lib(SB)

TEXT _main<>(SB),NOSPLIT,$-8
	// In a statically linked binary, the stack contains argc,
	// argv as argc string pointers followed by a NULL, envv as a
	// sequence of string pointers followed by a NULL, and auxv.
	// The TLS pointer should be initialized to 0.
	//
	// In an ELFv2 compliant dynamically linked binary, R3 contains argc,
	// R4 contains argv, R5 contains envp, R6 contains auxv, and R13
	// contains the TLS pointer.
	//
	// When loading via glibc, the first doubleword on the stack points
	// to NULL a value. (that is *(uintptr)(R1) == 0). This is used to
	// differentiate static vs dynamically linked binaries.
	//
	// If loading with the musl loader, it doesn't follow the ELFv2 ABI. It
	// passes argc/argv similar to the linux kernel, R13 (TLS) is
	// initialized, and R3/R4 are undefined.
	MOVD	(R1), R12
	CMP	R12, $0
	BEQ	tls_and_argcv_in_reg

	// Arguments are passed via the stack (musl loader or a static binary)
	MOVD	0(R1), R3 // argc
	ADD	$8, R1, R4 // argv

	// Did the TLS pointer get set? If so, don't change it (e.g musl).
	CMP	R13, $0
	BNE	tls_and_argcv_in_reg

	MOVD	$runtime·m0+m_tls(SB), R13 // TLS
	ADD	$0x7000, R13

tls_and_argcv_in_reg:
	BR	main(SB)

TEXT main(SB),NOSPLIT,$-8
	MOVD	$runtime·rt0_go(SB), R12
	MOVD	R12, CTR
	BR	(CTR)
