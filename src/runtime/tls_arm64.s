// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "go_tls.h"
#include "funcdata.h"
#include "textflag.h"
#include "tls_arm64.h"

TEXT runtime·load_g(SB),NOSPLIT,$0
#ifndef GOOS_darwin
#ifndef GOOS_openbsd
#ifndef GOOS_windows
	MOVB	runtime·iscgo(SB), R0
	CBZ	R0, nocgo
#endif
#endif
#endif

#ifdef TLS_GD
	// General Dynamic TLS model (TLSDESC).
	// The MOVD below generates: adrp x0; ldr x1; add x0; blr x1
	// TLSDESC ABI clobbers X0, X1, and LR (X30).
	// Callers expect save_g/load_g to only clobber R0 and R27,
	// so we save R1 and LR to scratch registers R10 and R11.
	MOVD	R1, R10
	MOVD	R30, R11
	MOVD	runtime·tls_g(SB), R0
	MOVD	R10, R1
	MOVD	R11, R30
	MOVD	R0, R27
	MRS_TPIDR_R0
	MOVD	(R0)(R27), g
#else
	MRS_TPIDR_R0
#ifdef TLS_darwin
	// Darwin sometimes returns unaligned pointers
	AND	$0xfffffffffffffff8, R0
#endif
	MOVD	runtime·tls_g(SB), R27
	MOVD	(R0)(R27), g
#endif

nocgo:
	RET

TEXT runtime·save_g(SB),NOSPLIT,$0
#ifndef GOOS_darwin
#ifndef GOOS_openbsd
#ifndef GOOS_windows
	MOVB	runtime·iscgo(SB), R0
	CBZ	R0, nocgo
#endif
#endif
#endif

#ifdef TLS_GD
	// General Dynamic TLS model (TLSDESC).
	// Save R1 and LR — callers may have live values in these regs.
	MOVD	R1, R10
	MOVD	R30, R11
	MOVD	runtime·tls_g(SB), R0
	MOVD	R10, R1
	MOVD	R11, R30
	MOVD	R0, R27
	MRS_TPIDR_R0
	MOVD	g, (R0)(R27)
#else
	MRS_TPIDR_R0
#ifdef TLS_darwin
	// Darwin sometimes returns unaligned pointers
	AND	$0xfffffffffffffff8, R0
#endif
	MOVD	runtime·tls_g(SB), R27
	MOVD	g, (R0)(R27)
#endif

nocgo:
	RET

#ifdef TLSG_IS_VARIABLE
#ifdef GOOS_android
// Use the free TLS_SLOT_APP slot #2 on Android Q.
// Earlier androids are set up in gcc_android.c.
DATA runtime·tls_g+0(SB)/8, $16
#endif
GLOBL runtime·tls_g+0(SB), NOPTR, $8
#else
GLOBL runtime·tls_g+0(SB), TLSBSS, $8
#endif
