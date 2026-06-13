// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "go_tls.h"
#include "funcdata.h"
#include "textflag.h"

// We have to resort to TLS variable to save g (R13).
// One reason is that external code might trigger
// SIGSEGV, and our runtime.sigtramp don't even know we
// are in external code, and will continue to use R13,
// this might well result in another SIGSEGV.

// save_g saves the g register into pthread-provided
// thread-local memory, so that we can call externally compiled
// s390x code that will overwrite this register.
//
// If !iscgo, this is a no-op.
//
// NOTE: setg_gcc<> and mcall assume this clobbers only R10 and R11.
// With TLS_GD, this also clobbers R12 and R14 (LR) via the __tls_get_addr call.
// Callers in GD mode must be aware.
TEXT runtime·save_g(SB),NOSPLIT|NOFRAME,$0-0
	MOVB	runtime·iscgo(SB),  R10
	CMPBEQ	R10, $0, nocgo
	MOVW	AR0, R11
	SLD	$32, R11
	MOVW	AR1, R11
#ifdef TLS_GD
	// General Dynamic TLS: MOVD runtime·tls_g(SB) generates a GOT
	// access that may call __tls_get_addr, clobbering R14 (LR).
	// Save R14 in R12 (caller-saved, not used by Go runtime here).
	MOVD	R14, R12
	MOVD	runtime·tls_g(SB), R10
	MOVD	g, 0(R10)(R11*1)
	MOVD	R12, R14
#else
	MOVD	runtime·tls_g(SB), R10
	MOVD	g, 0(R10)(R11*1)
#endif
nocgo:
	RET

// load_g loads the g register from pthread-provided
// thread-local memory, for use after calling externally compiled
// s390x code that overwrote those registers.
//
// This is never called directly from C code (it doesn't have to
// follow the C ABI), but it may be called from a C context, where the
// usual Go registers aren't set up.
//
// NOTE: _cgo_topofstack assumes this only clobbers g (R13), R10 and R11.
// With TLS_GD, this also clobbers R12 and R14 (LR).
TEXT runtime·load_g(SB),NOSPLIT|NOFRAME,$0-0
	MOVW	AR0, R11
	SLD	$32, R11
	MOVW	AR1, R11
#ifdef TLS_GD
	MOVD	R14, R12
	MOVD	runtime·tls_g(SB), R10
	MOVD	0(R10)(R11*1), g
	MOVD	R12, R14
#else
	MOVD	runtime·tls_g(SB), R10
	MOVD	0(R10)(R11*1), g
#endif
	RET

GLOBL runtime·tls_g+0(SB),TLSBSS,$8
