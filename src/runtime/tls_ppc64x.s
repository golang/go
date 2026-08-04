// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ppc64 || ppc64le

#include "go_asm.h"
#include "go_tls.h"
#include "funcdata.h"
#include "textflag.h"

// We have to resort to TLS variable to save g (R30).
// One reason is that external code might trigger
// SIGSEGV, and our runtime.sigtramp don't even know we
// are in external code, and will continue to use R30,
// this might well result in another SIGSEGV.

// save_g saves the g register into pthread-provided
// thread-local memory, so that we can call externally compiled
// ppc64 code that will overwrite this register.
//
// If !iscgo, this is a no-op.
//
// NOTE: setg_gcc<> assume this clobbers only R31.
// With TLS_GD, this also clobbers R18 and LR via the __tls_get_addr call.
// Callers in GD mode must be aware.
TEXT runtime·save_g(SB),NOSPLIT|NOFRAME,$0-0
#ifndef GOOS_aix
#ifndef GOOS_openbsd
	MOVBZ	runtime·iscgo(SB), R31
	CMP	R31, $0
	BEQ	nocgo
#endif
#endif
#ifdef TLS_GD
	// General Dynamic TLS: MOVD runtime·tls_g(SB) generates a GOT
	// access that may call __tls_get_addr, clobbering LR.
	// Save LR in R18 (caller-saved, not used by Go).
	MOVD	LR, R18
	MOVD	runtime·tls_g(SB), R31
	MOVD	g, 0(R31)
	MOVD	R18, LR
#else
	MOVD	runtime·tls_g(SB), R31
	MOVD	g, 0(R31)
#endif

nocgo:
	RET

// load_g loads the g register from pthread-provided
// thread-local memory, for use after calling externally compiled
// ppc64 code that overwrote those registers.
//
// This is never called directly from C code (it doesn't have to
// follow the C ABI), but it may be called from a C context, where the
// usual Go registers aren't set up.
//
// NOTE: _cgo_topofstack assumes this only clobbers g (R30), and R31.
// With TLS_GD, this also clobbers R18 and LR.
TEXT runtime·load_g(SB),NOSPLIT|NOFRAME,$0-0
#ifdef TLS_GD
	MOVD	LR, R18
	MOVD	runtime·tls_g(SB), R31
	MOVD	0(R31), g
	MOVD	R18, LR
#else
	MOVD	runtime·tls_g(SB), R31
	MOVD	0(R31), g
#endif
	RET

GLOBL runtime·tls_g+0(SB), TLSBSS+DUPOK, $8
