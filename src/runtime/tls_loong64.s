// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "go_tls.h"
#include "funcdata.h"
#include "textflag.h"

// If !iscgo, this is a no-op.
//
// NOTE: mcall() assumes this clobbers only R30 (REGTMP).
// With TLS_GD, this also clobbers R12 and R1 (RA/LR) via the __tls_get_addr call.
// Callers in GD mode must be aware.
TEXT runtime·save_g(SB),NOSPLIT|NOFRAME,$0-0
	MOVB	runtime·iscgo(SB), R30
	BEQ	R30, nocgo

#ifdef TLS_GD
	// General Dynamic TLS: MOVV g, runtime·tls_g(SB) generates a GOT
	// access that may call __tls_get_addr, clobbering R1 (RA/LR).
	// Save R1 in R12 (caller-saved, not used by Go runtime here).
	MOVV	R1, R12
	MOVV	g, runtime·tls_g(SB)
	MOVV	R12, R1
#else
	MOVV	g, runtime·tls_g(SB)
#endif

nocgo:
	RET

TEXT runtime·load_g(SB),NOSPLIT|NOFRAME,$0-0
#ifdef TLS_GD
	MOVV	R1, R12
	MOVV	runtime·tls_g(SB), g
	MOVV	R12, R1
#else
	MOVV	runtime·tls_g(SB), g
#endif
	RET

GLOBL runtime·tls_g(SB), TLSBSS, $8
