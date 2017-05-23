// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build mips64 mips64le

#include "go_asm.h"
#include "go_tls.h"
#include "funcdata.h"
#include "textflag.h"

// If !iscgo, this is a no-op.
//
// NOTE: mcall() assumes this clobbers only R23 (REGTMP).
TEXT runtime·save_g(SB),NOSPLIT,$-8-0
	MOVB	runtime·iscgo(SB), R23
	BEQ	R23, nocgo

	MOVV	R3, R23	// save R3
	MOVV	g, runtime·tls_g(SB) // TLS relocation clobbers R3
	MOVV	R23, R3	// restore R3

nocgo:
	RET

TEXT runtime·load_g(SB),NOSPLIT,$-8-0
	MOVV	runtime·tls_g(SB), g // TLS relocation clobbers R3
	RET

GLOBL runtime·tls_g(SB), TLSBSS, $8
