// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "go_tls.h"
#include "funcdata.h"
#include "textflag.h"

// If !iscgo, this is a no-op.
//
// NOTE: mcall() assumes this clobbers only X31 (REG_TMP).
// With TLS_GD, this also clobbers X10 (A0), X11 (A1), X1 (RA)
// via the __tls_get_addr call. Callers in GD mode must be aware.
TEXT runtime·save_g(SB),NOSPLIT|NOFRAME,$0-0
#ifndef GOOS_openbsd
	MOVB	runtime·iscgo(SB), X31
	BEQZ	X31, nocgo
#endif
#ifdef TLS_GD
	// General Dynamic TLS: MOV g, runtime·tls_g(SB) generates
	// AUIPC+ADDI (GD GOT addr) + AUIPC+JALR (__tls_get_addr call) + SD.
	// __tls_get_addr returns the variable address in A0.
	// Clobbers: A0, RA, and caller-saved regs per C ABI.
	// Save RA (X1) since JALR overwrites it.
	MOV	X1, X5
	MOV	g, runtime·tls_g(SB)
	MOV	X5, X1
#else
	MOV	g, runtime·tls_g(SB)
#endif
nocgo:
	RET

TEXT runtime·load_g(SB),NOSPLIT|NOFRAME,$0-0
#ifdef TLS_GD
	MOV	X1, X5
	MOV	runtime·tls_g(SB), g
	MOV	X5, X1
#else
	MOV	runtime·tls_g(SB), g
#endif
	RET

GLOBL runtime·tls_g(SB), TLSBSS, $8
