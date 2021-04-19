// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build mips mipsle

#include "go_asm.h"
#include "go_tls.h"
#include "funcdata.h"
#include "textflag.h"

// If !iscgo, this is a no-op.
// NOTE: gogo asumes load_g only clobers g (R30) and REGTMP (R23)
TEXT runtime·save_g(SB),NOSPLIT,$-4-0
	MOVB	runtime·iscgo(SB), R23
	BEQ	R23, nocgo

	MOVW	R3, R23
	MOVW	g, runtime·tls_g(SB) // TLS relocation clobbers R3
	MOVW	R23, R3

nocgo:
	RET

TEXT runtime·load_g(SB),NOSPLIT,$-4-0
	MOVW	runtime·tls_g(SB), g // TLS relocation clobbers R3
	RET

GLOBL runtime·tls_g(SB), TLSBSS, $4
