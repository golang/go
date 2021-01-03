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
TEXT runtime·save_g(SB),NOSPLIT|NOFRAME,$0-0
	MOVB	runtime·iscgo(SB), X31
	BEQ	X0, X31, nocgo

	MOV	runtime·tls_g(SB), X31
	ADD	X4, X31		// add offset to thread pointer (X4)
	MOV	g, (X31)

nocgo:
	RET

TEXT runtime·load_g(SB),NOSPLIT|NOFRAME,$0-0
	MOV	runtime·tls_g(SB), X31
	ADD	X4, X31		// add offset to thread pointer (X4)
	MOV	(X31), g
	RET

GLOBL runtime·tls_g(SB), TLSBSS, $8
