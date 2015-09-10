// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build mips64 mips64le

#include "go_asm.h"
#include "go_tls.h"
#include "funcdata.h"
#include "textflag.h"

// If !iscgo, this is a no-op.
//
// NOTE: mcall() assumes this clobbers only R28 (REGTMP).
TEXT runtime·save_g(SB),NOSPLIT,$-8-0
	MOVB	runtime·iscgo(SB), R28
	BEQ	R28, nocgo

nocgo:
	RET

TEXT runtime·load_g(SB),NOSPLIT,$-8-0
	RET
