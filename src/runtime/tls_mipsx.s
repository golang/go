// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build mips mipsle

#include "go_asm.h"
#include "go_tls.h"
#include "funcdata.h"
#include "textflag.h"

// If !iscgo, this is a no-op.
TEXT runtime·save_g(SB),NOSPLIT,$-4-0
	MOVB	runtime·iscgo(SB), R23
	BEQ	R23, nocgo
	UNDEF
nocgo:
	RET

TEXT runtime·load_g(SB),NOSPLIT,$-4-0
	RET
