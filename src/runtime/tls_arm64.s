// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "go_tls.h"
#include "funcdata.h"
#include "textflag.h"
#include "tls_arm64.h"

TEXT runtime·load_g(SB),NOSPLIT,$0
	MOVB	runtime·iscgo(SB), R0
	CMP	$0, R0
	BEQ	nocgo

	MRS_TPIDR_R0
	MOVD	0x10(R0), g

nocgo:
	RET

TEXT runtime·save_g(SB),NOSPLIT,$0
	MOVB	runtime·iscgo(SB), R0
	CMP	$0, R0
	BEQ	nocgo

	MRS_TPIDR_R0
	MOVD	g, 0x10(R0)

nocgo:
	RET
