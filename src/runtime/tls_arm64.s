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
#ifdef GOOS_darwin
	// Darwin sometimes returns unaligned pointers
	AND	$0xfffffffffffffff8, R0
#endif
#ifdef TLSG_IS_VARIABLE
	MOVD	runtime·tls_g(SB), R27
	ADD	R27, R0
#else
	// TODO(minux): use real TLS relocation, instead of hard-code for Linux
	ADD	$0x10, R0
#endif
	MOVD	0(R0), g

nocgo:
	RET

TEXT runtime·save_g(SB),NOSPLIT,$0
	MOVB	runtime·iscgo(SB), R0
	CMP	$0, R0
	BEQ	nocgo

	MRS_TPIDR_R0
#ifdef GOOS_darwin
	// Darwin sometimes returns unaligned pointers
	AND	$0xfffffffffffffff8, R0
#endif
#ifdef TLSG_IS_VARIABLE
	MOVD	runtime·tls_g(SB), R27
	ADD	R27, R0
#else
	// TODO(minux): use real TLS relocation, instead of hard-code for Linux
	ADD	$0x10, R0
#endif
	MOVD	g, 0(R0)

nocgo:
	RET

#ifdef TLSG_IS_VARIABLE
// The runtime.tlsg name is being handled specially in the
// linker. As we just need a regular variable here, don't
// use that name.
GLOBL runtime·tls_g+0(SB), NOPTR, $8
#endif
