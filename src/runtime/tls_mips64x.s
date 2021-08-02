// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build mips64 || mips64le
// +build mips64 mips64le

#include "go_asm.h"
#include "go_tls.h"
#include "funcdata.h"
#include "textflag.h"

// If !iscgo, this is a no-op.
//
// NOTE: mcall() assumes this clobbers only R23 (REGTMP).
TEXT runtime·save_g(SB),NOSPLIT|NOFRAME,$0-0
	MOVB	runtime·iscgo(SB), R23
	BEQ	R23, nocgo

	MOVV	R31,-32(R29)
	ADDV	$-32, R29
	MOVV	R2, 8(R29)
	MOVV	R4, 16(R29)
	MOVV	R25, 24(R29)
	MOVV	R3, R23	// save R3
	// TLS relocation clobbers R2,R4,R25 when buildmode=c-shared
	// TLS relocation clobbers R3 when buildmode=exe
	MOVV	g, runtime·tls_g(SB)
	MOVV	R23, R3	// restore R3
	MOVV	8(R29),R2
	MOVV	16(R29),R4
	MOVV	24(R29),R25
	MOVV	0(R29), R31
	ADDV	$32, R29

nocgo:
	RET

TEXT runtime·load_g(SB),NOSPLIT|NOFRAME,$0-0
	MOVV	R31,-32(R29)
	ADDV	$-32, R29
	MOVV	R2, 8(R29)
	MOVV	R4, 16(R29)
	MOVV	R25, 24(R29)
	// TLS relocation clobbers R2,R4,R25 when buildmode=c-shared
	// TLS relocation clobbers R3 when buildmode=exe
	MOVV	runtime·tls_g(SB), g
	MOVV	8(R29),R2
	MOVV	16(R29),R4
	MOVV	24(R29),R25
	MOVV	0(R29), R31
	ADDV	$32, R29
	RET

GLOBL runtime·tls_g(SB), TLSBSS, $8
