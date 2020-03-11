// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT ·clflush(SB),NOSPLIT,$0-8
	MOVQ arg+0(FP), AX
	CLFLUSH 0(AX)
	RET

TEXT ·rdtscp(SB),NOSPLIT,$0-8
	RDTSCP
	SHLQ $32, DX
	ORQ DX, AX
	MOVQ AX, ret+0(FP)
	RET

TEXT ·nop(SB),NOSPLIT,$0-0
	RET

TEXT ·cpuid(SB),NOSPLIT,$0-0
	CPUID
	RET

TEXT ·features(SB),NOSPLIT,$0-2
	MOVL $0, AX
	MOVL $0, CX
	CPUID
	CMPL AX, $1
	JLT none

	MOVL $1, AX
	MOVL $0, CX
	CPUID
	SHRL $19, DX
	ANDL $1, DX
	MOVB DX, hasCLFLUSH+0(FP)

	MOVL $0x80000001, AX
	MOVL $0, CX
	CPUID
	SHRL $27, DX
	ANDL $1, DX
	MOVB DX, hasRDTSCP+0(FP)
	RET

none:
	MOVB $0, hasCLFLUSH+0(FP)
	MOVB $0, hasRDTSCP+1(FP)
	RET
