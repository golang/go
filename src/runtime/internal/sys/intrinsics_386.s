// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT runtime∕internal∕sys·TrailingZeros64(SB), NOSPLIT, $0-12
	// Try low 32 bits.
	MOVL	x_lo+0(FP), AX
	BSFL	AX, AX
	JZ	tryhigh
	MOVL	AX, ret+8(FP)
	RET

tryhigh:
	// Try high 32 bits.
	MOVL	x_hi+4(FP), AX
	BSFL	AX, AX
	JZ	none
	ADDL	$32, AX
	MOVL	AX, ret+8(FP)
	RET

none:
	// No bits are set.
	MOVL	$64, ret+8(FP)
	RET

TEXT runtime∕internal∕sys·TrailingZeros32(SB), NOSPLIT, $0-8
	MOVL	x+0(FP), AX
	BSFL	AX, AX
	JNZ	2(PC)
	MOVL	$32, AX
	MOVL	AX, ret+4(FP)
	RET

TEXT runtime∕internal∕sys·TrailingZeros8(SB), NOSPLIT, $0-8
	MOVBLZX	x+0(FP), AX
	BSFL	AX, AX
	JNZ	2(PC)
	MOVL	$8, AX
	MOVL	AX, ret+4(FP)
	RET
