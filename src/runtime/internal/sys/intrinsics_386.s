// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT runtime∕internal∕sys·Ctz64(SB), NOSPLIT, $0-16
	MOVL	$0, ret_hi+12(FP)

	// Try low 32 bits.
	MOVL	x_lo+0(FP), AX
	BSFL	AX, AX
	JZ	tryhigh
	MOVL	AX, ret_lo+8(FP)
	RET

tryhigh:
	// Try high 32 bits.
	MOVL	x_hi+4(FP), AX
	BSFL	AX, AX
	JZ	none
	ADDL	$32, AX
	MOVL	AX, ret_lo+8(FP)
	RET

none:
	// No bits are set.
	MOVL	$64, ret_lo+8(FP)
	RET

TEXT runtime∕internal∕sys·Ctz32(SB), NOSPLIT, $0-8
	MOVL	x+0(FP), AX
	BSFL	AX, AX
	JNZ	2(PC)
	MOVL	$32, AX
	MOVL	AX, ret+4(FP)
	RET

TEXT runtime∕internal∕sys·Ctz16(SB), NOSPLIT, $0-6
	MOVW	x+0(FP), AX
	BSFW	AX, AX
	JNZ	2(PC)
	MOVW	$16, AX
	MOVW	AX, ret+4(FP)
	RET

TEXT runtime∕internal∕sys·Ctz8(SB), NOSPLIT, $0-5
	MOVBLZX	x+0(FP), AX
	BSFL	AX, AX
	JNZ	2(PC)
	MOVB	$8, AX
	MOVB	AX, ret+4(FP)
	RET

TEXT runtime∕internal∕sys·Bswap64(SB), NOSPLIT, $0-16
	MOVL	x_lo+0(FP), AX
	MOVL	x_hi+4(FP), BX
	BSWAPL	AX
	BSWAPL	BX
	MOVL	BX, ret_lo+8(FP)
	MOVL	AX, ret_hi+12(FP)
	RET

TEXT runtime∕internal∕sys·Bswap32(SB), NOSPLIT, $0-8
	MOVL	x+0(FP), AX
	BSWAPL	AX
	MOVL	AX, ret+4(FP)
	RET
