// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

/*
 * void crosscall2(void (*fn)(void*, int32), void*, int32)
 * Save registers and call fn with two arguments.
 */
TEXT crosscall2(SB),NOSPLIT,$-4
	/* 
	 * We still need to save all callee save register as before, and then
	 *  push 2 args for fn (R1 and R2).
	 * Also note that at procedure entry in gc world, 4(R13) will be the
	 *  first arg, so we must push another dummy reg (R0) for 0(R13).
	 *  Additionally, runtime·load_g will clobber R0, so we need to save R0
	 *  nevertheless.
	 */
	MOVM.WP	[R0, R1, R2, R4, R5, R6, R7, R8, R9, g, R11, R12, R14], (R13)
	BL	runtime·load_g(SB)
	MOVW	R15, R14 // R15 is PC.
	MOVW	0(R13), R15
	MOVM.IAW	(R13), [R0, R1, R2, R4, R5, R6, R7, R8, R9, g, R11, R12, R15]
