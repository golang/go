// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

TEXT ·IndexByte<ABIInternal>(SB),NOSPLIT,$0-40
#ifndef GOEXPERIMENT_regabiargs
	MOVV	b_base+0(FP), R4
	MOVV	b_len+8(FP), R5
	MOVBU	c+24(FP), R7	// byte to find
#endif
	// R4 = b_base
	// R5 = b_len
	// R6 = b_cap (unused)
	// R7 = byte to find
	AND	$0xff, R7
	MOVV	R4, R6		// store base for later
	ADDV	R4, R5		// end
	ADDV	$-1, R4

	PCALIGN	$16
loop:
	ADDV	$1, R4
	BEQ	R4, R5, notfound
	MOVBU	(R4), R8
	BNE	R7, R8, loop

	SUBV	R6, R4		// remove base
#ifndef GOEXPERIMENT_regabiargs
	MOVV	R4, ret+32(FP)
#endif
	RET

notfound:
	MOVV	$-1, R4
#ifndef GOEXPERIMENT_regabiargs
	MOVV	R4, ret+32(FP)
#endif
	RET

TEXT ·IndexByteString<ABIInternal>(SB),NOSPLIT,$0-32
#ifndef GOEXPERIMENT_regabiargs
	MOVV	s_base+0(FP), R4
	MOVV	s_len+8(FP), R5
	MOVBU	c+16(FP), R6	// byte to find
#endif
	// R4 = s_base
	// R5 = s_len
	// R6 = byte to find
	MOVV	R4, R7		// store base for later
	ADDV	R4, R5		// end
	ADDV	$-1, R4

	PCALIGN	$16
loop:
	ADDV	$1, R4
	BEQ	R4, R5, notfound
	MOVBU	(R4), R8
	BNE	R6, R8, loop

	SUBV	R7, R4		// remove base
#ifndef GOEXPERIMENT_regabiargs
	MOVV	R4, ret+24(FP)
#endif
	RET

notfound:
	MOVV	$-1, R4
#ifndef GOEXPERIMENT_regabiargs
	MOVV	R4, ret+24(FP)
#endif
	RET
