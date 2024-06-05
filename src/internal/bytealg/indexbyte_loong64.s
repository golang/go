// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

TEXT ·IndexByte<ABIInternal>(SB),NOSPLIT,$0-40
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
	RET

notfound:
	MOVV	$-1, R4
	RET

TEXT ·IndexByteString<ABIInternal>(SB),NOSPLIT,$0-32
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
	RET

notfound:
	MOVV	$-1, R4
	RET
