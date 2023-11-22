// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

TEXT ·Count<ABIInternal>(SB),NOSPLIT,$0-40
	// X10 = b_base
	// X11 = b_len
	// X12 = b_cap (unused)
	// X13 = byte to count (want in X12)
	AND	$0xff, X13, X12
	MOV	ZERO, X14	// count
	ADD	X10, X11	// end

	PCALIGN	$16
loop:
	BEQ	X10, X11, done
	MOVBU	(X10), X15
	ADD	$1, X10
	BNE	X12, X15, loop
	ADD	$1, X14
	JMP	loop

done:
	MOV	X14, X10
	RET

TEXT ·CountString<ABIInternal>(SB),NOSPLIT,$0-32
	// X10 = s_base
	// X11 = s_len
	// X12 = byte to count
	AND	$0xff, X12
	MOV	ZERO, X14	// count
	ADD	X10, X11	// end

	PCALIGN	$16
loop:
	BEQ	X10, X11, done
	MOVBU	(X10), X15
	ADD	$1, X10
	BNE	X12, X15, loop
	ADD	$1, X14
	JMP	loop

done:
	MOV	X14, X10
	RET
