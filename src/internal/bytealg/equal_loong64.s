// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

#define	REGCTXT	R29

// memequal(a, b unsafe.Pointer, size uintptr) bool
TEXT runtime·memequal<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-25
	BEQ	R4, R5, eq
	ADDV	R4, R6, R7
	PCALIGN	$16
loop:
	BNE	R4, R7, test
	MOVV	$1, R4
	RET
test:
	MOVBU	(R4), R9
	ADDV	$1, R4
	MOVBU	(R5), R10
	ADDV	$1, R5
	BEQ	R9, R10, loop

	MOVB    R0, R4
	RET
eq:
	MOVV	$1, R4
	RET

// memequal_varlen(a, b unsafe.Pointer) bool
TEXT runtime·memequal_varlen<ABIInternal>(SB),NOSPLIT,$40-17
	BEQ	R4, R5, eq
	MOVV	8(REGCTXT), R6    // compiler stores size at offset 8 in the closure
	MOVV	R4, 8(R3)
	MOVV	R5, 16(R3)
	MOVV	R6, 24(R3)
	JAL	runtime·memequal(SB)
	MOVBU	32(R3), R4
	RET
eq:
	MOVV	$1, R4
	RET
