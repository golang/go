// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build mips64 || mips64le

#include "go_asm.h"
#include "textflag.h"

#define	REGCTXT	R22

// memequal(a, b unsafe.Pointer, size uintptr) bool
TEXT runtime·memequal(SB),NOSPLIT|NOFRAME,$0-25
	MOVV	a+0(FP), R1
	MOVV	b+8(FP), R2
	BEQ	R1, R2, eq
	MOVV	size+16(FP), R3
	ADDV	R1, R3, R4

	// chunk size is 16
	SGTU	$16, R3, R8
	BEQ	R0, R8, chunk_entry

byte_loop:
	BNE	R1, R4, byte_test
	MOVV	$1, R1
	MOVB	R1, ret+24(FP)
	RET
byte_test:
	MOVBU	(R1), R6
	ADDV	$1, R1
	MOVBU	(R2), R7
	ADDV	$1, R2
	BEQ	R6, R7, byte_loop
	JMP	not_eq

chunk_entry:
	// make sure both a and b are aligned
	OR	R1, R2, R9
	AND	$0x7, R9
	BNE	R0, R9, byte_loop
	JMP	chunk_loop_1

chunk_loop:
	// chunk size is 16
	SGTU	$16, R3, R8
	BNE	R0, R8, chunk_tail_8
chunk_loop_1:
	MOVV	(R1), R6
	MOVV	(R2), R7
	BNE	R6, R7, not_eq
	MOVV	8(R1), R12
	MOVV	8(R2), R13
	ADDV	$16, R1
	ADDV	$16, R2
	SUBV	$16, R3
	BEQ	R12, R13, chunk_loop
	JMP	not_eq

chunk_tail_8:
	AND	$8, R3, R14
	BEQ	R0, R14, chunk_tail_4
	MOVV	(R1), R6
	MOVV	(R2), R7
	BNE	R6, R7, not_eq
	ADDV	$8, R1
	ADDV	$8, R2

chunk_tail_4:
	AND	$4, R3, R14
	BEQ	R0, R14, chunk_tail_2
	MOVWU	(R1), R6
	MOVWU	(R2), R7
	BNE	R6, R7, not_eq
	ADDV	$4, R1
	ADDV	$4, R2

chunk_tail_2:
	AND	$2, R3, R14
	BEQ	R0, R14, chunk_tail_1
	MOVHU	(R1), R6
	MOVHU	(R2), R7
	BNE	R6, R7, not_eq
	ADDV	$2, R1
	ADDV	$2, R2

chunk_tail_1:
	AND	$1, R3, R14
	BEQ	R0, R14, eq
	MOVBU	(R1), R6
	MOVBU	(R2), R7
	BEQ	R6, R7, eq

not_eq:
	MOVB	R0, ret+24(FP)
	RET
eq:
	MOVV	$1, R1
	MOVB	R1, ret+24(FP)
	RET

// memequal_varlen(a, b unsafe.Pointer) bool
TEXT runtime·memequal_varlen(SB),NOSPLIT,$40-17
	MOVV	a+0(FP), R1
	MOVV	b+8(FP), R2
	BEQ	R1, R2, eq
	MOVV	8(REGCTXT), R3    // compiler stores size at offset 8 in the closure
	MOVV	R1, 8(R29)
	MOVV	R2, 16(R29)
	MOVV	R3, 24(R29)
	JAL	runtime·memequal(SB)
	MOVBU	32(R29), R1
	MOVB	R1, ret+16(FP)
	RET
eq:
	MOVV	$1, R1
	MOVB	R1, ret+16(FP)
	RET
