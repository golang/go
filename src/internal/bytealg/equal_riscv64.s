// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "asm_riscv64.h"
#include "go_asm.h"
#include "textflag.h"

#define	CTXT	S10

// func memequal_varlen(a, b unsafe.Pointer) bool
TEXT runtime·memequal_varlen<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-17
	// X10 = a_base
	// X11 = b_base
	MOV	8(CTXT), X12    // compiler stores size at offset 8 in the closure
	JMP	runtime·memequal<ABIInternal>(SB)

// func memequal(a, b unsafe.Pointer, size uintptr) bool
TEXT runtime·memequal<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-25
	// X10 = a_base
	// X11 = b_base
	// X12 = size
	BNE	X10, X11, length_check
	MOV	$0, X12

length_check:
	BEQZ	X12, done

	MOV	$32, X23
	BLT	X12, X23, loop4_check

#ifndef hasV
	MOVB	internal∕cpu·RISCV64+const_offsetRISCV64HasV(SB), X5
	BEQZ	X5, equal_scalar
#endif

	// Use vector if not 8 byte aligned.
	OR	X10, X11, X5
	AND	$7, X5
	BNEZ	X5, vector_loop

	// Use scalar if 8 byte aligned and <= 64 bytes.
	SUB	$64, X12, X6
	BLEZ	X6, loop32_check

	PCALIGN	$16
vector_loop:
	VSETVLI	X12, E8, M8, TA, MA, X5
	VLE8V	(X10), V8
	VLE8V	(X11), V16
	VMSNEVV	V8, V16, V0
	VFIRSTM	V0, X6
	BGEZ	X6, done
	ADD	X5, X10
	ADD	X5, X11
	SUB	X5, X12
	BNEZ	X12, vector_loop
	JMP	done

equal_scalar:
	// Check alignment - if alignment differs we have to do one byte at a time.
	AND	$7, X10, X9
	AND	$7, X11, X19
	BNE	X9, X19, loop4_check
	BEQZ	X9, loop32_check

	// Check one byte at a time until we reach 8 byte alignment.
	SUB	X9, X0, X9
	ADD	$8, X9, X9
	SUB	X9, X12, X12
align:
	SUB	$1, X9
	MOVBU	0(X10), X19
	MOVBU	0(X11), X20
	BNE	X19, X20, done
	ADD	$1, X10
	ADD	$1, X11
	BNEZ	X9, align

loop32_check:
	MOV	$32, X9
	BLT	X12, X9, loop16_check
loop32:
	MOV	0(X10), X19
	MOV	0(X11), X20
	MOV	8(X10), X21
	MOV	8(X11), X22
	BNE	X19, X20, done
	BNE	X21, X22, done
	MOV	16(X10), X14
	MOV	16(X11), X15
	MOV	24(X10), X16
	MOV	24(X11), X17
	BNE	X14, X15, done
	BNE	X16, X17, done
	ADD	$32, X10
	ADD	$32, X11
	SUB	$32, X12
	BGE	X12, X9, loop32
	BEQZ	X12, done

loop16_check:
	MOV	$16, X23
	BLT	X12, X23, loop4_check
loop16:
	MOV	0(X10), X19
	MOV	0(X11), X20
	MOV	8(X10), X21
	MOV	8(X11), X22
	BNE	X19, X20, done
	BNE	X21, X22, done
	ADD	$16, X10
	ADD	$16, X11
	SUB	$16, X12
	BGE	X12, X23, loop16
	BEQZ	X12, done

loop4_check:
	MOV	$4, X23
	BLT	X12, X23, loop1
loop4:
	MOVBU	0(X10), X19
	MOVBU	0(X11), X20
	MOVBU	1(X10), X21
	MOVBU	1(X11), X22
	BNE	X19, X20, done
	BNE	X21, X22, done
	MOVBU	2(X10), X14
	MOVBU	2(X11), X15
	MOVBU	3(X10), X16
	MOVBU	3(X11), X17
	BNE	X14, X15, done
	BNE	X16, X17, done
	ADD	$4, X10
	ADD	$4, X11
	SUB	$4, X12
	BGE	X12, X23, loop4

loop1:
	BEQZ	X12, done
	MOVBU	0(X10), X19
	MOVBU	0(X11), X20
	BNE	X19, X20, done
	ADD	$1, X10
	ADD	$1, X11
	SUB	$1, X12
	JMP	loop1

done:
	// If X12 is zero then memory is equivalent.
	SEQZ	X12, X10
	RET
