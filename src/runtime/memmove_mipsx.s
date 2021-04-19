// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build mips mipsle

#include "textflag.h"

#ifdef GOARCH_mips
#define MOVWHI  MOVWL
#define MOVWLO  MOVWR
#else
#define MOVWHI  MOVWR
#define MOVWLO  MOVWL
#endif

// void runtime·memmove(void*, void*, uintptr)
TEXT runtime·memmove(SB),NOSPLIT,$-0-12
	MOVW	n+8(FP), R3
	MOVW	from+4(FP), R2
	MOVW	to+0(FP), R1

	ADDU	R3, R2, R4	// end pointer for source
	ADDU	R3, R1, R5	// end pointer for destination

	// if destination is ahead of source, start at the end of the buffer and go backward.
	SGTU	R1, R2, R6
	BNE	R6, backward

	// if less than 4 bytes, use byte by byte copying
	SGTU	$4, R3, R6
	BNE	R6, f_small_copy

	// align destination to 4 bytes
	AND	$3, R1, R6
	BEQ	R6, f_dest_aligned
	SUBU	R1, R0, R6
	AND	$3, R6
	MOVWHI	0(R2), R7
	SUBU	R6, R3
	MOVWLO	3(R2), R7
	ADDU	R6, R2
	MOVWHI	R7, 0(R1)
	ADDU	R6, R1

f_dest_aligned:
	AND	$31, R3, R7
	AND	$3, R3, R6
	SUBU	R7, R5, R7	// end pointer for 32-byte chunks
	SUBU	R6, R5, R6	// end pointer for 4-byte chunks

	// if source is not aligned, use unaligned reads
	AND	$3, R2, R8
	BNE	R8, f_large_ua

f_large:
	BEQ	R1, R7, f_words
	ADDU	$32, R1
	MOVW	0(R2), R8
	MOVW	4(R2), R9
	MOVW	8(R2), R10
	MOVW	12(R2), R11
	MOVW	16(R2), R12
	MOVW	20(R2), R13
	MOVW	24(R2), R14
	MOVW	28(R2), R15
	ADDU	$32, R2
	MOVW	R8, -32(R1)
	MOVW	R9, -28(R1)
	MOVW	R10, -24(R1)
	MOVW	R11, -20(R1)
	MOVW	R12, -16(R1)
	MOVW	R13, -12(R1)
	MOVW	R14, -8(R1)
	MOVW	R15, -4(R1)
	JMP	f_large

f_words:
	BEQ	R1, R6, f_tail
	ADDU	$4, R1
	MOVW	0(R2), R8
	ADDU	$4, R2
	MOVW	R8, -4(R1)
	JMP	f_words

f_tail:
	BEQ	R1, R5, ret
	MOVWLO	-1(R4), R8
	MOVWLO	R8, -1(R5)

ret:
	RET

f_large_ua:
	BEQ	R1, R7, f_words_ua
	ADDU	$32, R1
	MOVWHI	0(R2), R8
	MOVWHI	4(R2), R9
	MOVWHI	8(R2), R10
	MOVWHI	12(R2), R11
	MOVWHI	16(R2), R12
	MOVWHI	20(R2), R13
	MOVWHI	24(R2), R14
	MOVWHI	28(R2), R15
	MOVWLO	3(R2), R8
	MOVWLO	7(R2), R9
	MOVWLO	11(R2), R10
	MOVWLO	15(R2), R11
	MOVWLO	19(R2), R12
	MOVWLO	23(R2), R13
	MOVWLO	27(R2), R14
	MOVWLO	31(R2), R15
	ADDU	$32, R2
	MOVW	R8, -32(R1)
	MOVW	R9, -28(R1)
	MOVW	R10, -24(R1)
	MOVW	R11, -20(R1)
	MOVW	R12, -16(R1)
	MOVW	R13, -12(R1)
	MOVW	R14, -8(R1)
	MOVW	R15, -4(R1)
	JMP	f_large_ua

f_words_ua:
	BEQ	R1, R6, f_tail_ua
	MOVWHI	0(R2), R8
	ADDU	$4, R1
	MOVWLO	3(R2), R8
	ADDU	$4, R2
	MOVW	R8, -4(R1)
	JMP	f_words_ua

f_tail_ua:
	BEQ	R1, R5, ret
	MOVWHI	-4(R4), R8
	MOVWLO	-1(R4), R8
	MOVWLO	R8, -1(R5)
	JMP	ret

f_small_copy:
	BEQ	R1, R5, ret
	ADDU	$1, R1
	MOVB	0(R2), R6
	ADDU	$1, R2
	MOVB	R6, -1(R1)
	JMP	f_small_copy

backward:
	SGTU	$4, R3, R6
	BNE	R6, b_small_copy

	AND	$3, R5, R6
	BEQ	R6, b_dest_aligned
	MOVWHI	-4(R4), R7
	SUBU	R6, R3
	MOVWLO	-1(R4), R7
	SUBU	R6, R4
	MOVWLO	R7, -1(R5)
	SUBU	R6, R5

b_dest_aligned:
	AND	$31, R3, R7
	AND	$3, R3, R6
	ADDU	R7, R1, R7
	ADDU	R6, R1, R6

	AND	$3, R4, R8
	BNE	R8, b_large_ua

b_large:
	BEQ	R5, R7, b_words
	ADDU	$-32, R5
	MOVW	-4(R4), R8
	MOVW	-8(R4), R9
	MOVW	-12(R4), R10
	MOVW	-16(R4), R11
	MOVW	-20(R4), R12
	MOVW	-24(R4), R13
	MOVW	-28(R4), R14
	MOVW	-32(R4), R15
	ADDU	$-32, R4
	MOVW	R8, 28(R5)
	MOVW	R9, 24(R5)
	MOVW	R10, 20(R5)
	MOVW	R11, 16(R5)
	MOVW	R12, 12(R5)
	MOVW	R13, 8(R5)
	MOVW	R14, 4(R5)
	MOVW	R15, 0(R5)
	JMP	b_large

b_words:
	BEQ	R5, R6, b_tail
	ADDU	$-4, R5
	MOVW	-4(R4), R8
	ADDU	$-4, R4
	MOVW	R8, 0(R5)
	JMP	b_words

b_tail:
	BEQ	R5, R1, ret
	MOVWHI	0(R2), R8	// R2 and R1 have the same alignment so we don't need to load a whole word
	MOVWHI	R8, 0(R1)
	JMP	ret

b_large_ua:
	BEQ	R5, R7, b_words_ua
	ADDU	$-32, R5
	MOVWHI	-4(R4), R8
	MOVWHI	-8(R4), R9
	MOVWHI	-12(R4), R10
	MOVWHI	-16(R4), R11
	MOVWHI	-20(R4), R12
	MOVWHI	-24(R4), R13
	MOVWHI	-28(R4), R14
	MOVWHI	-32(R4), R15
	MOVWLO	-1(R4), R8
	MOVWLO	-5(R4), R9
	MOVWLO	-9(R4), R10
	MOVWLO	-13(R4), R11
	MOVWLO	-17(R4), R12
	MOVWLO	-21(R4), R13
	MOVWLO	-25(R4), R14
	MOVWLO	-29(R4), R15
	ADDU	$-32, R4
	MOVW	R8, 28(R5)
	MOVW	R9, 24(R5)
	MOVW	R10, 20(R5)
	MOVW	R11, 16(R5)
	MOVW	R12, 12(R5)
	MOVW	R13, 8(R5)
	MOVW	R14, 4(R5)
	MOVW	R15, 0(R5)
	JMP	b_large_ua

b_words_ua:
	BEQ	R5, R6, b_tail_ua
	MOVWHI	-4(R4), R8
	ADDU	$-4, R5
	MOVWLO	-1(R4), R8
	ADDU	$-4, R4
	MOVW	R8, 0(R5)
	JMP	b_words_ua

b_tail_ua:
	BEQ	R5, R1, ret
	MOVWHI	(R2), R8
	MOVWLO	3(R2), R8
	MOVWHI	R8, 0(R1)
	JMP ret

b_small_copy:
	BEQ	R5, R1, ret
	ADDU	$-1, R5
	MOVB	-1(R4), R6
	ADDU	$-1, R4
	MOVB	R6, 0(R5)
	JMP	b_small_copy
