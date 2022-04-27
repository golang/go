// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !math_big_pure_go
// +build !math_big_pure_go

#include "textflag.h"

// This file provides fast assembly versions for the elementary
// arithmetic operations on vectors implemented in arith.go.

// TODO: Consider re-implementing using Advanced SIMD
// once the assembler supports those instructions.

// func addVV(z, x, y []Word) (c Word)
TEXT ·addVV(SB),NOSPLIT,$0
	MOVD	z_len+8(FP), R0
	MOVD	x+24(FP), R8
	MOVD	y+48(FP), R9
	MOVD	z+0(FP), R10
	ADDS	$0, R0		// clear carry flag
	TBZ	$0, R0, two
	MOVD.P	8(R8), R11
	MOVD.P	8(R9), R15
	ADCS	R15, R11
	MOVD.P	R11, 8(R10)
	SUB	$1, R0
two:
	TBZ	$1, R0, loop
	LDP.P	16(R8), (R11, R12)
	LDP.P	16(R9), (R15, R16)
	ADCS	R15, R11
	ADCS	R16, R12
	STP.P	(R11, R12), 16(R10)
	SUB	$2, R0
loop:
	CBZ	R0, done	// careful not to touch the carry flag
	LDP.P	32(R8), (R11, R12)
	LDP	-16(R8), (R13, R14)
	LDP.P	32(R9), (R15, R16)
	LDP	-16(R9), (R17, R19)
	ADCS	R15, R11
	ADCS	R16, R12
	ADCS	R17, R13
	ADCS	R19, R14
	STP.P	(R11, R12), 32(R10)
	STP	(R13, R14), -16(R10)
	SUB	$4, R0
	B	loop
done:
	CSET	HS, R0		// extract carry flag
	MOVD	R0, c+72(FP)
	RET


// func subVV(z, x, y []Word) (c Word)
TEXT ·subVV(SB),NOSPLIT,$0
	MOVD	z_len+8(FP), R0
	MOVD	x+24(FP), R8
	MOVD	y+48(FP), R9
	MOVD	z+0(FP), R10
	CMP	R0, R0		// set carry flag
	TBZ	$0, R0, two
	MOVD.P	8(R8), R11
	MOVD.P	8(R9), R15
	SBCS	R15, R11
	MOVD.P	R11, 8(R10)
	SUB	$1, R0
two:
	TBZ	$1, R0, loop
	LDP.P	16(R8), (R11, R12)
	LDP.P	16(R9), (R15, R16)
	SBCS	R15, R11
	SBCS	R16, R12
	STP.P	(R11, R12), 16(R10)
	SUB	$2, R0
loop:
	CBZ	R0, done	// careful not to touch the carry flag
	LDP.P	32(R8), (R11, R12)
	LDP	-16(R8), (R13, R14)
	LDP.P	32(R9), (R15, R16)
	LDP	-16(R9), (R17, R19)
	SBCS	R15, R11
	SBCS	R16, R12
	SBCS	R17, R13
	SBCS	R19, R14
	STP.P	(R11, R12), 32(R10)
	STP	(R13, R14), -16(R10)
	SUB	$4, R0
	B	loop
done:
	CSET	LO, R0		// extract carry flag
	MOVD	R0, c+72(FP)
	RET

#define vwOneOp(instr, op1)				\
	MOVD.P	8(R1), R4;				\
	instr	op1, R4;				\
	MOVD.P	R4, 8(R3);

// handle the first 1~4 elements before starting iteration in addVW/subVW
#define vwPreIter(instr1, instr2, counter, target)	\
	vwOneOp(instr1, R2);				\
	SUB	$1, counter;				\
	CBZ	counter, target;			\
	vwOneOp(instr2, $0);				\
	SUB	$1, counter;				\
	CBZ	counter, target;			\
	vwOneOp(instr2, $0);				\
	SUB	$1, counter;				\
	CBZ	counter, target;			\
	vwOneOp(instr2, $0);

// do one iteration of add or sub in addVW/subVW
#define vwOneIter(instr, counter, exit)	\
	CBZ	counter, exit;		\	// careful not to touch the carry flag
	LDP.P	32(R1), (R4, R5);	\
	LDP	-16(R1), (R6, R7);	\
	instr	$0, R4, R8;		\
	instr	$0, R5, R9;		\
	instr	$0, R6, R10;		\
	instr	$0, R7, R11;		\
	STP.P	(R8, R9), 32(R3);	\
	STP	(R10, R11), -16(R3);	\
	SUB	$4, counter;

// do one iteration of copy in addVW/subVW
#define vwOneIterCopy(counter, exit)			\
	CBZ	counter, exit;				\
	LDP.P	32(R1), (R4, R5);			\
	LDP	-16(R1), (R6, R7);			\
	STP.P	(R4, R5), 32(R3);			\
	STP	(R6, R7), -16(R3);			\
	SUB	$4, counter;

// func addVW(z, x []Word, y Word) (c Word)
// The 'large' branch handles large 'z'. It checks the carry flag on every iteration
// and switches to copy if we are done with carries. The copying is skipped as well
// if 'x' and 'z' happen to share the same underlying storage.
// The overhead of the checking and branching is visible when 'z' are small (~5%),
// so set a threshold of 32, and remain the small-sized part entirely untouched.
TEXT ·addVW(SB),NOSPLIT,$0
	MOVD	z+0(FP), R3
	MOVD	z_len+8(FP), R0
	MOVD	x+24(FP), R1
	MOVD	y+48(FP), R2
	CMP	$32, R0
	BGE	large		// large-sized 'z' and 'x'
	CBZ	R0, len0	// the length of z is 0
	MOVD.P	8(R1), R4
	ADDS	R2, R4		// z[0] = x[0] + y, set carry
	MOVD.P	R4, 8(R3)
	SUB	$1, R0
	CBZ	R0, len1	// the length of z is 1
	TBZ	$0, R0, two
	MOVD.P	8(R1), R4	// do it once
	ADCS	$0, R4
	MOVD.P	R4, 8(R3)
	SUB	$1, R0
two:				// do it twice
	TBZ	$1, R0, loop
	LDP.P	16(R1), (R4, R5)
	ADCS	$0, R4, R8	// c, z[i] = x[i] + c
	ADCS	$0, R5, R9
	STP.P	(R8, R9), 16(R3)
	SUB	$2, R0
loop:				// do four times per round
	vwOneIter(ADCS, R0, len1)
	B	loop
len1:
	CSET	HS, R2		// extract carry flag
len0:
	MOVD	R2, c+56(FP)
done:
	RET
large:
	AND	$0x3, R0, R10
	AND	$~0x3, R0
	// unrolling for the first 1~4 elements to avoid saving the carry
	// flag in each step, adjust $R0 if we unrolled 4 elements
	vwPreIter(ADDS, ADCS, R10, add4)
	SUB	$4, R0
add4:
	BCC	copy
	vwOneIter(ADCS, R0, len1)
	B	add4
copy:
	MOVD	ZR, c+56(FP)
	CMP	R1, R3
	BEQ	done
copy_4:				// no carry flag, copy the rest
	vwOneIterCopy(R0, done)
	B	copy_4

// func subVW(z, x []Word, y Word) (c Word)
// The 'large' branch handles large 'z'. It checks the carry flag on every iteration
// and switches to copy if we are done with carries. The copying is skipped as well
// if 'x' and 'z' happen to share the same underlying storage.
// The overhead of the checking and branching is visible when 'z' are small (~5%),
// so set a threshold of 32, and remain the small-sized part entirely untouched.
TEXT ·subVW(SB),NOSPLIT,$0
	MOVD	z+0(FP), R3
	MOVD	z_len+8(FP), R0
	MOVD	x+24(FP), R1
	MOVD	y+48(FP), R2
	CMP	$32, R0
	BGE	large		// large-sized 'z' and 'x'
	CBZ	R0, len0	// the length of z is 0
	MOVD.P	8(R1), R4
	SUBS	R2, R4		// z[0] = x[0] - y, set carry
	MOVD.P	R4, 8(R3)
	SUB	$1, R0
	CBZ	R0, len1	// the length of z is 1
	TBZ	$0, R0, two	// do it once
	MOVD.P	8(R1), R4
	SBCS	$0, R4
	MOVD.P	R4, 8(R3)
	SUB	$1, R0
two:				// do it twice
	TBZ	$1, R0, loop
	LDP.P	16(R1), (R4, R5)
	SBCS	$0, R4, R8	// c, z[i] = x[i] + c
	SBCS	$0, R5, R9
	STP.P	(R8, R9), 16(R3)
	SUB	$2, R0
loop:				// do four times per round
	vwOneIter(SBCS, R0, len1)
	B	loop
len1:
	CSET	LO, R2		// extract carry flag
len0:
	MOVD	R2, c+56(FP)
done:
	RET
large:
	AND	$0x3, R0, R10
	AND	$~0x3, R0
	// unrolling for the first 1~4 elements to avoid saving the carry
	// flag in each step, adjust $R0 if we unrolled 4 elements
	vwPreIter(SUBS, SBCS, R10, sub4)
	SUB	$4, R0
sub4:
	BCS	copy
	vwOneIter(SBCS, R0, len1)
	B	sub4
copy:
	MOVD	ZR, c+56(FP)
	CMP	R1, R3
	BEQ	done
copy_4:				// no carry flag, copy the rest
	vwOneIterCopy(R0, done)
	B	copy_4

// func shlVU(z, x []Word, s uint) (c Word)
// This implementation handles the shift operation from the high word to the low word,
// which may be an error for the case where the low word of x overlaps with the high
// word of z. When calling this function directly, you need to pay attention to this
// situation.
TEXT ·shlVU(SB),NOSPLIT,$0
	LDP	z+0(FP), (R0, R1)	// R0 = z.ptr, R1 = len(z)
	MOVD	x+24(FP), R2
	MOVD	s+48(FP), R3
	ADD	R1<<3, R0	// R0 = &z[n]
	ADD	R1<<3, R2	// R2 = &x[n]
	CBZ	R1, len0
	CBZ	R3, copy	// if the number of shift is 0, just copy x to z
	MOVD	$64, R4
	SUB	R3, R4
	// handling the most significant element x[n-1]
	MOVD.W	-8(R2), R6
	LSR	R4, R6, R5	// return value
	LSL	R3, R6, R8	// x[i] << s
	SUB	$1, R1
one:	TBZ	$0, R1, two
	MOVD.W	-8(R2), R6
	LSR	R4, R6, R7
	ORR	R8, R7
	LSL	R3, R6, R8
	SUB	$1, R1
	MOVD.W	R7, -8(R0)
two:
	TBZ	$1, R1, loop
	LDP.W	-16(R2), (R6, R7)
	LSR	R4, R7, R10
	ORR	R8, R10
	LSL	R3, R7
	LSR	R4, R6, R9
	ORR	R7, R9
	LSL	R3, R6, R8
	SUB	$2, R1
	STP.W	(R9, R10), -16(R0)
loop:
	CBZ	R1, done
	LDP.W	-32(R2), (R10, R11)
	LDP	16(R2), (R12, R13)
	LSR	R4, R13, R23
	ORR	R8, R23		// z[i] = (x[i] << s) | (x[i-1] >> (64 - s))
	LSL	R3, R13
	LSR	R4, R12, R22
	ORR	R13, R22
	LSL	R3, R12
	LSR	R4, R11, R21
	ORR	R12, R21
	LSL	R3, R11
	LSR	R4, R10, R20
	ORR	R11, R20
	LSL	R3, R10, R8
	STP.W	(R20, R21), -32(R0)
	STP	(R22, R23), 16(R0)
	SUB	$4, R1
	B	loop
done:
	MOVD.W	R8, -8(R0)	// the first element x[0]
	MOVD	R5, c+56(FP)	// the part moved out from x[n-1]
	RET
copy:
	CMP	R0, R2
	BEQ	len0
	TBZ	$0, R1, ctwo
	MOVD.W	-8(R2), R4
	MOVD.W	R4, -8(R0)
	SUB	$1, R1
ctwo:
	TBZ	$1, R1, cloop
	LDP.W	-16(R2), (R4, R5)
	STP.W	(R4, R5), -16(R0)
	SUB	$2, R1
cloop:
	CBZ	R1, len0
	LDP.W	-32(R2), (R4, R5)
	LDP	16(R2), (R6, R7)
	STP.W	(R4, R5), -32(R0)
	STP	(R6, R7), 16(R0)
	SUB	$4, R1
	B	cloop
len0:
	MOVD	$0, c+56(FP)
	RET

// func shrVU(z, x []Word, s uint) (c Word)
// This implementation handles the shift operation from the low word to the high word,
// which may be an error for the case where the high word of x overlaps with the low
// word of z. When calling this function directly, you need to pay attention to this
// situation.
TEXT ·shrVU(SB),NOSPLIT,$0
	MOVD	z+0(FP), R0
	MOVD	z_len+8(FP), R1
	MOVD	x+24(FP), R2
	MOVD	s+48(FP), R3
	MOVD	$0, R8
	MOVD	$64, R4
	SUB	R3, R4
	CBZ	R1, len0
	CBZ	R3, copy	// if the number of shift is 0, just copy x to z

	MOVD.P	8(R2), R20
	LSR	R3, R20, R8
	LSL	R4, R20
	MOVD	R20, c+56(FP)	// deal with the first element
	SUB	$1, R1

	TBZ	$0, R1, two
	MOVD.P	8(R2), R6
	LSL	R4, R6, R20
	ORR	R8, R20
	LSR	R3, R6, R8
	MOVD.P	R20, 8(R0)
	SUB	$1, R1
two:
	TBZ	$1, R1, loop
	LDP.P	16(R2), (R6, R7)
	LSL	R4, R6, R20
	LSR	R3, R6
	ORR	R8, R20
	LSL	R4, R7, R21
	LSR	R3, R7, R8
	ORR	R6, R21
	STP.P	(R20, R21), 16(R0)
	SUB	$2, R1
loop:
	CBZ	R1, done
	LDP.P	32(R2), (R10, R11)
	LDP	-16(R2), (R12, R13)
	LSL	R4, R10, R20
	LSR	R3, R10
	ORR	R8, R20		// z[i] = (x[i] >> s) | (x[i+1] << (64 - s))
	LSL	R4, R11, R21
	LSR	R3, R11
	ORR	R10, R21
	LSL	R4, R12, R22
	LSR	R3, R12
	ORR	R11, R22
	LSL	R4, R13, R23
	LSR	R3, R13, R8
	ORR	R12, R23
	STP.P	(R20, R21), 32(R0)
	STP	(R22, R23), -16(R0)
	SUB	$4, R1
	B	loop
done:
	MOVD	R8, (R0)	// deal with the last element
	RET
copy:
	CMP	R0, R2
	BEQ	len0
	TBZ	$0, R1, ctwo
	MOVD.P	8(R2), R3
	MOVD.P	R3, 8(R0)
	SUB	$1, R1
ctwo:
	TBZ	$1, R1, cloop
	LDP.P	16(R2), (R4, R5)
	STP.P	(R4, R5), 16(R0)
	SUB	$2, R1
cloop:
	CBZ	R1, len0
	LDP.P	32(R2), (R4, R5)
	LDP	-16(R2), (R6, R7)
	STP.P	(R4, R5), 32(R0)
	STP	(R6, R7), -16(R0)
	SUB	$4, R1
	B	cloop
len0:
	MOVD	$0, c+56(FP)
	RET


// func mulAddVWW(z, x []Word, y, r Word) (c Word)
TEXT ·mulAddVWW(SB),NOSPLIT,$0
	MOVD	z+0(FP), R1
	MOVD	z_len+8(FP), R0
	MOVD	x+24(FP), R2
	MOVD	y+48(FP), R3
	MOVD	r+56(FP), R4
	// c, z = x * y + r
	TBZ	$0, R0, two
	MOVD.P	8(R2), R5
	MUL	R3, R5, R7
	UMULH	R3, R5, R8
	ADDS	R4, R7
	ADC	$0, R8, R4	// c, z[i] = x[i] * y +  r
	MOVD.P	R7, 8(R1)
	SUB	$1, R0
two:
	TBZ	$1, R0, loop
	LDP.P	16(R2), (R5, R6)
	MUL	R3, R5, R10
	UMULH	R3, R5, R11
	ADDS	R4, R10
	MUL	R3, R6, R12
	UMULH	R3, R6, R13
	ADCS	R12, R11
	ADC	$0, R13, R4

	STP.P	(R10, R11), 16(R1)
	SUB	$2, R0
loop:
	CBZ	R0, done
	LDP.P	32(R2), (R5, R6)
	LDP	-16(R2), (R7, R8)

	MUL	R3, R5, R10
	UMULH	R3, R5, R11
	ADDS	R4, R10
	MUL	R3, R6, R12
	UMULH	R3, R6, R13
	ADCS	R11, R12

	MUL	R3, R7, R14
	UMULH	R3, R7, R15
	ADCS	R13, R14
	MUL	R3, R8, R16
	UMULH	R3, R8, R17
	ADCS	R15, R16
	ADC	$0, R17, R4

	STP.P	(R10, R12), 32(R1)
	STP	(R14, R16), -16(R1)
	SUB	$4, R0
	B	loop
done:
	MOVD	R4, c+64(FP)
	RET


// func addMulVVW(z, x []Word, y Word) (c Word)
TEXT ·addMulVVW(SB),NOSPLIT,$0
	MOVD	z+0(FP), R1
	MOVD	z_len+8(FP), R0
	MOVD	x+24(FP), R2
	MOVD	y+48(FP), R3
	MOVD	$0, R4

	TBZ	$0, R0, two

	MOVD.P	8(R2), R5
	MOVD	(R1), R6

	MUL	R5, R3, R7
	UMULH	R5, R3, R8

	ADDS	R7, R6
	ADC	$0, R8, R4

	MOVD.P	R6, 8(R1)
	SUB	$1, R0

two:
	TBZ	$1, R0, loop

	LDP.P	16(R2), (R5, R10)
	LDP	(R1), (R6, R11)

	MUL	R10, R3, R13
	UMULH	R10, R3, R12

	MUL	R5, R3, R7
	UMULH	R5, R3, R8

	ADDS	R4, R6
	ADCS	R13, R11
	ADC	$0, R12

	ADDS	R7, R6
	ADCS	R8, R11
	ADC	$0, R12, R4

	STP.P	(R6, R11), 16(R1)
	SUB	$2, R0

// The main loop of this code operates on a block of 4 words every iteration
// performing [R4:R12:R11:R10:R9] = R4 + R3 * [R8:R7:R6:R5] + [R12:R11:R10:R9]
// where R4 is carried from the previous iteration, R8:R7:R6:R5 hold the next
// 4 words of x, R3 is y and R12:R11:R10:R9 are part of the result z.
loop:
	CBZ	R0, done

	LDP.P	16(R2), (R5, R6)
	LDP.P	16(R2), (R7, R8)

	LDP	(R1), (R9, R10)
	ADDS	R4, R9
	MUL	R6, R3, R14
	ADCS	R14, R10
	MUL	R7, R3, R15
	LDP	16(R1), (R11, R12)
	ADCS	R15, R11
	MUL	R8, R3, R16
	ADCS	R16, R12
	UMULH	R8, R3, R20
	ADC	$0, R20

	MUL	R5, R3, R13
	ADDS	R13, R9
	UMULH	R5, R3, R17
	ADCS	R17, R10
	UMULH	R6, R3, R21
	STP.P	(R9, R10), 16(R1)
	ADCS	R21, R11
	UMULH	R7, R3, R19
	ADCS	R19, R12
	STP.P	(R11, R12), 16(R1)
	ADC	$0, R20, R4

	SUB	$4, R0
	B	loop

done:
	MOVD	R4, c+56(FP)
	RET


