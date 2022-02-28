// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ppc64 || ppc64le

#include "go_asm.h"
#include "textflag.h"

TEXT ·Compare<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-56
#ifdef GOEXPERIMENT_regabiargs
// incoming:
// R3 a addr -> R5
// R4 a len  -> R3
// R5 a cap unused
// R6 b addr -> R6
// R7 b len  -> R4
// R8 b cap unused
	MOVD	R3, R5
	MOVD	R4, R3
	MOVD	R7, R4
#else
	MOVD	a_base+0(FP), R5
	MOVD	b_base+24(FP), R6
	MOVD	a_len+8(FP), R3
	MOVD	b_len+32(FP), R4
	MOVD	$ret+48(FP), R7
#endif
	CMP     R5,R6,CR7
	CMP	R3,R4,CR6
	BEQ	CR7,equal
#ifdef	GOARCH_ppc64le
	BR	cmpbodyLE<>(SB)
#else
	BR      cmpbodyBE<>(SB)
#endif
equal:
	BEQ	CR6,done
	MOVD	$1, R8
	BGT	CR6,greater
	NEG	R8
greater:
#ifdef GOEXPERIMENT_regabiargs
	MOVD	R8, R3
#else
	MOVD	R8, (R7)
#endif
	RET
done:
#ifdef GOEXPERIMENT_regabiargs
	MOVD	$0, R3
#else
	MOVD	$0, (R7)
#endif
	RET

TEXT runtime·cmpstring<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-40
#ifdef GOEXPERIMENT_regabiargs
// incoming:
// R3 a addr -> R5
// R4 a len  -> R3
// R5 b addr -> R6
// R6 b len  -> R4
	MOVD	R6, R7
	MOVD	R5, R6
	MOVD	R3, R5
	MOVD	R4, R3
	MOVD	R7, R4
#else
	MOVD	a_base+0(FP), R5
	MOVD	b_base+16(FP), R6
	MOVD	a_len+8(FP), R3
	MOVD	b_len+24(FP), R4
	MOVD	$ret+32(FP), R7
#endif
	CMP     R5,R6,CR7
	CMP	R3,R4,CR6
	BEQ	CR7,equal
#ifdef	GOARCH_ppc64le
	BR	cmpbodyLE<>(SB)
#else
	BR      cmpbodyBE<>(SB)
#endif
equal:
	BEQ	CR6,done
	MOVD	$1, R8
	BGT	CR6,greater
	NEG	R8
greater:
#ifdef GOEXPERIMENT_regabiargs
	MOVD	R8, R3
#else
	MOVD	R8, (R7)
#endif
	RET

done:
#ifdef GOEXPERIMENT_regabiargs
	MOVD	$0, R3
#else
	MOVD	$0, (R7)
#endif
	RET

// Do an efficient memcmp for ppc64le
// R3 = a len
// R4 = b len
// R5 = a addr
// R6 = b addr
// R7 = addr of return value if not regabi
TEXT cmpbodyLE<>(SB),NOSPLIT|NOFRAME,$0-0
	MOVD	R3,R8		// set up length
	CMP	R3,R4,CR2	// unequal?
	BC	12,8,setuplen	// BLT CR2
	MOVD	R4,R8		// use R4 for comparison len
setuplen:
	MOVD	R8,CTR		// set up loop counter
	CMP	R8,$8		// only optimize >=8
	BLT	simplecheck
	DCBT	(R5)		// cache hint
	DCBT	(R6)
	CMP	R8,$32		// optimize >= 32
	MOVD	R8,R9
	BLT	setup8a		// 8 byte moves only
setup32a:
	SRADCC	$5,R8,R9	// number of 32 byte chunks
	MOVD	R9,CTR

        // Special processing for 32 bytes or longer.
        // Loading this way is faster and correct as long as the
	// doublewords being compared are equal. Once they
	// are found unequal, reload them in proper byte order
	// to determine greater or less than.
loop32a:
	MOVD	0(R5),R9	// doublewords to compare
	MOVD	0(R6),R10	// get 4 doublewords
	MOVD	8(R5),R14
	MOVD	8(R6),R15
	CMPU	R9,R10		// bytes equal?
	MOVD	$0,R16		// set up for cmpne
	BNE	cmpne		// further compare for LT or GT
	MOVD	16(R5),R9	// get next pair of doublewords
	MOVD	16(R6),R10
	CMPU	R14,R15		// bytes match?
	MOVD	$8,R16		// set up for cmpne
	BNE	cmpne		// further compare for LT or GT
	MOVD	24(R5),R14	// get next pair of doublewords
	MOVD    24(R6),R15
	CMPU	R9,R10		// bytes match?
	MOVD	$16,R16		// set up for cmpne
	BNE	cmpne		// further compare for LT or GT
	MOVD	$-8,R16		// for cmpne, R5,R6 already inc by 32
	ADD	$32,R5		// bump up to next 32
	ADD	$32,R6
	CMPU    R14,R15		// bytes match?
	BC	8,2,loop32a	// br ctr and cr
	BNE	cmpne
	ANDCC	$24,R8,R9	// Any 8 byte chunks?
	BEQ	leftover	// and result is 0
setup8a:
	SRADCC	$3,R9,R9	// get the 8 byte count
	BEQ	leftover	// shifted value is 0
	MOVD	R9,CTR		// loop count for doublewords
loop8:
	MOVDBR	(R5+R0),R9	// doublewords to compare
	MOVDBR	(R6+R0),R10	// LE compare order
	ADD	$8,R5
	ADD	$8,R6
	CMPU	R9,R10		// match?
	BC	8,2,loop8	// bt ctr <> 0 && cr
	BGT	greater
	BLT	less
leftover:
	ANDCC	$7,R8,R9	// check for leftover bytes
	MOVD	R9,CTR		// save the ctr
	BNE	simple		// leftover bytes
	BC	12,10,equal	// test CR2 for length comparison
	BC	12,8,less
	BR	greater
simplecheck:
	CMP	R8,$0		// remaining compare length 0
	BNE	simple		// do simple compare
	BC	12,10,equal	// test CR2 for length comparison
	BC	12,8,less	// 1st len < 2nd len, result less
	BR	greater		// 1st len > 2nd len must be greater
simple:
	MOVBZ	0(R5), R9	// get byte from 1st operand
	ADD	$1,R5
	MOVBZ	0(R6), R10	// get byte from 2nd operand
	ADD	$1,R6
	CMPU	R9, R10
	BC	8,2,simple	// bc ctr <> 0 && cr
	BGT	greater		// 1st > 2nd
	BLT	less		// 1st < 2nd
	BC	12,10,equal	// test CR2 for length comparison
	BC	12,9,greater	// 2nd len > 1st len
	BR	less		// must be less
cmpne:				// only here is not equal
	MOVDBR	(R5+R16),R8	// reload in reverse order
	MOVDBR	(R6+R16),R9
	CMPU	R8,R9		// compare correct endianness
	BGT	greater		// here only if NE
less:
	MOVD	$-1,R3
#ifndef GOEXPERIMENT_regabiargs
	MOVD	R3,(R7)		// return value if A < B
#endif
	RET
equal:
#ifdef GOEXPERIMENT_regabiargs
	MOVD	$0, R3
#else
	MOVD	$0,(R7)		// return value if A == B
#endif
	RET
greater:
	MOVD	$1,R3
#ifndef GOEXPERIMENT_regabiargs
	MOVD	R3,(R7)		// return value if A > B
#endif
	RET

// Do an efficient memcmp for ppc64 (BE)
// R3 = a len
// R4 = b len
// R5 = a addr
// R6 = b addr
// R7 = addr of return value
TEXT cmpbodyBE<>(SB),NOSPLIT|NOFRAME,$0-0
	MOVD	R3,R8		// set up length
	CMP	R3,R4,CR2	// unequal?
	BC	12,8,setuplen	// BLT CR2
	MOVD	R4,R8		// use R4 for comparison len
setuplen:
	MOVD	R8,CTR		// set up loop counter
	CMP	R8,$8		// only optimize >=8
	BLT	simplecheck
	DCBT	(R5)		// cache hint
	DCBT	(R6)
	CMP	R8,$32		// optimize >= 32
	MOVD	R8,R9
	BLT	setup8a		// 8 byte moves only

setup32a:
	SRADCC	$5,R8,R9	// number of 32 byte chunks
	MOVD	R9,CTR
loop32a:
	MOVD	0(R5),R9	// doublewords to compare
	MOVD	0(R6),R10	// get 4 doublewords
	MOVD	8(R5),R14
	MOVD	8(R6),R15
	CMPU	R9,R10		// bytes equal?
	BLT	less		// found to be less
	BGT	greater		// found to be greater
	MOVD	16(R5),R9	// get next pair of doublewords
	MOVD	16(R6),R10
	CMPU	R14,R15		// bytes match?
	BLT	less		// found less
	BGT	greater		// found greater
	MOVD	24(R5),R14	// get next pair of doublewords
	MOVD	24(R6),R15
	CMPU	R9,R10		// bytes match?
	BLT	less		// found to be less
	BGT	greater		// found to be greater
	ADD	$32,R5		// bump up to next 32
	ADD	$32,R6
	CMPU	R14,R15		// bytes match?
	BC	8,2,loop32a	// br ctr and cr
	BLT	less		// with BE, byte ordering is
	BGT	greater		// good for compare
	ANDCC	$24,R8,R9	// Any 8 byte chunks?
	BEQ	leftover	// and result is 0
setup8a:
	SRADCC	$3,R9,R9	// get the 8 byte count
	BEQ	leftover	// shifted value is 0
	MOVD	R9,CTR		// loop count for doublewords
loop8:
	MOVD	(R5),R9
	MOVD	(R6),R10
	ADD	$8,R5
	ADD	$8,R6
	CMPU	R9,R10		// match?
	BC	8,2,loop8	// bt ctr <> 0 && cr
	BGT	greater
	BLT	less
leftover:
	ANDCC	$7,R8,R9	// check for leftover bytes
	MOVD	R9,CTR		// save the ctr
	BNE	simple		// leftover bytes
	BC	12,10,equal	// test CR2 for length comparison
	BC	12,8,less
	BR	greater
simplecheck:
	CMP	R8,$0		// remaining compare length 0
	BNE	simple		// do simple compare
	BC	12,10,equal	// test CR2 for length comparison
	BC 	12,8,less	// 1st len < 2nd len, result less
	BR	greater		// same len, must be equal
simple:
	MOVBZ	0(R5),R9	// get byte from 1st operand
	ADD	$1,R5
	MOVBZ	0(R6),R10	// get byte from 2nd operand
	ADD	$1,R6
	CMPU	R9,R10
	BC	8,2,simple	// bc ctr <> 0 && cr
	BGT	greater		// 1st > 2nd
	BLT	less		// 1st < 2nd
	BC	12,10,equal	// test CR2 for length comparison
	BC	12,9,greater	// 2nd len > 1st len
less:
	MOVD	$-1,R3
#ifndef GOEXPERIMENT_regabiargs
	MOVD    R3,(R7)		// return value if A < B
#endif
	RET
equal:
#ifdef GOEXPERIMENT_regabiargs
	MOVD	$0, R3
#else
	MOVD    $0,(R7)		// return value if A == B
#endif
	RET
greater:
	MOVD	$1,R3
#ifndef GOEXPERIMENT_regabiargs
	MOVD	R3,(R7)		// return value if A > B
#endif
	RET
