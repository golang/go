// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

// condition code masks
#define EQ 8
#define NE 7

// register assignments
#define R_ZERO R0
#define R_VAL  R1
#define R_TMP  R2
#define R_PTR  R3
#define R_LEN  R4
#define R_CHAR R5
#define R_RET  R6
#define R_ITER R7
#define R_CNT  R8
#define R_MPTR R9

// vector register assignments
#define V_ZERO V0
#define V_CHAR V1
#define V_MASK V2
#define V_VAL  V3
#define V_CNT  V4

// mask for trailing bytes in vector implementation
GLOBL countbytemask<>(SB), RODATA, $16
DATA countbytemask<>+0(SB)/8, $0x0101010101010101
DATA countbytemask<>+8(SB)/8, $0x0101010101010101

// func Count(b []byte, c byte) int
TEXT ·Count(SB), NOSPLIT|NOFRAME, $0-40
	LMG   b+0(FP), R_PTR, R_LEN
	MOVBZ c+24(FP), R_CHAR
	MOVD  $ret+32(FP), R_RET
	BR    countbytebody<>(SB)

// func CountString(s string, c byte) int
TEXT ·CountString(SB), NOSPLIT|NOFRAME, $0-32
	LMG   s+0(FP), R_PTR, R_LEN
	MOVBZ c+16(FP), R_CHAR
	MOVD  $ret+24(FP), R_RET
	BR    countbytebody<>(SB)

// input:
// R_PTR  = address of array of bytes
// R_LEN  = number of bytes in array
// R_CHAR = byte value to count zero (extended to register width)
// R_RET  = address of return value
TEXT countbytebody<>(SB), NOSPLIT|NOFRAME, $0-0
	MOVD  $internal∕cpu·S390X+const_offsetS390xHasVX(SB), R_TMP
	MOVD  $countbytemask<>(SB), R_MPTR
	CGIJ  $EQ, R_LEN, $0, ret0 // return if length is 0.
	SRD   $4, R_LEN, R_ITER    // R_ITER is the number of 16-byte chunks
	MOVBZ (R_TMP), R_TMP       // load bool indicating support for vector facility
	CGIJ  $EQ, R_TMP, $0, novx // jump to scalar code if the vector facility is not available

	// Start of vector code (have vector facility).
	//
	// Set R_LEN to be the length mod 16 minus 1 to use as an index for
	// vector 'load with length' (VLL). It will be in the range [-1,14].
	// Also replicate c across a 16-byte vector and initialize V_ZERO.
	ANDW  $0xf, R_LEN
	VLVGB $0, R_CHAR, V_CHAR // V_CHAR = [16]byte{c, 0, ..., 0, 0}
	VZERO V_ZERO             // V_ZERO = [1]uint128{0}
	ADDW  $-1, R_LEN
	VREPB $0, V_CHAR, V_CHAR // V_CHAR = [16]byte{c, c, ..., c, c}

	// Jump to loop if we have more than 15 bytes to process.
	CGIJ $NE, R_ITER, $0, vxchunks

	// Load 1-15 bytes and corresponding mask.
	// Note: only the low 32-bits of R_LEN are used for the index.
	VLL R_LEN, (R_PTR), V_VAL
	VLL R_LEN, (R_MPTR), V_MASK

	// Compare each byte in input chunk against byte to be counted.
	// Each byte element will be set to either 0 (no match) or 1 (match).
	VCEQB V_CHAR, V_VAL, V_VAL // each byte will be either 0xff or 0x00
	VN    V_MASK, V_VAL, V_VAL // mask out most significant 7 bits

	// Accumulate matched byte count in 128-bit integer value.
	VSUMB  V_VAL, V_ZERO, V_VAL // [16]byte{x0, x1, ..., x14, x15} → [4]uint32{x0+x1+x2+x3, ..., x12+x13+x14+x15}
	VSUMQF V_VAL, V_ZERO, V_CNT // [4]uint32{x0, x1, x2, x3} → [1]uint128{x0+x1+x2+x3}

	// Return rightmost (lowest) 64-bit part of accumulator.
	VSTEG $1, V_CNT, (R_RET)
	RET

vxchunks:
	// Load 0x01 into every byte element in the 16-byte mask vector.
	VREPIB $1, V_MASK // V_MASK = [16]byte{1, 1, ..., 1, 1}
	VZERO  V_CNT      // initial uint128 count of 0

vxloop:
	// Load input bytes in 16-byte chunks.
	VL (R_PTR), V_VAL

	// Compare each byte in input chunk against byte to be counted.
	// Each byte element will be set to either 0 (no match) or 1 (match).
	VCEQB V_CHAR, V_VAL, V_VAL // each byte will be either 0xff or 0x00
	VN    V_MASK, V_VAL, V_VAL // mask out most significant 7 bits

	// Increment input string address.
	MOVD $16(R_PTR), R_PTR

	// Accumulate matched byte count in 128-bit integer value.
	VSUMB  V_VAL, V_ZERO, V_VAL // [16]byte{x0, x1, ..., x14, x15} → [4]uint32{x0+x1+x2+x3, ..., x12+x13+x14+x15}
	VSUMQF V_VAL, V_ZERO, V_VAL // [4]uint32{x0, x1, x2, x3} → [1]uint128{x0+x1+x2+x3}
	VAQ    V_VAL, V_CNT, V_CNT  // accumulate

	// Repeat until all 16-byte chunks are done.
	BRCTG R_ITER, vxloop

	// Skip to end if there are no trailing bytes.
	CIJ $EQ, R_LEN, $-1, vxret

	// Load 1-15 bytes and corresponding mask.
	// Note: only the low 32-bits of R_LEN are used for the index.
	VLL R_LEN, (R_PTR), V_VAL
	VLL R_LEN, (R_MPTR), V_MASK

	// Compare each byte in input chunk against byte to be counted.
	// Each byte element will be set to either 0 (no match) or 1 (match).
	VCEQB V_CHAR, V_VAL, V_VAL
	VN    V_MASK, V_VAL, V_VAL

	// Accumulate matched byte count in 128-bit integer value.
	VSUMB  V_VAL, V_ZERO, V_VAL // [16]byte{x0, x1, ..., x14, x15} → [4]uint32{x0+x1+x2+x3, ..., x12+x13+x14+x15}
	VSUMQF V_VAL, V_ZERO, V_VAL // [4]uint32{x0, x1, x2, x3} → [1]uint128{x0+x1+x2+x3}
	VAQ    V_VAL, V_CNT, V_CNT  // accumulate

vxret:
	// Return rightmost (lowest) 64-bit part of accumulator.
	VSTEG $1, V_CNT, (R_RET)
	RET

novx:
	// Start of non-vector code (the vector facility not available).
	//
	// Initialise counter and constant zero.
	MOVD $0, R_CNT
	MOVD $0, R_ZERO

loop:
	// Read 1-byte from input and compare.
	// Note: avoid putting LOCGR in critical path.
	MOVBZ (R_PTR), R_VAL
	MOVD  $1, R_TMP
	MOVD  $1(R_PTR), R_PTR
	CMPW  R_VAL, R_CHAR
	LOCGR $NE, R_ZERO, R_TMP // select 0 if no match (1 if there is a match)
	ADD   R_TMP, R_CNT       // accumulate 64-bit result

	// Repeat until all bytes have been checked.
	BRCTG R_LEN, loop

ret:
	MOVD R_CNT, (R_RET)
	RET

ret0:
	MOVD $0, (R_RET)
	RET
