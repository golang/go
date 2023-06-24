// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// This is a port of the s390x asm implementation.
// to ppc64le.

// Some changes were needed due to differences in
// the Go opcodes and/or available instructions
// between s390x and ppc64le.

// 1. There were operand order differences in the
// VSUBUQM, VSUBCUQ, and VSEL instructions.

// 2. ppc64 does not have a multiply high and low
// like s390x, so those were implemented using
// macros to compute the equivalent values.

// 3. The LVX, STVX instructions on ppc64 require
// 16 byte alignment of the data.  To avoid that
// requirement, data is loaded using LXVD2X and
// STXVD2X with VPERM to reorder bytes correctly.

// I have identified some areas where I believe
// changes would be needed to make this work for big
// endian; however additional changes beyond what I
// have noted are most likely needed to make it work.
// - The string used with VPERM to swap the byte order
//   for loads and stores.
// - The constants that are loaded from CPOOL.
//

// The following constants are defined in an order
// that is correct for use with LXVD2X/STXVD2X
// on little endian.
DATA p256<>+0x00(SB)/8, $0xffffffff00000001 // P256
DATA p256<>+0x08(SB)/8, $0x0000000000000000 // P256
DATA p256<>+0x10(SB)/8, $0x00000000ffffffff // P256
DATA p256<>+0x18(SB)/8, $0xffffffffffffffff // P256
DATA p256<>+0x20(SB)/8, $0x0c0d0e0f1c1d1e1f // SEL d1 d0 d1 d0
DATA p256<>+0x28(SB)/8, $0x0c0d0e0f1c1d1e1f // SEL d1 d0 d1 d0
DATA p256<>+0x30(SB)/8, $0x0000000010111213 // SEL 0  d1 d0  0
DATA p256<>+0x38(SB)/8, $0x1415161700000000 // SEL 0  d1 d0  0
DATA p256<>+0x40(SB)/8, $0x18191a1b1c1d1e1f // SEL d1 d0 d1 d0
DATA p256<>+0x48(SB)/8, $0x18191a1b1c1d1e1f // SEL d1 d0 d1 d0
DATA p256mul<>+0x00(SB)/8, $0x00000000ffffffff // P256 original
DATA p256mul<>+0x08(SB)/8, $0xffffffffffffffff // P256
DATA p256mul<>+0x10(SB)/8, $0xffffffff00000001 // P256 original
DATA p256mul<>+0x18(SB)/8, $0x0000000000000000 // P256
DATA p256mul<>+0x20(SB)/8, $0x1c1d1e1f00000000 // SEL d0  0  0 d0
DATA p256mul<>+0x28(SB)/8, $0x000000001c1d1e1f // SEL d0  0  0 d0
DATA p256mul<>+0x30(SB)/8, $0x0001020304050607 // SEL d0  0 d1 d0
DATA p256mul<>+0x38(SB)/8, $0x1c1d1e1f0c0d0e0f // SEL d0  0 d1 d0
DATA p256mul<>+0x40(SB)/8, $0x040506071c1d1e1f // SEL  0 d1 d0 d1
DATA p256mul<>+0x48(SB)/8, $0x0c0d0e0f1c1d1e1f // SEL  0 d1 d0 d1
DATA p256mul<>+0x50(SB)/8, $0x0405060704050607 // SEL  0  0 d1 d0
DATA p256mul<>+0x58(SB)/8, $0x1c1d1e1f0c0d0e0f // SEL  0  0 d1 d0
DATA p256mul<>+0x60(SB)/8, $0x0c0d0e0f1c1d1e1f // SEL d1 d0 d1 d0
DATA p256mul<>+0x68(SB)/8, $0x0c0d0e0f1c1d1e1f // SEL d1 d0 d1 d0
DATA p256mul<>+0x70(SB)/8, $0x141516170c0d0e0f // SEL 0  d1 d0  0
DATA p256mul<>+0x78(SB)/8, $0x1c1d1e1f14151617 // SEL 0  d1 d0  0
DATA p256mul<>+0x80(SB)/8, $0xffffffff00000000 // (1*2^256)%P256
DATA p256mul<>+0x88(SB)/8, $0x0000000000000001 // (1*2^256)%P256
DATA p256mul<>+0x90(SB)/8, $0x00000000fffffffe // (1*2^256)%P256
DATA p256mul<>+0x98(SB)/8, $0xffffffffffffffff // (1*2^256)%P256

// External declarations for constants
GLOBL p256ord<>(SB), 8, $32
GLOBL p256<>(SB), 8, $80
GLOBL p256mul<>(SB), 8, $160

// The following macros are used to implement the ppc64le
// equivalent function from the corresponding s390x
// instruction for vector multiply high, low, and add,
// since there aren't exact equivalent instructions.
// The corresponding s390x instructions appear in the
// comments.
// Implementation for big endian would have to be
// investigated, I think it would be different.
//
//
// Vector multiply word
//
//	VMLF  x0, x1, out_low
//	VMLHF x0, x1, out_hi
#define VMULT(x1, x2, out_low, out_hi) \
	VMULEUW x1, x2, TMP1; \
	VMULOUW x1, x2, TMP2; \
	VMRGEW TMP1, TMP2, out_hi; \
	VMRGOW TMP1, TMP2, out_low

//
// Vector multiply add word
//
//	VMALF  x0, x1, y, out_low
//	VMALHF x0, x1, y, out_hi
#define VMULT_ADD(x1, x2, y, one, out_low, out_hi) \
	VMULEUW  y, one, TMP2; \
	VMULOUW  y, one, TMP1; \
	VMULEUW  x1, x2, out_low; \
	VMULOUW  x1, x2, out_hi; \
	VADDUDM  TMP2, out_low, TMP2; \
	VADDUDM  TMP1, out_hi, TMP1; \
	VMRGOW   TMP2, TMP1, out_low; \
	VMRGEW   TMP2, TMP1, out_hi

#define res_ptr R3
#define a_ptr R4

#undef res_ptr
#undef a_ptr

#define P1ptr   R3
#define CPOOL   R7

#define Y1L   V0
#define Y1H   V1
#define T1L   V2
#define T1H   V3

#define PL    V30
#define PH    V31

#define CAR1  V6
// func p256NegCond(val *p256Point, cond int)
TEXT ·p256NegCond(SB), NOSPLIT, $0-16
	MOVD val+0(FP), P1ptr
	MOVD $16, R16

	MOVD cond+8(FP), R6
	CMP  $0, R6
	BC   12, 2, LR      // just return if cond == 0

	MOVD $p256mul<>+0x00(SB), CPOOL

	LXVD2X (P1ptr)(R0), Y1L
	LXVD2X (P1ptr)(R16), Y1H

	XXPERMDI Y1H, Y1H, $2, Y1H
	XXPERMDI Y1L, Y1L, $2, Y1L

	LXVD2X (CPOOL)(R0), PL
	LXVD2X (CPOOL)(R16), PH

	VSUBCUQ  PL, Y1L, CAR1      // subtract part2 giving carry
	VSUBUQM  PL, Y1L, T1L       // subtract part2 giving result
	VSUBEUQM PH, Y1H, CAR1, T1H // subtract part1 using carry from part2

	XXPERMDI T1H, T1H, $2, T1H
	XXPERMDI T1L, T1L, $2, T1L

	STXVD2X T1L, (R0+P1ptr)
	STXVD2X T1H, (R16+P1ptr)
	RET

#undef P1ptr
#undef CPOOL
#undef Y1L
#undef Y1H
#undef T1L
#undef T1H
#undef PL
#undef PH
#undef CAR1

#define P3ptr   R3
#define P1ptr   R4
#define P2ptr   R5

#define X1L    V0
#define X1H    V1
#define Y1L    V2
#define Y1H    V3
#define Z1L    V4
#define Z1H    V5
#define X2L    V6
#define X2H    V7
#define Y2L    V8
#define Y2H    V9
#define Z2L    V10
#define Z2H    V11
#define SEL    V12
#define ZER    V13

// This function uses LXVD2X and STXVD2X to avoid the
// data alignment requirement for LVX, STVX. Since
// this code is just moving bytes and not doing arithmetic,
// order of the bytes doesn't matter.
//
// func p256MovCond(res, a, b *p256Point, cond int)
TEXT ·p256MovCond(SB), NOSPLIT, $0-32
	MOVD res+0(FP), P3ptr
	MOVD a+8(FP), P1ptr
	MOVD b+16(FP), P2ptr
	MOVD $16, R16
	MOVD $32, R17
	MOVD $48, R18
	MOVD $56, R21
	MOVD $64, R19
	MOVD $80, R20
	// cond is R1 + 24 (cond offset) + 32
	LXVDSX (R1)(R21), SEL
	VSPLTISB $0, ZER
	// SEL controls whether to store a or b
	VCMPEQUD SEL, ZER, SEL

	LXVD2X (P1ptr+R0), X1H
	LXVD2X (P1ptr+R16), X1L
	LXVD2X (P1ptr+R17), Y1H
	LXVD2X (P1ptr+R18), Y1L
	LXVD2X (P1ptr+R19), Z1H
	LXVD2X (P1ptr+R20), Z1L

	LXVD2X (P2ptr+R0), X2H
	LXVD2X (P2ptr+R16), X2L
	LXVD2X (P2ptr+R17), Y2H
	LXVD2X (P2ptr+R18), Y2L
	LXVD2X (P2ptr+R19), Z2H
	LXVD2X (P2ptr+R20), Z2L

	VSEL X1H, X2H, SEL, X1H
	VSEL X1L, X2L, SEL, X1L
	VSEL Y1H, Y2H, SEL, Y1H
	VSEL Y1L, Y2L, SEL, Y1L
	VSEL Z1H, Z2H, SEL, Z1H
	VSEL Z1L, Z2L, SEL, Z1L

	STXVD2X X1H, (P3ptr+R0)
	STXVD2X X1L, (P3ptr+R16)
	STXVD2X Y1H, (P3ptr+R17)
	STXVD2X Y1L, (P3ptr+R18)
	STXVD2X Z1H, (P3ptr+R19)
	STXVD2X Z1L, (P3ptr+R20)

	RET

#undef P3ptr
#undef P1ptr
#undef P2ptr
#undef X1L
#undef X1H
#undef Y1L
#undef Y1H
#undef Z1L
#undef Z1H
#undef X2L
#undef X2H
#undef Y2L
#undef Y2H
#undef Z2L
#undef Z2H
#undef SEL
#undef ZER

#define P3ptr   R3
#define P1ptr   R4
#define COUNT   R5

#define X1L    V0
#define X1H    V1
#define Y1L    V2
#define Y1H    V3
#define Z1L    V4
#define Z1H    V5
#define X2L    V6
#define X2H    V7
#define Y2L    V8
#define Y2H    V9
#define Z2L    V10
#define Z2H    V11

#define ONE   V18
#define IDX   V19
#define SEL1  V20
#define SEL2  V21
// func p256Select(point *p256Point, table *p256Table, idx int)
TEXT ·p256Select(SB), NOSPLIT, $0-24
	MOVD res+0(FP), P3ptr
	MOVD table+8(FP), P1ptr
	MOVD $16, R16
	MOVD $32, R17
	MOVD $48, R18
	MOVD $64, R19
	MOVD $80, R20

	LXVDSX   (R1)(R18), SEL1 // VLREPG idx+32(FP), SEL1
	VSPLTB   $7, SEL1, IDX    // splat byte
	VSPLTISB $1, ONE          // VREPIB $1, ONE
	VSPLTISB $1, SEL2         // VREPIB $1, SEL2
	MOVD     $17, COUNT
	MOVD     COUNT, CTR       // set up ctr

	VSPLTISB $0, X1H // VZERO  X1H
	VSPLTISB $0, X1L // VZERO  X1L
	VSPLTISB $0, Y1H // VZERO  Y1H
	VSPLTISB $0, Y1L // VZERO  Y1L
	VSPLTISB $0, Z1H // VZERO  Z1H
	VSPLTISB $0, Z1L // VZERO  Z1L

loop_select:

	// LVXD2X is used here since data alignment doesn't
	// matter.

	LXVD2X (P1ptr+R0), X2H
	LXVD2X (P1ptr+R16), X2L
	LXVD2X (P1ptr+R17), Y2H
	LXVD2X (P1ptr+R18), Y2L
	LXVD2X (P1ptr+R19), Z2H
	LXVD2X (P1ptr+R20), Z2L

	VCMPEQUD SEL2, IDX, SEL1 // VCEQG SEL2, IDX, SEL1 OK

	// This will result in SEL1 being all 0s or 1s, meaning
	// the result is either X1L or X2L, no individual byte
	// selection.

	VSEL X1L, X2L, SEL1, X1L
	VSEL X1H, X2H, SEL1, X1H
	VSEL Y1L, Y2L, SEL1, Y1L
	VSEL Y1H, Y2H, SEL1, Y1H
	VSEL Z1L, Z2L, SEL1, Z1L
	VSEL Z1H, Z2H, SEL1, Z1H

	// Add 1 to all bytes in SEL2
	VADDUBM SEL2, ONE, SEL2    // VAB  SEL2, ONE, SEL2 OK
	ADD     $96, P1ptr
	BDNZ    loop_select

	// STXVD2X is used here so that alignment doesn't
	// need to be verified. Since values were loaded
	// using LXVD2X this is OK.
	STXVD2X X1H, (P3ptr+R0)
	STXVD2X X1L, (P3ptr+R16)
	STXVD2X Y1H, (P3ptr+R17)
	STXVD2X Y1L, (P3ptr+R18)
	STXVD2X Z1H, (P3ptr+R19)
	STXVD2X Z1L, (P3ptr+R20)
	RET

#undef P3ptr
#undef P1ptr
#undef COUNT
#undef X1L
#undef X1H
#undef Y1L
#undef Y1H
#undef Z1L
#undef Z1H
#undef X2L
#undef X2H
#undef Y2L
#undef Y2H
#undef Z2L
#undef Z2H
#undef ONE
#undef IDX
#undef SEL1
#undef SEL2

// The following functions all reverse the byte order.

//func p256BigToLittle(res *p256Element, in *[32]byte)
TEXT ·p256BigToLittle(SB), NOSPLIT, $0-16
	MOVD	res+0(FP), R3
	MOVD	in+8(FP), R4
	BR	p256InternalEndianSwap<>(SB)

//func p256LittleToBig(res *[32]byte, in *p256Element)
TEXT ·p256LittleToBig(SB), NOSPLIT, $0-16
	MOVD	res+0(FP), R3
	MOVD	in+8(FP), R4
	BR	p256InternalEndianSwap<>(SB)

//func p256OrdBigToLittle(res *p256OrdElement, in *[32]byte)
TEXT ·p256OrdBigToLittle(SB), NOSPLIT, $0-16
	MOVD	res+0(FP), R3
	MOVD	in+8(FP), R4
	BR	p256InternalEndianSwap<>(SB)

//func p256OrdLittleToBig(res *[32]byte, in *p256OrdElement)
TEXT ·p256OrdLittleToBig(SB), NOSPLIT, $0-16
	MOVD	res+0(FP), R3
	MOVD	in+8(FP), R4
	BR	p256InternalEndianSwap<>(SB)

TEXT p256InternalEndianSwap<>(SB), NOSPLIT, $0-0
	// Index registers needed for BR movs
	MOVD	$8, R9
	MOVD	$16, R10
	MOVD	$24, R14

	MOVDBR	(R0)(R4), R5
	MOVDBR	(R9)(R4), R6
	MOVDBR	(R10)(R4), R7
	MOVDBR	(R14)(R4), R8

	MOVD	R8, 0(R3)
	MOVD	R7, 8(R3)
	MOVD	R6, 16(R3)
	MOVD	R5, 24(R3)

	RET

#define P3ptr   R3
#define P1ptr   R4
#define COUNT   R5

#define X1L    V0
#define X1H    V1
#define Y1L    V2
#define Y1H    V3
#define Z1L    V4
#define Z1H    V5
#define X2L    V6
#define X2H    V7
#define Y2L    V8
#define Y2H    V9
#define Z2L    V10
#define Z2H    V11

#define ONE   V18
#define IDX   V19
#define SEL1  V20
#define SEL2  V21

// func p256SelectAffine(res *p256AffinePoint, table *p256AffineTable, idx int)
TEXT ·p256SelectAffine(SB), NOSPLIT, $0-24
	MOVD res+0(FP), P3ptr
	MOVD table+8(FP), P1ptr
	MOVD $16, R16
	MOVD $32, R17
	MOVD $48, R18

	LXVDSX (R1)(R18), SEL1
	VSPLTB $7, SEL1, IDX    // splat byte

	VSPLTISB $1, ONE    // Vector with byte 1s
	VSPLTISB $1, SEL2   // Vector with byte 1s
	MOVD     $64, COUNT
	MOVD     COUNT, CTR // loop count

	VSPLTISB $0, X1H // VZERO  X1H
	VSPLTISB $0, X1L // VZERO  X1L
	VSPLTISB $0, Y1H // VZERO  Y1H
	VSPLTISB $0, Y1L // VZERO  Y1L

loop_select:
	LXVD2X (P1ptr+R0), X2H
	LXVD2X (P1ptr+R16), X2L
	LXVD2X (P1ptr+R17), Y2H
	LXVD2X (P1ptr+R18), Y2L

	VCMPEQUD SEL2, IDX, SEL1 // Compare against idx

	VSEL X1L, X2L, SEL1, X1L // Select if idx matched
	VSEL X1H, X2H, SEL1, X1H
	VSEL Y1L, Y2L, SEL1, Y1L
	VSEL Y1H, Y2H, SEL1, Y1H

	VADDUBM SEL2, ONE, SEL2    // Increment SEL2 bytes by 1
	ADD     $64, P1ptr         // Next chunk
	BDNZ	loop_select

	STXVD2X X1H, (P3ptr+R0)
	STXVD2X X1L, (P3ptr+R16)
	STXVD2X Y1H, (P3ptr+R17)
	STXVD2X Y1L, (P3ptr+R18)
	RET

#undef P3ptr
#undef P1ptr
#undef COUNT
#undef X1L
#undef X1H
#undef Y1L
#undef Y1H
#undef Z1L
#undef Z1H
#undef X2L
#undef X2H
#undef Y2L
#undef Y2H
#undef Z2L
#undef Z2H
#undef ONE
#undef IDX
#undef SEL1
#undef SEL2

#define res_ptr R3
#define x_ptr   R4
#define CPOOL   R7

#define T0   V0
#define T1   V1
#define T2   V2
#define TT0  V3
#define TT1  V4

#define ZER   V6
#define SEL1  V7
#define SEL2  V8
#define CAR1  V9
#define CAR2  V10
#define RED1  V11
#define RED2  V12
#define PL    V13
#define PH    V14

// func p256FromMont(res, in *p256Element)
TEXT ·p256FromMont(SB), NOSPLIT, $0-16
	MOVD res+0(FP), res_ptr
	MOVD in+8(FP), x_ptr

	MOVD $16, R16
	MOVD $32, R17
	MOVD $48, R18
	MOVD $64, R19
	MOVD $p256<>+0x00(SB), CPOOL

	VSPLTISB $0, T2  // VZERO T2
	VSPLTISB $0, ZER // VZERO ZER

	// Constants are defined so that the LXVD2X is correct
	LXVD2X (CPOOL+R0), PH
	LXVD2X (CPOOL+R16), PL

	// VPERM byte selections
	LXVD2X (CPOOL+R18), SEL2
	LXVD2X (CPOOL+R19), SEL1

	LXVD2X (R16)(x_ptr), T1
	LXVD2X (R0)(x_ptr), T0

	// Put in true little endian order
	XXPERMDI T0, T0, $2, T0
	XXPERMDI T1, T1, $2, T1

	// First round
	VPERM   T1, T0, SEL1, RED2    // d1 d0 d1 d0
	VPERM   ZER, RED2, SEL2, RED1 // 0  d1 d0  0
	VSUBUQM RED2, RED1, RED2      // VSQ   RED1, RED2, RED2      // Guaranteed not to underflow

	VSLDOI $8, T1, T0, T0 // VSLDB $8, T1, T0, T0
	VSLDOI $8, T2, T1, T1 // VSLDB $8, T2, T1, T1

	VADDCUQ  T0, RED1, CAR1       // VACCQ  T0, RED1, CAR1
	VADDUQM  T0, RED1, T0         // VAQ    T0, RED1, T0
	VADDECUQ T1, RED2, CAR1, CAR2 // VACCCQ T1, RED2, CAR1, CAR2
	VADDEUQM T1, RED2, CAR1, T1   // VACQ   T1, RED2, CAR1, T1
	VADDUQM  T2, CAR2, T2         // VAQ    T2, CAR2, T2

	// Second round
	VPERM   T1, T0, SEL1, RED2    // d1 d0 d1 d0
	VPERM   ZER, RED2, SEL2, RED1 // 0  d1 d0  0
	VSUBUQM RED2, RED1, RED2      // VSQ   RED1, RED2, RED2      // Guaranteed not to underflow

	VSLDOI $8, T1, T0, T0 // VSLDB $8, T1, T0, T0
	VSLDOI $8, T2, T1, T1 // VSLDB $8, T2, T1, T1

	VADDCUQ  T0, RED1, CAR1       // VACCQ  T0, RED1, CAR1
	VADDUQM  T0, RED1, T0         // VAQ    T0, RED1, T0
	VADDECUQ T1, RED2, CAR1, CAR2 // VACCCQ T1, RED2, CAR1, CAR2
	VADDEUQM T1, RED2, CAR1, T1   // VACQ   T1, RED2, CAR1, T1
	VADDUQM  T2, CAR2, T2         // VAQ    T2, CAR2, T2

	// Third round
	VPERM   T1, T0, SEL1, RED2    // d1 d0 d1 d0
	VPERM   ZER, RED2, SEL2, RED1 // 0  d1 d0  0
	VSUBUQM RED2, RED1, RED2      // VSQ   RED1, RED2, RED2      // Guaranteed not to underflow

	VSLDOI $8, T1, T0, T0 // VSLDB $8, T1, T0, T0
	VSLDOI $8, T2, T1, T1 // VSLDB $8, T2, T1, T1

	VADDCUQ  T0, RED1, CAR1       // VACCQ  T0, RED1, CAR1
	VADDUQM  T0, RED1, T0         // VAQ    T0, RED1, T0
	VADDECUQ T1, RED2, CAR1, CAR2 // VACCCQ T1, RED2, CAR1, CAR2
	VADDEUQM T1, RED2, CAR1, T1   // VACQ   T1, RED2, CAR1, T1
	VADDUQM  T2, CAR2, T2         // VAQ    T2, CAR2, T2

	// Last round
	VPERM   T1, T0, SEL1, RED2    // d1 d0 d1 d0
	VPERM   ZER, RED2, SEL2, RED1 // 0  d1 d0  0
	VSUBUQM RED2, RED1, RED2      // VSQ   RED1, RED2, RED2      // Guaranteed not to underflow

	VSLDOI $8, T1, T0, T0 // VSLDB $8, T1, T0, T0
	VSLDOI $8, T2, T1, T1 // VSLDB $8, T2, T1, T1

	VADDCUQ  T0, RED1, CAR1       // VACCQ  T0, RED1, CAR1
	VADDUQM  T0, RED1, T0         // VAQ    T0, RED1, T0
	VADDECUQ T1, RED2, CAR1, CAR2 // VACCCQ T1, RED2, CAR1, CAR2
	VADDEUQM T1, RED2, CAR1, T1   // VACQ   T1, RED2, CAR1, T1
	VADDUQM  T2, CAR2, T2         // VAQ    T2, CAR2, T2

	// ---------------------------------------------------

	VSUBCUQ  T0, PL, CAR1       // VSCBIQ  PL, T0, CAR1
	VSUBUQM  T0, PL, TT0        // VSQ     PL, T0, TT0
	VSUBECUQ T1, PH, CAR1, CAR2 // VSBCBIQ T1, PH, CAR1, CAR2
	VSUBEUQM T1, PH, CAR1, TT1  // VSBIQ   T1, PH, CAR1, TT1
	VSUBEUQM T2, ZER, CAR2, T2  // VSBIQ   T2, ZER, CAR2, T2

	VSEL TT0, T0, T2, T0
	VSEL TT1, T1, T2, T1

	// Reorder the bytes so STXVD2X can be used.
	// TT0, TT1 used for VPERM result in case
	// the caller expects T0, T1 to be good.
	XXPERMDI T0, T0, $2, TT0
	XXPERMDI T1, T1, $2, TT1

	STXVD2X TT0, (R0)(res_ptr)
	STXVD2X TT1, (R16)(res_ptr)
	RET

#undef res_ptr
#undef x_ptr
#undef CPOOL
#undef T0
#undef T1
#undef T2
#undef TT0
#undef TT1
#undef ZER
#undef SEL1
#undef SEL2
#undef CAR1
#undef CAR2
#undef RED1
#undef RED2
#undef PL
#undef PH

// ---------------------------------------
// p256MulInternal
// V0-V3 V30,V31 - Not Modified
// V4-V15 V27-V29 - Volatile

#define CPOOL   R7

// Parameters
#define X0    V0 // Not modified
#define X1    V1 // Not modified
#define Y0    V2 // Not modified
#define Y1    V3 // Not modified
#define T0    V4 // Result
#define T1    V5 // Result
#define P0    V30 // Not modified
#define P1    V31 // Not modified

// Temporaries: lots of reused vector regs
#define YDIG  V6 // Overloaded with CAR2
#define ADD1H V7 // Overloaded with ADD3H
#define ADD2H V8 // Overloaded with ADD4H
#define ADD3  V9 // Overloaded with SEL2,SEL5
#define ADD4  V10 // Overloaded with SEL3,SEL6
#define RED1  V11 // Overloaded with CAR2
#define RED2  V12
#define RED3  V13 // Overloaded with SEL1
#define T2    V14
// Overloaded temporaries
#define ADD1  V4 // Overloaded with T0
#define ADD2  V5 // Overloaded with T1
#define ADD3H V7 // Overloaded with ADD1H
#define ADD4H V8 // Overloaded with ADD2H
#define ZER   V28 // Overloaded with TMP1
#define CAR1  V6 // Overloaded with YDIG
#define CAR2  V11 // Overloaded with RED1
// Constant Selects
#define SEL1  V13 // Overloaded with RED3
#define SEL2  V9 // Overloaded with ADD3,SEL5
#define SEL3  V10 // Overloaded with ADD4,SEL6
#define SEL4  V6 // Overloaded with YDIG,CAR1
#define SEL5  V9 // Overloaded with ADD3,SEL2
#define SEL6  V10 // Overloaded with ADD4,SEL3

// TMP1, TMP2 used in
// VMULT macros
#define TMP1  V13 // Overloaded with RED3
#define TMP2  V27
#define ONE   V29 // 1s splatted by word

/* *
 * To follow the flow of bits, for your own sanity a stiff drink, need you shall.
 * Of a single round, a 'helpful' picture, here is. Meaning, column position has.
 * With you, SIMD be...
 *
 *                                           +--------+--------+
 *                                  +--------|  RED2  |  RED1  |
 *                                  |        +--------+--------+
 *                                  |       ---+--------+--------+
 *                                  |  +---- T2|   T1   |   T0   |--+
 *                                  |  |    ---+--------+--------+  |
 *                                  |  |                            |
 *                                  |  |    ======================= |
 *                                  |  |                            |
 *                                  |  |       +--------+--------+<-+
 *                                  |  +-------|  ADD2  |  ADD1  |--|-----+
 *                                  |  |       +--------+--------+  |     |
 *                                  |  |     +--------+--------+<---+     |
 *                                  |  |     | ADD2H  | ADD1H  |--+       |
 *                                  |  |     +--------+--------+  |       |
 *                                  |  |     +--------+--------+<-+       |
 *                                  |  |     |  ADD4  |  ADD3  |--|-+     |
 *                                  |  |     +--------+--------+  | |     |
 *                                  |  |   +--------+--------+<---+ |     |
 *                                  |  |   | ADD4H  | ADD3H  |------|-+   |(+vzero)
 *                                  |  |   +--------+--------+      | |   V
 *                                  |  | ------------------------   | | +--------+
 *                                  |  |                            | | |  RED3  |  [d0 0 0 d0]
 *                                  |  |                            | | +--------+
 *                                  |  +---->+--------+--------+    | |   |
 *   (T2[1w]||ADD2[4w]||ADD1[3w])   +--------|   T1   |   T0   |    | |   |
 *                                  |        +--------+--------+    | |   |
 *                                  +---->---+--------+--------+    | |   |
 *                                         T2|   T1   |   T0   |----+ |   |
 *                                        ---+--------+--------+    | |   |
 *                                        ---+--------+--------+<---+ |   |
 *                                    +--- T2|   T1   |   T0   |----------+
 *                                    |   ---+--------+--------+      |   |
 *                                    |  +--------+--------+<-------------+
 *                                    |  |  RED2  |  RED1  |-----+    |   | [0 d1 d0 d1] [d0 0 d1 d0]
 *                                    |  +--------+--------+     |    |   |
 *                                    |  +--------+<----------------------+
 *                                    |  |  RED3  |--------------+    |     [0 0 d1 d0]
 *                                    |  +--------+              |    |
 *                                    +--->+--------+--------+   |    |
 *                                         |   T1   |   T0   |--------+
 *                                         +--------+--------+   |    |
 *                                   --------------------------- |    |
 *                                                               |    |
 *                                       +--------+--------+<----+    |
 *                                       |  RED2  |  RED1  |          |
 *                                       +--------+--------+          |
 *                                      ---+--------+--------+<-------+
 *                                       T2|   T1   |   T0   |            (H1P-H1P-H00RRAY!)
 *                                      ---+--------+--------+
 *
 *                                                                *Mi obra de arte de siglo XXI @vpaprots
 *
 *
 * First group is special, doesn't get the two inputs:
 *                                             +--------+--------+<-+
 *                                     +-------|  ADD2  |  ADD1  |--|-----+
 *                                     |       +--------+--------+  |     |
 *                                     |     +--------+--------+<---+     |
 *                                     |     | ADD2H  | ADD1H  |--+       |
 *                                     |     +--------+--------+  |       |
 *                                     |     +--------+--------+<-+       |
 *                                     |     |  ADD4  |  ADD3  |--|-+     |
 *                                     |     +--------+--------+  | |     |
 *                                     |   +--------+--------+<---+ |     |
 *                                     |   | ADD4H  | ADD3H  |------|-+   |(+vzero)
 *                                     |   +--------+--------+      | |   V
 *                                     | ------------------------   | | +--------+
 *                                     |                            | | |  RED3  |  [d0 0 0 d0]
 *                                     |                            | | +--------+
 *                                     +---->+--------+--------+    | |   |
 *   (T2[1w]||ADD2[4w]||ADD1[3w])            |   T1   |   T0   |----+ |   |
 *                                           +--------+--------+    | |   |
 *                                        ---+--------+--------+<---+ |   |
 *                                    +--- T2|   T1   |   T0   |----------+
 *                                    |   ---+--------+--------+      |   |
 *                                    |  +--------+--------+<-------------+
 *                                    |  |  RED2  |  RED1  |-----+    |   | [0 d1 d0 d1] [d0 0 d1 d0]
 *                                    |  +--------+--------+     |    |   |
 *                                    |  +--------+<----------------------+
 *                                    |  |  RED3  |--------------+    |     [0 0 d1 d0]
 *                                    |  +--------+              |    |
 *                                    +--->+--------+--------+   |    |
 *                                         |   T1   |   T0   |--------+
 *                                         +--------+--------+   |    |
 *                                   --------------------------- |    |
 *                                                               |    |
 *                                       +--------+--------+<----+    |
 *                                       |  RED2  |  RED1  |          |
 *                                       +--------+--------+          |
 *                                      ---+--------+--------+<-------+
 *                                       T2|   T1   |   T0   |            (H1P-H1P-H00RRAY!)
 *                                      ---+--------+--------+
 *
 * Last 'group' needs to RED2||RED1 shifted less
 */
TEXT p256MulInternal<>(SB), NOSPLIT, $0-16
	// CPOOL loaded from caller
	MOVD $16, R16
	MOVD $32, R17
	MOVD $48, R18
	MOVD $64, R19
	MOVD $80, R20
	MOVD $96, R21
	MOVD $112, R22

	// ---------------------------------------------------

	VSPLTW $3, Y0, YDIG // VREPF Y0 is input

	//	VMLHF X0, YDIG, ADD1H
	//	VMLHF X1, YDIG, ADD2H
	//	VMLF  X0, YDIG, ADD1
	//	VMLF  X1, YDIG, ADD2
	//
	VMULT(X0, YDIG, ADD1, ADD1H)
	VMULT(X1, YDIG, ADD2, ADD2H)

	VSPLTISW $1, ONE
	VSPLTW $2, Y0, YDIG // VREPF

	//	VMALF  X0, YDIG, ADD1H, ADD3
	//	VMALF  X1, YDIG, ADD2H, ADD4
	//	VMALHF X0, YDIG, ADD1H, ADD3H // ADD1H Free
	//	VMALHF X1, YDIG, ADD2H, ADD4H // ADD2H Free
	VMULT_ADD(X0, YDIG, ADD1H, ONE, ADD3, ADD3H)
	VMULT_ADD(X1, YDIG, ADD2H, ONE, ADD4, ADD4H)

	LXVD2X   (R17)(CPOOL), SEL1
	VSPLTISB $0, ZER               // VZERO ZER
	VPERM    ZER, ADD1, SEL1, RED3 // [d0 0 0 d0]

	VSLDOI $12, ADD2, ADD1, T0 // ADD1 Free	// VSLDB
	VSLDOI $12, ZER, ADD2, T1  // ADD2 Free	// VSLDB

	VADDCUQ  T0, ADD3, CAR1     // VACCQ
	VADDUQM  T0, ADD3, T0       // ADD3 Free	// VAQ
	VADDECUQ T1, ADD4, CAR1, T2 // VACCCQ
	VADDEUQM T1, ADD4, CAR1, T1 // ADD4 Free	// VACQ

	LXVD2X  (R18)(CPOOL), SEL2
	LXVD2X  (R19)(CPOOL), SEL3
	LXVD2X  (R20)(CPOOL), SEL4
	VPERM   RED3, T0, SEL2, RED1 // [d0  0 d1 d0]
	VPERM   RED3, T0, SEL3, RED2 // [ 0 d1 d0 d1]
	VPERM   RED3, T0, SEL4, RED3 // [ 0  0 d1 d0]
	VSUBUQM RED2, RED3, RED2     // Guaranteed not to underflow -->? // VSQ

	VSLDOI $12, T1, T0, T0 // VSLDB
	VSLDOI $12, T2, T1, T1 // VSLDB

	VADDCUQ  T0, ADD3H, CAR1     // VACCQ
	VADDUQM  T0, ADD3H, T0       // VAQ
	VADDECUQ T1, ADD4H, CAR1, T2 // VACCCQ
	VADDEUQM T1, ADD4H, CAR1, T1 // VACQ

	// ---------------------------------------------------

	VSPLTW $1, Y0, YDIG                // VREPF

	//	VMALHF X0, YDIG, T0, ADD1H
	//	VMALHF X1, YDIG, T1, ADD2H
	//	VMALF  X0, YDIG, T0, ADD1  // T0 Free->ADD1
	//	VMALF  X1, YDIG, T1, ADD2  // T1 Free->ADD2
	VMULT_ADD(X0, YDIG, T0, ONE, ADD1, ADD1H)
	VMULT_ADD(X1, YDIG, T1, ONE, ADD2, ADD2H)

	VSPLTW $0, Y0, YDIG // VREPF

	//	VMALF  X0, YDIG, ADD1H, ADD3
	//	VMALF  X1, YDIG, ADD2H, ADD4
	//	VMALHF X0, YDIG, ADD1H, ADD3H // ADD1H Free->ADD3H
	//	VMALHF X1, YDIG, ADD2H, ADD4H // ADD2H Free->ADD4H , YDIG Free->ZER
	VMULT_ADD(X0, YDIG, ADD1H, ONE, ADD3, ADD3H)
	VMULT_ADD(X1, YDIG, ADD2H, ONE, ADD4, ADD4H)

	VSPLTISB $0, ZER               // VZERO ZER
	LXVD2X   (R17)(CPOOL), SEL1
	VPERM    ZER, ADD1, SEL1, RED3 // [d0 0 0 d0]

	VSLDOI $12, ADD2, ADD1, T0 // ADD1 Free->T0		// VSLDB
	VSLDOI $12, T2, ADD2, T1   // ADD2 Free->T1, T2 Free	// VSLDB

	VADDCUQ  T0, RED1, CAR1     // VACCQ
	VADDUQM  T0, RED1, T0       // VAQ
	VADDECUQ T1, RED2, CAR1, T2 // VACCCQ
	VADDEUQM T1, RED2, CAR1, T1 // VACQ

	VADDCUQ  T0, ADD3, CAR1       // VACCQ
	VADDUQM  T0, ADD3, T0         // VAQ
	VADDECUQ T1, ADD4, CAR1, CAR2 // VACCCQ
	VADDEUQM T1, ADD4, CAR1, T1   // VACQ
	VADDUQM  T2, CAR2, T2         // VAQ

	LXVD2X  (R18)(CPOOL), SEL2
	LXVD2X  (R19)(CPOOL), SEL3
	LXVD2X  (R20)(CPOOL), SEL4
	VPERM   RED3, T0, SEL2, RED1 // [d0  0 d1 d0]
	VPERM   RED3, T0, SEL3, RED2 // [ 0 d1 d0 d1]
	VPERM   RED3, T0, SEL4, RED3 // [ 0  0 d1 d0]
	VSUBUQM RED2, RED3, RED2     // Guaranteed not to underflow	// VSQ

	VSLDOI $12, T1, T0, T0 // VSLDB
	VSLDOI $12, T2, T1, T1 // VSLDB

	VADDCUQ  T0, ADD3H, CAR1     // VACCQ
	VADDUQM  T0, ADD3H, T0       // VAQ
	VADDECUQ T1, ADD4H, CAR1, T2 // VACCCQ
	VADDEUQM T1, ADD4H, CAR1, T1 // VACQ

	// ---------------------------------------------------

	VSPLTW $3, Y1, YDIG                // VREPF

	//	VMALHF X0, YDIG, T0, ADD1H
	//	VMALHF X1, YDIG, T1, ADD2H
	//	VMALF  X0, YDIG, T0, ADD1
	//	VMALF  X1, YDIG, T1, ADD2
	VMULT_ADD(X0, YDIG, T0, ONE, ADD1, ADD1H)
	VMULT_ADD(X1, YDIG, T1, ONE, ADD2, ADD2H)

	VSPLTW $2, Y1, YDIG // VREPF

	//	VMALF  X0, YDIG, ADD1H, ADD3
	//	VMALF  X1, YDIG, ADD2H, ADD4
	//	VMALHF X0, YDIG, ADD1H, ADD3H // ADD1H Free
	//	VMALHF X1, YDIG, ADD2H, ADD4H // ADD2H Free
	VMULT_ADD(X0, YDIG, ADD1H, ONE, ADD3, ADD3H)
	VMULT_ADD(X1, YDIG, ADD2H, ONE, ADD4, ADD4H)

	LXVD2X   (R17)(CPOOL), SEL1
	VSPLTISB $0, ZER               // VZERO ZER
	LXVD2X   (R17)(CPOOL), SEL1
	VPERM    ZER, ADD1, SEL1, RED3 // [d0 0 0 d0]

	VSLDOI $12, ADD2, ADD1, T0 // ADD1 Free		// VSLDB
	VSLDOI $12, T2, ADD2, T1   // ADD2 Free		// VSLDB

	VADDCUQ  T0, RED1, CAR1     // VACCQ
	VADDUQM  T0, RED1, T0       // VAQ
	VADDECUQ T1, RED2, CAR1, T2 // VACCCQ
	VADDEUQM T1, RED2, CAR1, T1 // VACQ

	VADDCUQ  T0, ADD3, CAR1       // VACCQ
	VADDUQM  T0, ADD3, T0         // VAQ
	VADDECUQ T1, ADD4, CAR1, CAR2 // VACCCQ
	VADDEUQM T1, ADD4, CAR1, T1   // VACQ
	VADDUQM  T2, CAR2, T2         // VAQ

	LXVD2X  (R18)(CPOOL), SEL2
	LXVD2X  (R19)(CPOOL), SEL3
	LXVD2X  (R20)(CPOOL), SEL4
	VPERM   RED3, T0, SEL2, RED1 // [d0  0 d1 d0]
	VPERM   RED3, T0, SEL3, RED2 // [ 0 d1 d0 d1]
	VPERM   RED3, T0, SEL4, RED3 // [ 0  0 d1 d0]
	VSUBUQM RED2, RED3, RED2     // Guaranteed not to underflow	// VSQ

	VSLDOI $12, T1, T0, T0 // VSLDB
	VSLDOI $12, T2, T1, T1 // VSLDB

	VADDCUQ  T0, ADD3H, CAR1     // VACCQ
	VADDUQM  T0, ADD3H, T0       // VAQ
	VADDECUQ T1, ADD4H, CAR1, T2 // VACCCQ
	VADDEUQM T1, ADD4H, CAR1, T1 // VACQ

	// ---------------------------------------------------

	VSPLTW $1, Y1, YDIG                // VREPF

	//	VMALHF X0, YDIG, T0, ADD1H
	//	VMALHF X1, YDIG, T1, ADD2H
	//	VMALF  X0, YDIG, T0, ADD1
	//	VMALF  X1, YDIG, T1, ADD2
	VMULT_ADD(X0, YDIG, T0, ONE, ADD1, ADD1H)
	VMULT_ADD(X1, YDIG, T1, ONE, ADD2, ADD2H)

	VSPLTW $0, Y1, YDIG // VREPF

	//	VMALF  X0, YDIG, ADD1H, ADD3
	//	VMALF  X1, YDIG, ADD2H, ADD4
	//	VMALHF X0, YDIG, ADD1H, ADD3H
	//	VMALHF X1, YDIG, ADD2H, ADD4H
	VMULT_ADD(X0, YDIG, ADD1H, ONE, ADD3, ADD3H)
	VMULT_ADD(X1, YDIG, ADD2H, ONE, ADD4, ADD4H)

	VSPLTISB $0, ZER               // VZERO ZER
	LXVD2X   (R17)(CPOOL), SEL1
	VPERM    ZER, ADD1, SEL1, RED3 // [d0 0 0 d0]

	VSLDOI $12, ADD2, ADD1, T0 // VSLDB
	VSLDOI $12, T2, ADD2, T1   // VSLDB

	VADDCUQ  T0, RED1, CAR1     // VACCQ
	VADDUQM  T0, RED1, T0       // VAQ
	VADDECUQ T1, RED2, CAR1, T2 // VACCCQ
	VADDEUQM T1, RED2, CAR1, T1 // VACQ

	VADDCUQ  T0, ADD3, CAR1       // VACCQ
	VADDUQM  T0, ADD3, T0         // VAQ
	VADDECUQ T1, ADD4, CAR1, CAR2 // VACCCQ
	VADDEUQM T1, ADD4, CAR1, T1   // VACQ
	VADDUQM  T2, CAR2, T2         // VAQ

	LXVD2X  (R21)(CPOOL), SEL5
	LXVD2X  (R22)(CPOOL), SEL6
	VPERM   T0, RED3, SEL5, RED2 // [d1 d0 d1 d0]
	VPERM   T0, RED3, SEL6, RED1 // [ 0 d1 d0  0]
	VSUBUQM RED2, RED1, RED2     // Guaranteed not to underflow	// VSQ

	VSLDOI $12, T1, T0, T0 // VSLDB
	VSLDOI $12, T2, T1, T1 // VSLDB

	VADDCUQ  T0, ADD3H, CAR1     // VACCQ
	VADDUQM  T0, ADD3H, T0       // VAQ
	VADDECUQ T1, ADD4H, CAR1, T2 // VACCCQ
	VADDEUQM T1, ADD4H, CAR1, T1 // VACQ

	VADDCUQ  T0, RED1, CAR1       // VACCQ
	VADDUQM  T0, RED1, T0         // VAQ
	VADDECUQ T1, RED2, CAR1, CAR2 // VACCCQ
	VADDEUQM T1, RED2, CAR1, T1   // VACQ
	VADDUQM  T2, CAR2, T2         // VAQ

	// ---------------------------------------------------

	VSPLTISB $0, RED3            // VZERO   RED3
	VSUBCUQ  T0, P0, CAR1        // VSCBIQ
	VSUBUQM  T0, P0, ADD1H       // VSQ
	VSUBECUQ T1, P1, CAR1, CAR2  // VSBCBIQ
	VSUBEUQM T1, P1, CAR1, ADD2H // VSBIQ
	VSUBEUQM T2, RED3, CAR2, T2  // VSBIQ

	// what output to use, ADD2H||ADD1H or T1||T0?
	VSEL ADD1H, T0, T2, T0
	VSEL ADD2H, T1, T2, T1
	RET

#undef CPOOL

#undef X0
#undef X1
#undef Y0
#undef Y1
#undef T0
#undef T1
#undef P0
#undef P1

#undef SEL1
#undef SEL2
#undef SEL3
#undef SEL4
#undef SEL5
#undef SEL6

#undef YDIG
#undef ADD1H
#undef ADD2H
#undef ADD3
#undef ADD4
#undef RED1
#undef RED2
#undef RED3
#undef T2
#undef ADD1
#undef ADD2
#undef ADD3H
#undef ADD4H
#undef ZER
#undef CAR1
#undef CAR2

#undef TMP1
#undef TMP2

#define p256SubInternal(T1, T0, X1, X0, Y1, Y0) \
	VSPLTISB $0, ZER            \ // VZERO
	VSUBCUQ  X0, Y0, CAR1       \
	VSUBUQM  X0, Y0, T0         \
	VSUBECUQ X1, Y1, CAR1, SEL1 \
	VSUBEUQM X1, Y1, CAR1, T1   \
	VSUBUQM  ZER, SEL1, SEL1    \ // VSQ
	                            \
	VADDCUQ  T0, PL, CAR1       \ // VACCQ
	VADDUQM  T0, PL, TT0        \ // VAQ
	VADDEUQM T1, PH, CAR1, TT1  \ // VACQ
	                            \
	VSEL     TT0, T0, SEL1, T0  \
	VSEL     TT1, T1, SEL1, T1  \

#define p256AddInternal(T1, T0, X1, X0, Y1, Y0) \
	VADDCUQ  X0, Y0, CAR1        \
	VADDUQM  X0, Y0, T0          \
	VADDECUQ X1, Y1, CAR1, T2    \ // VACCCQ
	VADDEUQM X1, Y1, CAR1, T1    \
	                             \
	VSPLTISB $0, ZER             \
	VSUBCUQ  T0, PL, CAR1        \ // VSCBIQ
	VSUBUQM  T0, PL, TT0         \
	VSUBECUQ T1, PH, CAR1, CAR2  \ // VSBCBIQ
	VSUBEUQM T1, PH, CAR1, TT1   \ // VSBIQ
	VSUBEUQM T2, ZER, CAR2, SEL1 \
	                             \
	VSEL     TT0, T0, SEL1, T0   \
	VSEL     TT1, T1, SEL1, T1

#define p256HalfInternal(T1, T0, X1, X0) \
	VSPLTISB $0, ZER            \
	VSUBEUQM ZER, ZER, X0, SEL1 \
	                            \
	VADDCUQ  X0, PL, CAR1       \
	VADDUQM  X0, PL, T0         \
	VADDECUQ X1, PH, CAR1, T2   \
	VADDEUQM X1, PH, CAR1, T1   \
	                            \
	VSEL     T0, X0, SEL1, T0   \
	VSEL     T1, X1, SEL1, T1   \
	VSEL     T2, ZER, SEL1, T2  \
	                            \
	VSLDOI   $15, T2, ZER, TT1  \
	VSLDOI   $15, T1, ZER, TT0  \
	VSPLTISB $1, SEL1           \
	VSR      T0, SEL1, T0       \ // VSRL
	VSR      T1, SEL1, T1       \
	VSPLTISB $7, SEL1           \ // VREPIB
	VSL      TT0, SEL1, TT0     \
	VSL      TT1, SEL1, TT1     \
	VOR      T0, TT0, T0        \
	VOR      T1, TT1, T1

#define res_ptr R3
#define x_ptr   R4
#define y_ptr   R5
#define CPOOL   R7
#define TEMP    R8
#define N       R9

// Parameters
#define X0    V0
#define X1    V1
#define Y0    V2
#define Y1    V3
#define T0    V4
#define T1    V5

// Constants
#define P0    V30
#define P1    V31
// func p256MulAsm(res, in1, in2 *p256Element)
TEXT ·p256Mul(SB), NOSPLIT, $0-24
	MOVD res+0(FP), res_ptr
	MOVD in1+8(FP), x_ptr
	MOVD in2+16(FP), y_ptr
	MOVD $16, R16
	MOVD $32, R17

	MOVD $p256mul<>+0x00(SB), CPOOL


	LXVD2X (R0)(x_ptr), X0
	LXVD2X (R16)(x_ptr), X1

	XXPERMDI X0, X0, $2, X0
	XXPERMDI X1, X1, $2, X1

	LXVD2X (R0)(y_ptr), Y0
	LXVD2X (R16)(y_ptr), Y1

	XXPERMDI Y0, Y0, $2, Y0
	XXPERMDI Y1, Y1, $2, Y1

	LXVD2X (R16)(CPOOL), P1
	LXVD2X (R0)(CPOOL), P0

	CALL p256MulInternal<>(SB)

	MOVD $p256mul<>+0x00(SB), CPOOL

	XXPERMDI T0, T0, $2, T0
	XXPERMDI T1, T1, $2, T1
	STXVD2X T0, (R0)(res_ptr)
	STXVD2X T1, (R16)(res_ptr)
	RET

// func p256Sqr(res, in *p256Element, n int)
TEXT ·p256Sqr(SB), NOSPLIT, $0-24
	MOVD res+0(FP), res_ptr
	MOVD in+8(FP), x_ptr
	MOVD $16, R16
	MOVD $32, R17

	MOVD $p256mul<>+0x00(SB), CPOOL

	LXVD2X (R0)(x_ptr), X0
	LXVD2X (R16)(x_ptr), X1

	XXPERMDI X0, X0, $2, X0
	XXPERMDI X1, X1, $2, X1

sqrLoop:
	// Sqr uses same value for both

	VOR	X0, X0, Y0
	VOR	X1, X1, Y1

	LXVD2X (R16)(CPOOL), P1
	LXVD2X (R0)(CPOOL), P0

	CALL p256MulInternal<>(SB)

	MOVD	n+16(FP), N
	ADD	$-1, N
	CMP	$0, N
	BEQ	done
	MOVD	N, n+16(FP)	// Save counter to avoid clobber
	VOR	T0, T0, X0
	VOR	T1, T1, X1
	BR	sqrLoop

done:
	MOVD $p256mul<>+0x00(SB), CPOOL

	XXPERMDI T0, T0, $2, T0
	XXPERMDI T1, T1, $2, T1
	STXVD2X T0, (R0)(res_ptr)
	STXVD2X T1, (R16)(res_ptr)
	RET

#undef res_ptr
#undef x_ptr
#undef y_ptr
#undef CPOOL

#undef X0
#undef X1
#undef Y0
#undef Y1
#undef T0
#undef T1
#undef P0
#undef P1

#define P3ptr   R3
#define P1ptr   R4
#define P2ptr   R5
#define CPOOL   R7

// Temporaries in REGs
#define Y2L    V15
#define Y2H    V16
#define T1L    V17
#define T1H    V18
#define T2L    V19
#define T2H    V20
#define T3L    V21
#define T3H    V22
#define T4L    V23
#define T4H    V24

// Temps for Sub and Add
#define TT0  V11
#define TT1  V12
#define T2   V13

// p256MulAsm Parameters
#define X0    V0
#define X1    V1
#define Y0    V2
#define Y1    V3
#define T0    V4
#define T1    V5

#define PL    V30
#define PH    V31

// Names for zero/sel selects
#define X1L    V0
#define X1H    V1
#define Y1L    V2 // p256MulAsmParmY
#define Y1H    V3 // p256MulAsmParmY
#define Z1L    V4
#define Z1H    V5
#define X2L    V0
#define X2H    V1
#define Z2L    V4
#define Z2H    V5
#define X3L    V17 // T1L
#define X3H    V18 // T1H
#define Y3L    V21 // T3L
#define Y3H    V22 // T3H
#define Z3L    V25
#define Z3H    V26

#define ZER   V6
#define SEL1  V7
#define CAR1  V8
#define CAR2  V9
/* *
 * Three operand formula:
 * Source: 2004 Hankerson–Menezes–Vanstone, page 91.
 * T1 = Z1²
 * T2 = T1*Z1
 * T1 = T1*X2
 * T2 = T2*Y2
 * T1 = T1-X1
 * T2 = T2-Y1
 * Z3 = Z1*T1
 * T3 = T1²
 * T4 = T3*T1
 * T3 = T3*X1
 * T1 = 2*T3
 * X3 = T2²
 * X3 = X3-T1
 * X3 = X3-T4
 * T3 = T3-X3
 * T3 = T3*T2
 * T4 = T4*Y1
 * Y3 = T3-T4

 * Three operand formulas, but with MulInternal X,Y used to store temps
X=Z1; Y=Z1; MUL;T-   // T1 = Z1²      T1
X=T ; Y-  ; MUL;T2=T // T2 = T1*Z1    T1   T2
X-  ; Y=X2; MUL;T1=T // T1 = T1*X2    T1   T2
X=T2; Y=Y2; MUL;T-   // T2 = T2*Y2    T1   T2
SUB(T2<T-Y1)         // T2 = T2-Y1    T1   T2
SUB(Y<T1-X1)         // T1 = T1-X1    T1   T2
X=Z1; Y- ;  MUL;Z3:=T// Z3 = Z1*T1         T2
X=Y;  Y- ;  MUL;X=T  // T3 = T1*T1         T2
X- ;  Y- ;  MUL;T4=T // T4 = T3*T1         T2        T4
X- ;  Y=X1; MUL;T3=T // T3 = T3*X1         T2   T3   T4
ADD(T1<T+T)          // T1 = T3+T3    T1   T2   T3   T4
X=T2; Y=T2; MUL;T-   // X3 = T2*T2    T1   T2   T3   T4
SUB(T<T-T1)          // X3 = X3-T1    T1   T2   T3   T4
SUB(T<T-T4) X3:=T    // X3 = X3-T4         T2   T3   T4
SUB(X<T3-T)          // T3 = T3-X3         T2   T3   T4
X- ;  Y- ;  MUL;T3=T // T3 = T3*T2         T2   T3   T4
X=T4; Y=Y1; MUL;T-   // T4 = T4*Y1              T3   T4
SUB(T<T3-T) Y3:=T    // Y3 = T3-T4              T3   T4

	*/
//
// V27 is clobbered by p256MulInternal so must be
// saved in a temp.
//
// func p256PointAddAffineAsm(res, in1 *P256Point, in2 *p256AffinePoint, sign, sel, zero int)
TEXT ·p256PointAddAffineAsm(SB), NOSPLIT, $16-48
	MOVD res+0(FP), P3ptr
	MOVD in1+8(FP), P1ptr
	MOVD in2+16(FP), P2ptr

	MOVD $p256mul<>+0x00(SB), CPOOL

	MOVD $16, R16
	MOVD $32, R17
	MOVD $48, R18
	MOVD $64, R19
	MOVD $80, R20
	MOVD $96, R21
	MOVD $112, R22
	MOVD $128, R23
	MOVD $144, R24
	MOVD $160, R25
	MOVD $104, R26 // offset of sign+24(FP)

	LXVD2X (R16)(CPOOL), PH
	LXVD2X (R0)(CPOOL), PL

	LXVD2X (R17)(P2ptr), Y2L
	LXVD2X (R18)(P2ptr), Y2H
	XXPERMDI Y2H, Y2H, $2, Y2H
	XXPERMDI Y2L, Y2L, $2, Y2L

	// Equivalent of VLREPG sign+24(FP), SEL1
	LXVDSX   (R1)(R26), SEL1
	VSPLTISB $0, ZER
	VCMPEQUD SEL1, ZER, SEL1

	VSUBCUQ  PL, Y2L, CAR1
	VSUBUQM  PL, Y2L, T1L
	VSUBEUQM PH, Y2H, CAR1, T1H

	VSEL T1L, Y2L, SEL1, Y2L
	VSEL T1H, Y2H, SEL1, Y2H

/* *
 * Three operand formula:
 * Source: 2004 Hankerson–Menezes–Vanstone, page 91.
 */
	// X=Z1; Y=Z1; MUL; T-   // T1 = Z1²      T1
	LXVD2X (R19)(P1ptr), X0     // Z1H
	LXVD2X (R20)(P1ptr), X1     // Z1L
	XXPERMDI X0, X0, $2, X0
	XXPERMDI X1, X1, $2, X1
	VOR    X0, X0, Y0
	VOR    X1, X1, Y1
	CALL   p256MulInternal<>(SB)

	// X=T ; Y-  ; MUL; T2=T // T2 = T1*Z1    T1   T2
	VOR  T0, T0, X0
	VOR  T1, T1, X1
	CALL p256MulInternal<>(SB)
	VOR  T0, T0, T2L
	VOR  T1, T1, T2H

	// X-  ; Y=X2; MUL; T1=T // T1 = T1*X2    T1   T2
	MOVD   in2+16(FP), P2ptr
	LXVD2X (R0)(P2ptr), Y0      // X2H
	LXVD2X (R16)(P2ptr), Y1     // X2L
	XXPERMDI Y0, Y0, $2, Y0
	XXPERMDI Y1, Y1, $2, Y1
	CALL   p256MulInternal<>(SB)
	VOR    T0, T0, T1L
	VOR    T1, T1, T1H

	// X=T2; Y=Y2; MUL; T-   // T2 = T2*Y2    T1   T2
	VOR  T2L, T2L, X0
	VOR  T2H, T2H, X1
	VOR  Y2L, Y2L, Y0
	VOR  Y2H, Y2H, Y1
	CALL p256MulInternal<>(SB)

	// SUB(T2<T-Y1)          // T2 = T2-Y1    T1   T2
	MOVD   in1+8(FP), P1ptr
	LXVD2X (R17)(P1ptr), Y1L
	LXVD2X (R18)(P1ptr), Y1H
	XXPERMDI Y1H, Y1H, $2, Y1H
	XXPERMDI Y1L, Y1L, $2, Y1L
	p256SubInternal(T2H,T2L,T1,T0,Y1H,Y1L)

	// SUB(Y<T1-X1)          // T1 = T1-X1    T1   T2
	LXVD2X (R0)(P1ptr), X1L
	LXVD2X (R16)(P1ptr), X1H
	XXPERMDI X1H, X1H, $2, X1H
	XXPERMDI X1L, X1L, $2, X1L
	p256SubInternal(Y1,Y0,T1H,T1L,X1H,X1L)

	// X=Z1; Y- ;  MUL; Z3:=T// Z3 = Z1*T1         T2
	LXVD2X (R19)(P1ptr), X0     // Z1H
	LXVD2X (R20)(P1ptr), X1     // Z1L
	XXPERMDI X0, X0, $2, X0
	XXPERMDI X1, X1, $2, X1
	CALL   p256MulInternal<>(SB)

	VOR T0, T0, Z3L
	VOR T1, T1, Z3H

	// X=Y;  Y- ;  MUL; X=T  // T3 = T1*T1         T2
	VOR  Y0, Y0, X0
	VOR  Y1, Y1, X1
	CALL p256MulInternal<>(SB)
	VOR  T0, T0, X0
	VOR  T1, T1, X1

	// X- ;  Y- ;  MUL; T4=T // T4 = T3*T1         T2        T4
	CALL p256MulInternal<>(SB)
	VOR  T0, T0, T4L
	VOR  T1, T1, T4H

	// X- ;  Y=X1; MUL; T3=T // T3 = T3*X1         T2   T3   T4
	MOVD   in1+8(FP), P1ptr
	LXVD2X (R0)(P1ptr), Y0      // X1H
	LXVD2X (R16)(P1ptr), Y1     // X1L
	XXPERMDI Y1, Y1, $2, Y1
	XXPERMDI Y0, Y0, $2, Y0
	CALL   p256MulInternal<>(SB)
	VOR    T0, T0, T3L
	VOR    T1, T1, T3H

	// ADD(T1<T+T)           // T1 = T3+T3    T1   T2   T3   T4
	p256AddInternal(T1H,T1L, T1,T0,T1,T0)

	// X=T2; Y=T2; MUL; T-   // X3 = T2*T2    T1   T2   T3   T4
	VOR  T2L, T2L, X0
	VOR  T2H, T2H, X1
	VOR  T2L, T2L, Y0
	VOR  T2H, T2H, Y1
	CALL p256MulInternal<>(SB)

	// SUB(T<T-T1)           // X3 = X3-T1    T1   T2   T3   T4  (T1 = X3)
	p256SubInternal(T1,T0,T1,T0,T1H,T1L)

	// SUB(T<T-T4) X3:=T     // X3 = X3-T4         T2   T3   T4
	p256SubInternal(T1,T0,T1,T0,T4H,T4L)
	VOR T0, T0, X3L
	VOR T1, T1, X3H

	// SUB(X<T3-T)           // T3 = T3-X3         T2   T3   T4
	p256SubInternal(X1,X0,T3H,T3L,T1,T0)

	// X- ;  Y- ;  MUL; T3=T // T3 = T3*T2         T2   T3   T4
	CALL p256MulInternal<>(SB)
	VOR  T0, T0, T3L
	VOR  T1, T1, T3H

	// X=T4; Y=Y1; MUL; T-   // T4 = T4*Y1              T3   T4
	VOR    T4L, T4L, X0
	VOR    T4H, T4H, X1
	MOVD   in1+8(FP), P1ptr
	LXVD2X (R17)(P1ptr), Y0     // Y1H
	LXVD2X (R18)(P1ptr), Y1     // Y1L
	XXPERMDI Y0, Y0, $2, Y0
	XXPERMDI Y1, Y1, $2, Y1
	CALL   p256MulInternal<>(SB)

	// SUB(T<T3-T) Y3:=T     // Y3 = T3-T4              T3   T4  (T3 = Y3)
	p256SubInternal(Y3H,Y3L,T3H,T3L,T1,T0)

	//	if (sel == 0) {
	//		copy(P3.x[:], X1)
	//		copy(P3.y[:], Y1)
	//		copy(P3.z[:], Z1)
	//	}

	LXVD2X (R0)(P1ptr), X1L
	LXVD2X (R16)(P1ptr), X1H
	XXPERMDI X1H, X1H, $2, X1H
	XXPERMDI X1L, X1L, $2, X1L

	// Y1 already loaded, left over from addition
	LXVD2X (R19)(P1ptr), Z1L
	LXVD2X (R20)(P1ptr), Z1H
	XXPERMDI Z1H, Z1H, $2, Z1H
	XXPERMDI Z1L, Z1L, $2, Z1L

	MOVD     $112, R26        // Get offset to sel+32
	LXVDSX   (R1)(R26), SEL1
	VSPLTISB $0, ZER
	VCMPEQUD SEL1, ZER, SEL1

	VSEL X3L, X1L, SEL1, X3L
	VSEL X3H, X1H, SEL1, X3H
	VSEL Y3L, Y1L, SEL1, Y3L
	VSEL Y3H, Y1H, SEL1, Y3H
	VSEL Z3L, Z1L, SEL1, Z3L
	VSEL Z3H, Z1H, SEL1, Z3H

	MOVD   in2+16(FP), P2ptr
	LXVD2X (R0)(P2ptr), X2L
	LXVD2X (R16)(P2ptr), X2H
	XXPERMDI X2H, X2H, $2, X2H
	XXPERMDI X2L, X2L, $2, X2L

	// Y2 already loaded
	LXVD2X (R23)(CPOOL), Z2L
	LXVD2X (R24)(CPOOL), Z2H

	MOVD     $120, R26        // Get the value from zero+40(FP)
	LXVDSX   (R1)(R26), SEL1
	VSPLTISB $0, ZER
	VCMPEQUD SEL1, ZER, SEL1

	VSEL X3L, X2L, SEL1, X3L
	VSEL X3H, X2H, SEL1, X3H
	VSEL Y3L, Y2L, SEL1, Y3L
	VSEL Y3H, Y2H, SEL1, Y3H
	VSEL Z3L, Z2L, SEL1, Z3L
	VSEL Z3H, Z2H, SEL1, Z3H

	// Reorder the bytes so they can be stored using STXVD2X.
	MOVD    res+0(FP), P3ptr
	XXPERMDI X3H, X3H, $2, X3H
	XXPERMDI X3L, X3L, $2, X3L
	XXPERMDI Y3H, Y3H, $2, Y3H
	XXPERMDI Y3L, Y3L, $2, Y3L
	XXPERMDI Z3H, Z3H, $2, Z3H
	XXPERMDI Z3L, Z3L, $2, Z3L
	STXVD2X X3L, (R0)(P3ptr)
	STXVD2X X3H, (R16)(P3ptr)
	STXVD2X Y3L, (R17)(P3ptr)
	STXVD2X Y3H, (R18)(P3ptr)
	STXVD2X Z3L, (R19)(P3ptr)
	STXVD2X Z3H, (R20)(P3ptr)

	RET

#undef P3ptr
#undef P1ptr
#undef P2ptr
#undef CPOOL

#undef Y2L
#undef Y2H
#undef T1L
#undef T1H
#undef T2L
#undef T2H
#undef T3L
#undef T3H
#undef T4L
#undef T4H

#undef TT0
#undef TT1
#undef T2

#undef X0
#undef X1
#undef Y0
#undef Y1
#undef T0
#undef T1

#undef PL
#undef PH

#undef X1L
#undef X1H
#undef Y1L
#undef Y1H
#undef Z1L
#undef Z1H
#undef X2L
#undef X2H
#undef Z2L
#undef Z2H
#undef X3L
#undef X3H
#undef Y3L
#undef Y3H
#undef Z3L
#undef Z3H

#undef ZER
#undef SEL1
#undef CAR1
#undef CAR2

// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian.html#doubling-dbl-2007-bl
// http://www.hyperelliptic.org/EFD/g1p/auto-shortw.html
// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-projective-3.html
#define P3ptr   R3
#define P1ptr   R4
#define CPOOL   R7

// Temporaries in REGs
#define X3L    V15
#define X3H    V16
#define Y3L    V17
#define Y3H    V18
#define T1L    V19
#define T1H    V20
#define T2L    V21
#define T2H    V22
#define T3L    V23
#define T3H    V24

#define X1L    V6
#define X1H    V7
#define Y1L    V8
#define Y1H    V9
#define Z1L    V10
#define Z1H    V11

// Temps for Sub and Add
#define TT0  V11
#define TT1  V12
#define T2   V13

// p256MulAsm Parameters
#define X0    V0
#define X1    V1
#define Y0    V2
#define Y1    V3
#define T0    V4
#define T1    V5

#define PL    V30
#define PH    V31

#define Z3L    V23
#define Z3H    V24

#define ZER   V26
#define SEL1  V27
#define CAR1  V28
#define CAR2  V29
/*
 * http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-3.html#doubling-dbl-2004-hmv
 * Cost: 4M + 4S + 1*half + 5add + 2*2 + 1*3.
 * Source: 2004 Hankerson–Menezes–Vanstone, page 91.
 * 	A  = 3(X₁-Z₁²)×(X₁+Z₁²)
 * 	B  = 2Y₁
 * 	Z₃ = B×Z₁
 * 	C  = B²
 * 	D  = C×X₁
 * 	X₃ = A²-2D
 * 	Y₃ = (D-X₃)×A-C²/2
 *
 * Three-operand formula:
 *       T1 = Z1²
 *       T2 = X1-T1
 *       T1 = X1+T1
 *       T2 = T2*T1
 *       T2 = 3*T2
 *       Y3 = 2*Y1
 *       Z3 = Y3*Z1
 *       Y3 = Y3²
 *       T3 = Y3*X1
 *       Y3 = Y3²
 *       Y3 = half*Y3
 *       X3 = T2²
 *       T1 = 2*T3
 *       X3 = X3-T1
 *       T1 = T3-X3
 *       T1 = T1*T2
 *       Y3 = T1-Y3
 */
// p256PointDoubleAsm(res, in1 *p256Point)
TEXT ·p256PointDoubleAsm(SB), NOSPLIT, $0-16
	MOVD res+0(FP), P3ptr
	MOVD in+8(FP), P1ptr

	MOVD $p256mul<>+0x00(SB), CPOOL

	MOVD $16, R16
	MOVD $32, R17
	MOVD $48, R18
	MOVD $64, R19
	MOVD $80, R20

	LXVD2X (R16)(CPOOL), PH
	LXVD2X (R0)(CPOOL), PL

	// X=Z1; Y=Z1; MUL; T-    // T1 = Z1²
	LXVD2X (R19)(P1ptr), X0 // Z1H
	LXVD2X (R20)(P1ptr), X1 // Z1L

	XXPERMDI X0, X0, $2, X0
	XXPERMDI X1, X1, $2, X1

	VOR  X0, X0, Y0
	VOR  X1, X1, Y1
	CALL p256MulInternal<>(SB)

	// SUB(X<X1-T)            // T2 = X1-T1
	LXVD2X (R0)(P1ptr), X1L
	LXVD2X (R16)(P1ptr), X1H
	XXPERMDI X1L, X1L, $2, X1L
	XXPERMDI X1H, X1H, $2, X1H

	p256SubInternal(X1,X0,X1H,X1L,T1,T0)

	// ADD(Y<X1+T)            // T1 = X1+T1
	p256AddInternal(Y1,Y0,X1H,X1L,T1,T0)

	// X-  ; Y-  ; MUL; T-    // T2 = T2*T1
	CALL p256MulInternal<>(SB)

	// ADD(T2<T+T); ADD(T2<T2+T)  // T2 = 3*T2
	p256AddInternal(T2H,T2L,T1,T0,T1,T0)
	p256AddInternal(T2H,T2L,T2H,T2L,T1,T0)

	// ADD(X<Y1+Y1)           // Y3 = 2*Y1
	LXVD2X (R17)(P1ptr), Y1L
	LXVD2X (R18)(P1ptr), Y1H
	XXPERMDI Y1L, Y1L, $2, Y1L
	XXPERMDI Y1H, Y1H, $2, Y1H

	p256AddInternal(X1,X0,Y1H,Y1L,Y1H,Y1L)

	// X-  ; Y=Z1; MUL; Z3:=T // Z3 = Y3*Z1
	LXVD2X (R19)(P1ptr), Y0
	LXVD2X (R20)(P1ptr), Y1
	XXPERMDI Y0, Y0, $2, Y0
	XXPERMDI Y1, Y1, $2, Y1

	CALL p256MulInternal<>(SB)

	// Leave T0, T1 as is.
	XXPERMDI T0, T0, $2, TT0
	XXPERMDI T1, T1, $2, TT1
	STXVD2X TT0, (R19)(P3ptr)
	STXVD2X TT1, (R20)(P3ptr)

	// X-  ; Y=X ; MUL; T-    // Y3 = Y3²
	VOR  X0, X0, Y0
	VOR  X1, X1, Y1
	CALL p256MulInternal<>(SB)

	// X=T ; Y=X1; MUL; T3=T  // T3 = Y3*X1
	VOR    T0, T0, X0
	VOR    T1, T1, X1
	LXVD2X (R0)(P1ptr), Y0
	LXVD2X (R16)(P1ptr), Y1
	XXPERMDI Y0, Y0, $2, Y0
	XXPERMDI Y1, Y1, $2, Y1
	CALL   p256MulInternal<>(SB)
	VOR    T0, T0, T3L
	VOR    T1, T1, T3H

	// X-  ; Y=X ; MUL; T-    // Y3 = Y3²
	VOR  X0, X0, Y0
	VOR  X1, X1, Y1
	CALL p256MulInternal<>(SB)

	// HAL(Y3<T)              // Y3 = half*Y3
	p256HalfInternal(Y3H,Y3L, T1,T0)

	// X=T2; Y=T2; MUL; T-    // X3 = T2²
	VOR  T2L, T2L, X0
	VOR  T2H, T2H, X1
	VOR  T2L, T2L, Y0
	VOR  T2H, T2H, Y1
	CALL p256MulInternal<>(SB)

	// ADD(T1<T3+T3)          // T1 = 2*T3
	p256AddInternal(T1H,T1L,T3H,T3L,T3H,T3L)

	// SUB(X3<T-T1) X3:=X3    // X3 = X3-T1
	p256SubInternal(X3H,X3L,T1,T0,T1H,T1L)

	XXPERMDI X3L, X3L, $2, TT0
	XXPERMDI X3H, X3H, $2, TT1
	STXVD2X TT0, (R0)(P3ptr)
	STXVD2X TT1, (R16)(P3ptr)

	// SUB(X<T3-X3)           // T1 = T3-X3
	p256SubInternal(X1,X0,T3H,T3L,X3H,X3L)

	// X-  ; Y-  ; MUL; T-    // T1 = T1*T2
	CALL p256MulInternal<>(SB)

	// SUB(Y3<T-Y3)           // Y3 = T1-Y3
	p256SubInternal(Y3H,Y3L,T1,T0,Y3H,Y3L)

	XXPERMDI Y3L, Y3L, $2, Y3L
	XXPERMDI Y3H, Y3H, $2, Y3H
	STXVD2X Y3L, (R17)(P3ptr)
	STXVD2X Y3H, (R18)(P3ptr)
	RET

#undef P3ptr
#undef P1ptr
#undef CPOOL
#undef X3L
#undef X3H
#undef Y3L
#undef Y3H
#undef T1L
#undef T1H
#undef T2L
#undef T2H
#undef T3L
#undef T3H
#undef X1L
#undef X1H
#undef Y1L
#undef Y1H
#undef Z1L
#undef Z1H
#undef TT0
#undef TT1
#undef T2
#undef X0
#undef X1
#undef Y0
#undef Y1
#undef T0
#undef T1
#undef PL
#undef PH
#undef Z3L
#undef Z3H
#undef ZER
#undef SEL1
#undef CAR1
#undef CAR2

#define P3ptr  R3
#define P1ptr  R4
#define P2ptr  R5
#define CPOOL  R7
#define TRUE   R14
#define RES1   R9
#define RES2   R10

// Temporaries in REGs
#define T1L   V16
#define T1H   V17
#define T2L   V18
#define T2H   V19
#define U1L   V20
#define U1H   V21
#define S1L   V22
#define S1H   V23
#define HL    V24
#define HH    V25
#define RL    V26
#define RH    V27

// Temps for Sub and Add
#define ZER   V6
#define SEL1  V7
#define CAR1  V8
#define CAR2  V9
#define TT0  V11
#define TT1  V12
#define T2   V13

// p256MulAsm Parameters
#define X0    V0
#define X1    V1
#define Y0    V2
#define Y1    V3
#define T0    V4
#define T1    V5

#define PL    V30
#define PH    V31
/*
 * https://choucroutage.com/Papers/SideChannelAttacks/ctrsa-2011-brown.pdf "Software Implementation of the NIST Elliptic Curves Over Prime Fields"
 *
 * A = X₁×Z₂²
 * B = Y₁×Z₂³
 * C = X₂×Z₁²-A
 * D = Y₂×Z₁³-B
 * X₃ = D² - 2A×C² - C³
 * Y₃ = D×(A×C² - X₃) - B×C³
 * Z₃ = Z₁×Z₂×C
 *
 * Three-operand formula (adopted): http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-3.html#addition-add-1998-cmo-2
 * Temp storage: T1,T2,U1,H,Z3=X3=Y3,S1,R
 *
 * T1 = Z1*Z1
 * T2 = Z2*Z2
 * U1 = X1*T2
 * H  = X2*T1
 * H  = H-U1
 * Z3 = Z1*Z2
 * Z3 = Z3*H << store-out Z3 result reg.. could override Z1, if slices have same backing array
 *
 * S1 = Z2*T2
 * S1 = Y1*S1
 * R  = Z1*T1
 * R  = Y2*R
 * R  = R-S1
 *
 * T1 = H*H
 * T2 = H*T1
 * U1 = U1*T1
 *
 * X3 = R*R
 * X3 = X3-T2
 * T1 = 2*U1
 * X3 = X3-T1 << store-out X3 result reg
 *
 * T2 = S1*T2
 * Y3 = U1-X3
 * Y3 = R*Y3
 * Y3 = Y3-T2 << store-out Y3 result reg

	// X=Z1; Y=Z1; MUL; T-   // T1 = Z1*Z1
	// X-  ; Y=T ; MUL; R=T  // R  = Z1*T1
	// X=X2; Y-  ; MUL; H=T  // H  = X2*T1
	// X=Z2; Y=Z2; MUL; T-   // T2 = Z2*Z2
	// X-  ; Y=T ; MUL; S1=T // S1 = Z2*T2
	// X=X1; Y-  ; MUL; U1=T // U1 = X1*T2
	// SUB(H<H-T)            // H  = H-U1
	// X=Z1; Y=Z2; MUL; T-   // Z3 = Z1*Z2
	// X=T ; Y=H ; MUL; Z3:=T// Z3 = Z3*H << store-out Z3 result reg.. could override Z1, if slices have same backing array
	// X=Y1; Y=S1; MUL; S1=T // S1 = Y1*S1
	// X=Y2; Y=R ; MUL; T-   // R  = Y2*R
	// SUB(R<T-S1)           // R  = R-S1
	// X=H ; Y=H ; MUL; T-   // T1 = H*H
	// X-  ; Y=T ; MUL; T2=T // T2 = H*T1
	// X=U1; Y-  ; MUL; U1=T // U1 = U1*T1
	// X=R ; Y=R ; MUL; T-   // X3 = R*R
	// SUB(T<T-T2)           // X3 = X3-T2
	// ADD(X<U1+U1)          // T1 = 2*U1
	// SUB(T<T-X) X3:=T      // X3 = X3-T1 << store-out X3 result reg
	// SUB(Y<U1-T)           // Y3 = U1-X3
	// X=R ; Y-  ; MUL; U1=T // Y3 = R*Y3
	// X=S1; Y=T2; MUL; T-   // T2 = S1*T2
	// SUB(T<U1-T); Y3:=T    // Y3 = Y3-T2 << store-out Y3 result reg
	*/
// p256PointAddAsm(res, in1, in2 *p256Point)
TEXT ·p256PointAddAsm(SB), NOSPLIT, $16-32
	MOVD res+0(FP), P3ptr
	MOVD in1+8(FP), P1ptr
	MOVD $p256mul<>+0x00(SB), CPOOL
	MOVD $16, R16
	MOVD $32, R17
	MOVD $48, R18
	MOVD $64, R19
	MOVD $80, R20

	LXVD2X (R16)(CPOOL), PH
	LXVD2X (R0)(CPOOL), PL

	// X=Z1; Y=Z1; MUL; T-   // T1 = Z1*Z1
	LXVD2X (R19)(P1ptr), X0     // Z1L
	LXVD2X (R20)(P1ptr), X1     // Z1H
	XXPERMDI X0, X0, $2, X0
	XXPERMDI X1, X1, $2, X1
	VOR    X0, X0, Y0
	VOR    X1, X1, Y1
	CALL   p256MulInternal<>(SB)

	// X-  ; Y=T ; MUL; R=T  // R  = Z1*T1
	VOR  T0, T0, Y0
	VOR  T1, T1, Y1
	CALL p256MulInternal<>(SB)
	VOR  T0, T0, RL            // SAVE: RL
	VOR  T1, T1, RH            // SAVE: RH

	STXVD2X RH, (R1)(R17) // V27 has to be saved

	// X=X2; Y-  ; MUL; H=T  // H  = X2*T1
	MOVD   in2+16(FP), P2ptr
	LXVD2X (R0)(P2ptr), X0      // X2L
	LXVD2X (R16)(P2ptr), X1     // X2H
	XXPERMDI X0, X0, $2, X0
	XXPERMDI X1, X1, $2, X1
	CALL   p256MulInternal<>(SB)
	VOR    T0, T0, HL            // SAVE: HL
	VOR    T1, T1, HH            // SAVE: HH

	// X=Z2; Y=Z2; MUL; T-   // T2 = Z2*Z2
	MOVD   in2+16(FP), P2ptr
	LXVD2X (R19)(P2ptr), X0     // Z2L
	LXVD2X (R20)(P2ptr), X1     // Z2H
	XXPERMDI X0, X0, $2, X0
	XXPERMDI X1, X1, $2, X1
	VOR    X0, X0, Y0
	VOR    X1, X1, Y1
	CALL   p256MulInternal<>(SB)

	// X-  ; Y=T ; MUL; S1=T // S1 = Z2*T2
	VOR  T0, T0, Y0
	VOR  T1, T1, Y1
	CALL p256MulInternal<>(SB)
	VOR  T0, T0, S1L           // SAVE: S1L
	VOR  T1, T1, S1H           // SAVE: S1H

	// X=X1; Y-  ; MUL; U1=T // U1 = X1*T2
	MOVD   in1+8(FP), P1ptr
	LXVD2X (R0)(P1ptr), X0      // X1L
	LXVD2X (R16)(P1ptr), X1     // X1H
	XXPERMDI X0, X0, $2, X0
	XXPERMDI X1, X1, $2, X1
	CALL   p256MulInternal<>(SB)
	VOR    T0, T0, U1L           // SAVE: U1L
	VOR    T1, T1, U1H           // SAVE: U1H

	// SUB(H<H-T)            // H  = H-U1
	p256SubInternal(HH,HL,HH,HL,T1,T0)

	// if H == 0 or H^P == 0 then ret=1 else ret=0
	// clobbers T1H and T1L
	MOVD       $1, TRUE
	VSPLTISB   $0, ZER
	VOR        HL, HH, T1H
	VCMPEQUDCC ZER, T1H, T1H

	// 26 = CR6 NE
	ISEL       $26, R0, TRUE, RES1
	VXOR       HL, PL, T1L         // SAVE: T1L
	VXOR       HH, PH, T1H         // SAVE: T1H
	VOR        T1L, T1H, T1H
	VCMPEQUDCC ZER, T1H, T1H

	// 26 = CR6 NE
	ISEL $26, R0, TRUE, RES2
	OR   RES2, RES1, RES1
	MOVD RES1, ret+24(FP)

	// X=Z1; Y=Z2; MUL; T-   // Z3 = Z1*Z2
	MOVD   in1+8(FP), P1ptr
	MOVD   in2+16(FP), P2ptr
	LXVD2X (R19)(P1ptr), X0        // Z1L
	LXVD2X (R20)(P1ptr), X1        // Z1H
	XXPERMDI X0, X0, $2, X0
	XXPERMDI X1, X1, $2, X1
	LXVD2X (R19)(P2ptr), Y0        // Z2L
	LXVD2X (R20)(P2ptr), Y1        // Z2H
	XXPERMDI Y0, Y0, $2, Y0
	XXPERMDI Y1, Y1, $2, Y1
	CALL   p256MulInternal<>(SB)

	// X=T ; Y=H ; MUL; Z3:=T// Z3 = Z3*H
	VOR     T0, T0, X0
	VOR     T1, T1, X1
	VOR     HL, HL, Y0
	VOR     HH, HH, Y1
	CALL    p256MulInternal<>(SB)
	MOVD    res+0(FP), P3ptr
	XXPERMDI T1, T1, $2, TT1
	XXPERMDI T0, T0, $2, TT0
	STXVD2X TT0, (R19)(P3ptr)
	STXVD2X TT1, (R20)(P3ptr)

	// X=Y1; Y=S1; MUL; S1=T // S1 = Y1*S1
	MOVD   in1+8(FP), P1ptr
	LXVD2X (R17)(P1ptr), X0
	LXVD2X (R18)(P1ptr), X1
	XXPERMDI X0, X0, $2, X0
	XXPERMDI X1, X1, $2, X1
	VOR    S1L, S1L, Y0
	VOR    S1H, S1H, Y1
	CALL   p256MulInternal<>(SB)
	VOR    T0, T0, S1L
	VOR    T1, T1, S1H

	// X=Y2; Y=R ; MUL; T-   // R  = Y2*R
	MOVD   in2+16(FP), P2ptr
	LXVD2X (R17)(P2ptr), X0
	LXVD2X (R18)(P2ptr), X1
	XXPERMDI X0, X0, $2, X0
	XXPERMDI X1, X1, $2, X1
	VOR    RL, RL, Y0

	// VOR RH, RH, Y1   RH was saved above in D2X format
	LXVD2X (R1)(R17), Y1
	CALL   p256MulInternal<>(SB)

	// SUB(R<T-S1)           // R  = T-S1
	p256SubInternal(RH,RL,T1,T0,S1H,S1L)

	STXVD2X RH, (R1)(R17) // Save RH

	// if R == 0 or R^P == 0 then ret=ret else ret=0
	// clobbers T1H and T1L
	// Redo this using ISEL??
	MOVD       $1, TRUE
	VSPLTISB   $0, ZER
	VOR        RL, RH, T1H
	VCMPEQUDCC ZER, T1H, T1H

	// 24 = CR6 NE
	ISEL       $26, R0, TRUE, RES1
	VXOR       RL, PL, T1L
	VXOR       RH, PH, T1H         // SAVE: T1L
	VOR        T1L, T1H, T1H
	VCMPEQUDCC ZER, T1H, T1H

	// 26 = CR6 NE
	ISEL $26, R0, TRUE, RES2
	OR   RES2, RES1, RES1
	MOVD ret+24(FP), RES2
	AND  RES2, RES1, RES1
	MOVD RES1, ret+24(FP)

	// X=H ; Y=H ; MUL; T-   // T1 = H*H
	VOR  HL, HL, X0
	VOR  HH, HH, X1
	VOR  HL, HL, Y0
	VOR  HH, HH, Y1
	CALL p256MulInternal<>(SB)

	// X-  ; Y=T ; MUL; T2=T // T2 = H*T1
	VOR  T0, T0, Y0
	VOR  T1, T1, Y1
	CALL p256MulInternal<>(SB)
	VOR  T0, T0, T2L
	VOR  T1, T1, T2H

	// X=U1; Y-  ; MUL; U1=T // U1 = U1*T1
	VOR  U1L, U1L, X0
	VOR  U1H, U1H, X1
	CALL p256MulInternal<>(SB)
	VOR  T0, T0, U1L
	VOR  T1, T1, U1H

	// X=R ; Y=R ; MUL; T-   // X3 = R*R
	VOR RL, RL, X0

	// VOR  RH, RH, X1
	VOR RL, RL, Y0

	// RH was saved above using STXVD2X
	LXVD2X (R1)(R17), X1
	VOR    X1, X1, Y1

	// VOR  RH, RH, Y1
	CALL p256MulInternal<>(SB)

	// SUB(T<T-T2)           // X3 = X3-T2
	p256SubInternal(T1,T0,T1,T0,T2H,T2L)

	// ADD(X<U1+U1)          // T1 = 2*U1
	p256AddInternal(X1,X0,U1H,U1L,U1H,U1L)

	// SUB(T<T-X) X3:=T      // X3 = X3-T1 << store-out X3 result reg
	p256SubInternal(T1,T0,T1,T0,X1,X0)
	MOVD    res+0(FP), P3ptr
	XXPERMDI T1, T1, $2, TT1
	XXPERMDI T0, T0, $2, TT0
	STXVD2X TT0, (R0)(P3ptr)
	STXVD2X TT1, (R16)(P3ptr)

	// SUB(Y<U1-T)           // Y3 = U1-X3
	p256SubInternal(Y1,Y0,U1H,U1L,T1,T0)

	// X=R ; Y-  ; MUL; U1=T // Y3 = R*Y3
	VOR RL, RL, X0

	// VOR  RH, RH, X1
	LXVD2X (R1)(R17), X1
	CALL   p256MulInternal<>(SB)
	VOR    T0, T0, U1L
	VOR    T1, T1, U1H

	// X=S1; Y=T2; MUL; T-   // T2 = S1*T2
	VOR  S1L, S1L, X0
	VOR  S1H, S1H, X1
	VOR  T2L, T2L, Y0
	VOR  T2H, T2H, Y1
	CALL p256MulInternal<>(SB)

	// SUB(T<U1-T); Y3:=T    // Y3 = Y3-T2 << store-out Y3 result reg
	p256SubInternal(T1,T0,U1H,U1L,T1,T0)
	MOVD    res+0(FP), P3ptr
	XXPERMDI T1, T1, $2, TT1
	XXPERMDI T0, T0, $2, TT0
	STXVD2X TT0, (R17)(P3ptr)
	STXVD2X TT1, (R18)(P3ptr)

	RET
