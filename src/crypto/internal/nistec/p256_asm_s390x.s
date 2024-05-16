// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

#include "textflag.h"
#include "go_asm.h"

DATA p256ordK0<>+0x00(SB)/4, $0xee00bc4f
DATA p256ord<>+0x00(SB)/8, $0xffffffff00000000
DATA p256ord<>+0x08(SB)/8, $0xffffffffffffffff
DATA p256ord<>+0x10(SB)/8, $0xbce6faada7179e84
DATA p256ord<>+0x18(SB)/8, $0xf3b9cac2fc632551
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
DATA p256<>+0x50(SB)/8, $0x0706050403020100 // LE2BE permute mask
DATA p256<>+0x58(SB)/8, $0x0f0e0d0c0b0a0908 // LE2BE permute mask
DATA p256mul<>+0x00(SB)/8, $0xffffffff00000001 // P256
DATA p256mul<>+0x08(SB)/8, $0x0000000000000000 // P256
DATA p256mul<>+0x10(SB)/8, $0x00000000ffffffff // P256
DATA p256mul<>+0x18(SB)/8, $0xffffffffffffffff // P256
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
DATA p256mul<>+0x80(SB)/8, $0x00000000fffffffe // (1*2^256)%P256
DATA p256mul<>+0x88(SB)/8, $0xffffffffffffffff // (1*2^256)%P256
DATA p256mul<>+0x90(SB)/8, $0xffffffff00000000 // (1*2^256)%P256
DATA p256mul<>+0x98(SB)/8, $0x0000000000000001 // (1*2^256)%P256
GLOBL p256ordK0<>(SB), 8, $4
GLOBL p256ord<>(SB), 8, $32
GLOBL p256<>(SB), 8, $96
GLOBL p256mul<>(SB), 8, $160

// func p256OrdLittleToBig(res *[32]byte, in *p256OrdElement)
TEXT ·p256OrdLittleToBig(SB), NOSPLIT, $0
	JMP ·p256BigToLittle(SB)

// func p256OrdBigToLittle(res *p256OrdElement, in *[32]byte)
TEXT ·p256OrdBigToLittle(SB), NOSPLIT, $0
	JMP ·p256BigToLittle(SB)

// ---------------------------------------
// func p256LittleToBig(res *[32]byte, in *p256Element)
TEXT ·p256LittleToBig(SB), NOSPLIT, $0
	JMP ·p256BigToLittle(SB)

// func p256BigToLittle(res *p256Element, in *[32]byte)
#define res_ptr   R1
#define in_ptr   R2
#define T1L   V2
#define T1H   V3

TEXT ·p256BigToLittle(SB), NOSPLIT, $0
	MOVD res+0(FP), res_ptr
	MOVD in+8(FP), in_ptr

	VL 0(in_ptr), T1H
	VL 16(in_ptr), T1L

	VPDI $0x4, T1L, T1L, T1L
	VPDI $0x4, T1H, T1H, T1H

	VST T1L, 0(res_ptr)
	VST T1H, 16(res_ptr)
	RET

#undef res_ptr
#undef in_ptr
#undef T1L
#undef T1H

// ---------------------------------------
// iff cond == 1  val <- -val
// func p256NegCond(val *p256Element, cond int)
#define P1ptr   R1
#define CPOOL   R4

#define Y1L   V0
#define Y1H   V1
#define T1L   V2
#define T1H   V3

#define PL    V30
#define PH    V31

#define ZER   V4
#define SEL1  V5
#define CAR1  V6
TEXT ·p256NegCond(SB), NOSPLIT, $0
	MOVD val+0(FP), P1ptr

	MOVD $p256mul<>+0x00(SB), CPOOL
	VL   16(CPOOL), PL
	VL   0(CPOOL), PH

	VL   16(P1ptr), Y1H
	VPDI $0x4, Y1H, Y1H, Y1H
	VL   0(P1ptr), Y1L
	VPDI $0x4, Y1L, Y1L, Y1L

	VLREPG cond+8(FP), SEL1
	VZERO  ZER
	VCEQG  SEL1, ZER, SEL1

	VSCBIQ Y1L, PL, CAR1
	VSQ    Y1L, PL, T1L
	VSBIQ  PH, Y1H, CAR1, T1H

	VSEL Y1L, T1L, SEL1, Y1L
	VSEL Y1H, T1H, SEL1, Y1H

	VPDI $0x4, Y1H, Y1H, Y1H
	VST  Y1H, 16(P1ptr)
	VPDI $0x4, Y1L, Y1L, Y1L
	VST  Y1L, 0(P1ptr)
	RET

#undef P1ptr
#undef CPOOL
#undef Y1L
#undef Y1H
#undef T1L
#undef T1H
#undef PL
#undef PH
#undef ZER
#undef SEL1
#undef CAR1

// ---------------------------------------
// if cond == 0 res <- b; else res <- a
// func p256MovCond(res, a, b *P256Point, cond int)
#define P3ptr   R1
#define P1ptr   R2
#define P2ptr   R3

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

#define ZER   V18
#define SEL1  V19
TEXT ·p256MovCond(SB), NOSPLIT, $0
	MOVD   res+0(FP), P3ptr
	MOVD   a+8(FP), P1ptr
	MOVD   b+16(FP), P2ptr
	VLREPG cond+24(FP), SEL1
	VZERO  ZER
	VCEQG  SEL1, ZER, SEL1

	VL 0(P1ptr), X1H
	VL 16(P1ptr), X1L
	VL 32(P1ptr), Y1H
	VL 48(P1ptr), Y1L
	VL 64(P1ptr), Z1H
	VL 80(P1ptr), Z1L

	VL 0(P2ptr), X2H
	VL 16(P2ptr), X2L
	VL 32(P2ptr), Y2H
	VL 48(P2ptr), Y2L
	VL 64(P2ptr), Z2H
	VL 80(P2ptr), Z2L

	VSEL X2L, X1L, SEL1, X1L
	VSEL X2H, X1H, SEL1, X1H
	VSEL Y2L, Y1L, SEL1, Y1L
	VSEL Y2H, Y1H, SEL1, Y1H
	VSEL Z2L, Z1L, SEL1, Z1L
	VSEL Z2H, Z1H, SEL1, Z1H

	VST X1H, 0(P3ptr)
	VST X1L, 16(P3ptr)
	VST Y1H, 32(P3ptr)
	VST Y1L, 48(P3ptr)
	VST Z1H, 64(P3ptr)
	VST Z1L, 80(P3ptr)

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
#undef ZER
#undef SEL1

// ---------------------------------------
// Constant time table access
// Indexed from 1 to 15, with -1 offset
// (index 0 is implicitly point at infinity)
// func p256Select(res *P256Point, table *p256Table, idx int)
#define P3ptr   R1
#define P1ptr   R2
#define COUNT   R4

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
TEXT ·p256Select(SB), NOSPLIT, $0
	MOVD   res+0(FP), P3ptr
	MOVD   table+8(FP), P1ptr
	VLREPB idx+(16+7)(FP), IDX
	VREPIB $1, ONE
	VREPIB $1, SEL2
	MOVD   $1, COUNT

	VZERO X1H
	VZERO X1L
	VZERO Y1H
	VZERO Y1L
	VZERO Z1H
	VZERO Z1L

loop_select:
	VL 0(P1ptr), X2H
	VL 16(P1ptr), X2L
	VL 32(P1ptr), Y2H
	VL 48(P1ptr), Y2L
	VL 64(P1ptr), Z2H
	VL 80(P1ptr), Z2L

	VCEQG SEL2, IDX, SEL1

	VSEL X2L, X1L, SEL1, X1L
	VSEL X2H, X1H, SEL1, X1H
	VSEL Y2L, Y1L, SEL1, Y1L
	VSEL Y2H, Y1H, SEL1, Y1H
	VSEL Z2L, Z1L, SEL1, Z1L
	VSEL Z2H, Z1H, SEL1, Z1H

	VAB  SEL2, ONE, SEL2
	ADDW $1, COUNT
	ADD  $96, P1ptr
	CMPW COUNT, $17
	BLT  loop_select

	VST X1H, 0(P3ptr)
	VST X1L, 16(P3ptr)
	VST Y1H, 32(P3ptr)
	VST Y1L, 48(P3ptr)
	VST Z1H, 64(P3ptr)
	VST Z1L, 80(P3ptr)
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

// ---------------------------------------

//  func p256FromMont(res, in *p256Element)
#define res_ptr R1
#define x_ptr   R2
#define CPOOL   R4

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

TEXT ·p256FromMont(SB), NOSPLIT, $0
	MOVD res+0(FP), res_ptr
	MOVD in+8(FP), x_ptr

	VZERO T2
	VZERO ZER
	MOVD  $p256<>+0x00(SB), CPOOL
	VL    16(CPOOL), PL
	VL    0(CPOOL), PH
	VL    48(CPOOL), SEL2
	VL    64(CPOOL), SEL1

	VL   (0*16)(x_ptr), T0
	VPDI $0x4, T0, T0, T0
	VL   (1*16)(x_ptr), T1
	VPDI $0x4, T1, T1, T1

	// First round
	VPERM T1, T0, SEL1, RED2    // d1 d0 d1 d0
	VPERM ZER, RED2, SEL2, RED1 // 0  d1 d0  0
	VSQ   RED1, RED2, RED2      // Guaranteed not to underflow

	VSLDB $8, T1, T0, T0
	VSLDB $8, T2, T1, T1

	VACCQ  T0, RED1, CAR1
	VAQ    T0, RED1, T0
	VACCCQ T1, RED2, CAR1, CAR2
	VACQ   T1, RED2, CAR1, T1
	VAQ    T2, CAR2, T2

	// Second round
	VPERM T1, T0, SEL1, RED2    // d1 d0 d1 d0
	VPERM ZER, RED2, SEL2, RED1 // 0  d1 d0  0
	VSQ   RED1, RED2, RED2      // Guaranteed not to underflow

	VSLDB $8, T1, T0, T0
	VSLDB $8, T2, T1, T1

	VACCQ  T0, RED1, CAR1
	VAQ    T0, RED1, T0
	VACCCQ T1, RED2, CAR1, CAR2
	VACQ   T1, RED2, CAR1, T1
	VAQ    T2, CAR2, T2

	// Third round
	VPERM T1, T0, SEL1, RED2    // d1 d0 d1 d0
	VPERM ZER, RED2, SEL2, RED1 // 0  d1 d0  0
	VSQ   RED1, RED2, RED2      // Guaranteed not to underflow

	VSLDB $8, T1, T0, T0
	VSLDB $8, T2, T1, T1

	VACCQ  T0, RED1, CAR1
	VAQ    T0, RED1, T0
	VACCCQ T1, RED2, CAR1, CAR2
	VACQ   T1, RED2, CAR1, T1
	VAQ    T2, CAR2, T2

	// Last round
	VPERM T1, T0, SEL1, RED2    // d1 d0 d1 d0
	VPERM ZER, RED2, SEL2, RED1 // 0  d1 d0  0
	VSQ   RED1, RED2, RED2      // Guaranteed not to underflow

	VSLDB $8, T1, T0, T0
	VSLDB $8, T2, T1, T1

	VACCQ  T0, RED1, CAR1
	VAQ    T0, RED1, T0
	VACCCQ T1, RED2, CAR1, CAR2
	VACQ   T1, RED2, CAR1, T1
	VAQ    T2, CAR2, T2

	// ---------------------------------------------------

	VSCBIQ  PL, T0, CAR1
	VSQ     PL, T0, TT0
	VSBCBIQ T1, PH, CAR1, CAR2
	VSBIQ   T1, PH, CAR1, TT1
	VSBIQ   T2, ZER, CAR2, T2

	// what output to use, TT1||TT0 or T1||T0?
	VSEL T0, TT0, T2, T0
	VSEL T1, TT1, T2, T1

	VPDI $0x4, T0, T0, TT0
	VST  TT0, (0*16)(res_ptr)
	VPDI $0x4, T1, T1, TT1
	VST  TT1, (1*16)(res_ptr)
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

// Constant time table access
// Indexed from 1 to 15, with -1 offset
// (index 0 is implicitly point at infinity)
// func p256SelectBase(point *p256Point, table []p256Point, idx int)
// new : func p256SelectAffine(res *p256AffinePoint, table *p256AffineTable, idx int)

#define P3ptr   R1
#define P1ptr   R2
#define COUNT   R4
#define CPOOL   R5

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
#define LE2BE  V12

#define ONE   V18
#define IDX   V19
#define SEL1  V20
#define SEL2  V21

TEXT ·p256SelectAffine(SB), NOSPLIT, $0
	MOVD   res+0(FP), P3ptr
	MOVD   table+8(FP), P1ptr
	MOVD   $p256<>+0x00(SB), CPOOL
	VLREPB idx+(16+7)(FP), IDX
	VREPIB $1, ONE
	VREPIB $1, SEL2
	MOVD   $1, COUNT
	VL     80(CPOOL), LE2BE

	VZERO X1H
	VZERO X1L
	VZERO Y1H
	VZERO Y1L

loop_select:
	VL 0(P1ptr), X2H
	VL 16(P1ptr), X2L
	VL 32(P1ptr), Y2H
	VL 48(P1ptr), Y2L

	VCEQG SEL2, IDX, SEL1

	VSEL X2L, X1L, SEL1, X1L
	VSEL X2H, X1H, SEL1, X1H
	VSEL Y2L, Y1L, SEL1, Y1L
	VSEL Y2H, Y1H, SEL1, Y1H

	VAB  SEL2, ONE, SEL2
	ADDW $1, COUNT
	ADD  $64, P1ptr
	CMPW COUNT, $65
	BLT  loop_select
	VST  X1H, 0(P3ptr)
	VST  X1L, 16(P3ptr)
	VST  Y1H, 32(P3ptr)
	VST  Y1L, 48(P3ptr)

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
#undef CPOOL

// ---------------------------------------

// func p256OrdMul(res, in1, in2 *p256OrdElement)
#define res_ptr R1
#define x_ptr R2
#define y_ptr R3
#define X0    V0
#define X1    V1
#define Y0    V2
#define Y1    V3
#define M0    V4
#define M1    V5
#define T0    V6
#define T1    V7
#define T2    V8
#define YDIG  V9

#define ADD1  V16
#define ADD1H V17
#define ADD2  V18
#define ADD2H V19
#define RED1  V20
#define RED1H V21
#define RED2  V22
#define RED2H V23
#define CAR1  V24
#define CAR1M V25

#define MK0   V30
#define K0    V31
TEXT ·p256OrdMul<>(SB), NOSPLIT, $0
	MOVD res+0(FP), res_ptr
	MOVD in1+8(FP), x_ptr
	MOVD in2+16(FP), y_ptr

	VZERO T2
	MOVD  $p256ordK0<>+0x00(SB), R4

	// VLEF    $3, 0(R4), K0
	WORD $0xE7F40000
	BYTE $0x38
	BYTE $0x03
	MOVD $p256ord<>+0x00(SB), R4
	VL   16(R4), M0
	VL   0(R4), M1

	VL   (0*16)(x_ptr), X0
	VPDI $0x4, X0, X0, X0
	VL   (1*16)(x_ptr), X1
	VPDI $0x4, X1, X1, X1
	VL   (0*16)(y_ptr), Y0
	VPDI $0x4, Y0, Y0, Y0
	VL   (1*16)(y_ptr), Y1
	VPDI $0x4, Y1, Y1, Y1

	// ---------------------------------------------------------------------------/
	VREPF $3, Y0, YDIG
	VMLF  X0, YDIG, ADD1
	VMLF  ADD1, K0, MK0
	VREPF $3, MK0, MK0

	VMLF  X1, YDIG, ADD2
	VMLHF X0, YDIG, ADD1H
	VMLHF X1, YDIG, ADD2H

	VMALF  M0, MK0, ADD1, RED1
	VMALHF M0, MK0, ADD1, RED1H
	VMALF  M1, MK0, ADD2, RED2
	VMALHF M1, MK0, ADD2, RED2H

	VSLDB $12, RED2, RED1, RED1
	VSLDB $12, T2, RED2, RED2

	VACCQ RED1, ADD1H, CAR1
	VAQ   RED1, ADD1H, T0
	VACCQ RED1H, T0, CAR1M
	VAQ   RED1H, T0, T0

	// << ready for next MK0

	VACQ   RED2, ADD2H, CAR1, T1
	VACCCQ RED2, ADD2H, CAR1, CAR1
	VACCCQ RED2H, T1, CAR1M, T2
	VACQ   RED2H, T1, CAR1M, T1
	VAQ    CAR1, T2, T2

	// ---------------------------------------------------
/* *
 * ---+--------+--------+
 *  T2|   T1   |   T0   |
 * ---+--------+--------+
 *           *(add)*
 *    +--------+--------+
 *    |   X1   |   X0   |
 *    +--------+--------+
 *           *(mul)*
 *    +--------+--------+
 *    |  YDIG  |  YDIG  |
 *    +--------+--------+
 *           *(add)*
 *    +--------+--------+
 *    |   M1   |   M0   |
 *    +--------+--------+
 *           *(mul)*
 *    +--------+--------+
 *    |   MK0  |   MK0  |
 *    +--------+--------+
 *
 *   ---------------------
 *
 *    +--------+--------+
 *    |  ADD2  |  ADD1  |
 *    +--------+--------+
 *  +--------+--------+
 *  | ADD2H  | ADD1H  |
 *  +--------+--------+
 *    +--------+--------+
 *    |  RED2  |  RED1  |
 *    +--------+--------+
 *  +--------+--------+
 *  | RED2H  | RED1H  |
 *  +--------+--------+
 */
	VREPF $2, Y0, YDIG
	VMALF X0, YDIG, T0, ADD1
	VMLF  ADD1, K0, MK0
	VREPF $3, MK0, MK0

	VMALF  X1, YDIG, T1, ADD2
	VMALHF X0, YDIG, T0, ADD1H
	VMALHF X1, YDIG, T1, ADD2H

	VMALF  M0, MK0, ADD1, RED1
	VMALHF M0, MK0, ADD1, RED1H
	VMALF  M1, MK0, ADD2, RED2
	VMALHF M1, MK0, ADD2, RED2H

	VSLDB $12, RED2, RED1, RED1
	VSLDB $12, T2, RED2, RED2

	VACCQ RED1, ADD1H, CAR1
	VAQ   RED1, ADD1H, T0
	VACCQ RED1H, T0, CAR1M
	VAQ   RED1H, T0, T0

	// << ready for next MK0

	VACQ   RED2, ADD2H, CAR1, T1
	VACCCQ RED2, ADD2H, CAR1, CAR1
	VACCCQ RED2H, T1, CAR1M, T2
	VACQ   RED2H, T1, CAR1M, T1
	VAQ    CAR1, T2, T2

	// ---------------------------------------------------
	VREPF $1, Y0, YDIG
	VMALF X0, YDIG, T0, ADD1
	VMLF  ADD1, K0, MK0
	VREPF $3, MK0, MK0

	VMALF  X1, YDIG, T1, ADD2
	VMALHF X0, YDIG, T0, ADD1H
	VMALHF X1, YDIG, T1, ADD2H

	VMALF  M0, MK0, ADD1, RED1
	VMALHF M0, MK0, ADD1, RED1H
	VMALF  M1, MK0, ADD2, RED2
	VMALHF M1, MK0, ADD2, RED2H

	VSLDB $12, RED2, RED1, RED1
	VSLDB $12, T2, RED2, RED2

	VACCQ RED1, ADD1H, CAR1
	VAQ   RED1, ADD1H, T0
	VACCQ RED1H, T0, CAR1M
	VAQ   RED1H, T0, T0

	// << ready for next MK0

	VACQ   RED2, ADD2H, CAR1, T1
	VACCCQ RED2, ADD2H, CAR1, CAR1
	VACCCQ RED2H, T1, CAR1M, T2
	VACQ   RED2H, T1, CAR1M, T1
	VAQ    CAR1, T2, T2

	// ---------------------------------------------------
	VREPF $0, Y0, YDIG
	VMALF X0, YDIG, T0, ADD1
	VMLF  ADD1, K0, MK0
	VREPF $3, MK0, MK0

	VMALF  X1, YDIG, T1, ADD2
	VMALHF X0, YDIG, T0, ADD1H
	VMALHF X1, YDIG, T1, ADD2H

	VMALF  M0, MK0, ADD1, RED1
	VMALHF M0, MK0, ADD1, RED1H
	VMALF  M1, MK0, ADD2, RED2
	VMALHF M1, MK0, ADD2, RED2H

	VSLDB $12, RED2, RED1, RED1
	VSLDB $12, T2, RED2, RED2

	VACCQ RED1, ADD1H, CAR1
	VAQ   RED1, ADD1H, T0
	VACCQ RED1H, T0, CAR1M
	VAQ   RED1H, T0, T0

	// << ready for next MK0

	VACQ   RED2, ADD2H, CAR1, T1
	VACCCQ RED2, ADD2H, CAR1, CAR1
	VACCCQ RED2H, T1, CAR1M, T2
	VACQ   RED2H, T1, CAR1M, T1
	VAQ    CAR1, T2, T2

	// ---------------------------------------------------
	VREPF $3, Y1, YDIG
	VMALF X0, YDIG, T0, ADD1
	VMLF  ADD1, K0, MK0
	VREPF $3, MK0, MK0

	VMALF  X1, YDIG, T1, ADD2
	VMALHF X0, YDIG, T0, ADD1H
	VMALHF X1, YDIG, T1, ADD2H

	VMALF  M0, MK0, ADD1, RED1
	VMALHF M0, MK0, ADD1, RED1H
	VMALF  M1, MK0, ADD2, RED2
	VMALHF M1, MK0, ADD2, RED2H

	VSLDB $12, RED2, RED1, RED1
	VSLDB $12, T2, RED2, RED2

	VACCQ RED1, ADD1H, CAR1
	VAQ   RED1, ADD1H, T0
	VACCQ RED1H, T0, CAR1M
	VAQ   RED1H, T0, T0

	// << ready for next MK0

	VACQ   RED2, ADD2H, CAR1, T1
	VACCCQ RED2, ADD2H, CAR1, CAR1
	VACCCQ RED2H, T1, CAR1M, T2
	VACQ   RED2H, T1, CAR1M, T1
	VAQ    CAR1, T2, T2

	// ---------------------------------------------------
	VREPF $2, Y1, YDIG
	VMALF X0, YDIG, T0, ADD1
	VMLF  ADD1, K0, MK0
	VREPF $3, MK0, MK0

	VMALF  X1, YDIG, T1, ADD2
	VMALHF X0, YDIG, T0, ADD1H
	VMALHF X1, YDIG, T1, ADD2H

	VMALF  M0, MK0, ADD1, RED1
	VMALHF M0, MK0, ADD1, RED1H
	VMALF  M1, MK0, ADD2, RED2
	VMALHF M1, MK0, ADD2, RED2H

	VSLDB $12, RED2, RED1, RED1
	VSLDB $12, T2, RED2, RED2

	VACCQ RED1, ADD1H, CAR1
	VAQ   RED1, ADD1H, T0
	VACCQ RED1H, T0, CAR1M
	VAQ   RED1H, T0, T0

	// << ready for next MK0

	VACQ   RED2, ADD2H, CAR1, T1
	VACCCQ RED2, ADD2H, CAR1, CAR1
	VACCCQ RED2H, T1, CAR1M, T2
	VACQ   RED2H, T1, CAR1M, T1
	VAQ    CAR1, T2, T2

	// ---------------------------------------------------
	VREPF $1, Y1, YDIG
	VMALF X0, YDIG, T0, ADD1
	VMLF  ADD1, K0, MK0
	VREPF $3, MK0, MK0

	VMALF  X1, YDIG, T1, ADD2
	VMALHF X0, YDIG, T0, ADD1H
	VMALHF X1, YDIG, T1, ADD2H

	VMALF  M0, MK0, ADD1, RED1
	VMALHF M0, MK0, ADD1, RED1H
	VMALF  M1, MK0, ADD2, RED2
	VMALHF M1, MK0, ADD2, RED2H

	VSLDB $12, RED2, RED1, RED1
	VSLDB $12, T2, RED2, RED2

	VACCQ RED1, ADD1H, CAR1
	VAQ   RED1, ADD1H, T0
	VACCQ RED1H, T0, CAR1M
	VAQ   RED1H, T0, T0

	// << ready for next MK0

	VACQ   RED2, ADD2H, CAR1, T1
	VACCCQ RED2, ADD2H, CAR1, CAR1
	VACCCQ RED2H, T1, CAR1M, T2
	VACQ   RED2H, T1, CAR1M, T1
	VAQ    CAR1, T2, T2

	// ---------------------------------------------------
	VREPF $0, Y1, YDIG
	VMALF X0, YDIG, T0, ADD1
	VMLF  ADD1, K0, MK0
	VREPF $3, MK0, MK0

	VMALF  X1, YDIG, T1, ADD2
	VMALHF X0, YDIG, T0, ADD1H
	VMALHF X1, YDIG, T1, ADD2H

	VMALF  M0, MK0, ADD1, RED1
	VMALHF M0, MK0, ADD1, RED1H
	VMALF  M1, MK0, ADD2, RED2
	VMALHF M1, MK0, ADD2, RED2H

	VSLDB $12, RED2, RED1, RED1
	VSLDB $12, T2, RED2, RED2

	VACCQ RED1, ADD1H, CAR1
	VAQ   RED1, ADD1H, T0
	VACCQ RED1H, T0, CAR1M
	VAQ   RED1H, T0, T0

	// << ready for next MK0

	VACQ   RED2, ADD2H, CAR1, T1
	VACCCQ RED2, ADD2H, CAR1, CAR1
	VACCCQ RED2H, T1, CAR1M, T2
	VACQ   RED2H, T1, CAR1M, T1
	VAQ    CAR1, T2, T2

	// ---------------------------------------------------

	VZERO   RED1
	VSCBIQ  M0, T0, CAR1
	VSQ     M0, T0, ADD1
	VSBCBIQ T1, M1, CAR1, CAR1M
	VSBIQ   T1, M1, CAR1, ADD2
	VSBIQ   T2, RED1, CAR1M, T2

	// what output to use, ADD2||ADD1 or T1||T0?
	VSEL T0, ADD1, T2, T0
	VSEL T1, ADD2, T2, T1

	VPDI $0x4, T0, T0, T0
	VST  T0, (0*16)(res_ptr)
	VPDI $0x4, T1, T1, T1
	VST  T1, (1*16)(res_ptr)
	RET

#undef res_ptr
#undef x_ptr
#undef y_ptr
#undef X0
#undef X1
#undef Y0
#undef Y1
#undef M0
#undef M1
#undef T0
#undef T1
#undef T2
#undef YDIG

#undef ADD1
#undef ADD1H
#undef ADD2
#undef ADD2H
#undef RED1
#undef RED1H
#undef RED2
#undef RED2H
#undef CAR1
#undef CAR1M

#undef MK0
#undef K0

// ---------------------------------------
// p256MulInternal
// V0-V3,V30,V31 - Not Modified
// V4-V15 - Volatile

#define CPOOL   R4

// Parameters
#define X0    V0 // Not modified
#define X1    V1 // Not modified
#define Y0    V2 // Not modified
#define Y1    V3 // Not modified
#define T0    V4
#define T1    V5
#define P0    V30 // Not modified
#define P1    V31 // Not modified

// Temporaries
#define YDIG  V6 // Overloaded with CAR2, ZER
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
#define ZER   V6 // Overloaded with YDIG, CAR2
#define CAR1  V6 // Overloaded with YDIG, ZER
#define CAR2  V11 // Overloaded with RED1
// Constant Selects
#define SEL1  V13 // Overloaded with RED3
#define SEL2  V9 // Overloaded with ADD3,SEL5
#define SEL3  V10 // Overloaded with ADD4,SEL6
#define SEL4  V6 // Overloaded with YDIG,CAR2,ZER
#define SEL5  V9 // Overloaded with ADD3,SEL2
#define SEL6  V10 // Overloaded with ADD4,SEL3

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
TEXT p256MulInternal<>(SB), NOSPLIT, $0-0
	VL 32(CPOOL), SEL1
	VL 48(CPOOL), SEL2
	VL 64(CPOOL), SEL3
	VL 80(CPOOL), SEL4

	// ---------------------------------------------------

	VREPF $3, Y0, YDIG
	VMLHF X0, YDIG, ADD1H
	VMLHF X1, YDIG, ADD2H
	VMLF  X0, YDIG, ADD1
	VMLF  X1, YDIG, ADD2

	VREPF  $2, Y0, YDIG
	VMALF  X0, YDIG, ADD1H, ADD3
	VMALF  X1, YDIG, ADD2H, ADD4
	VMALHF X0, YDIG, ADD1H, ADD3H // ADD1H Free
	VMALHF X1, YDIG, ADD2H, ADD4H // ADD2H Free

	VZERO ZER
	VL    32(CPOOL), SEL1
	VPERM ZER, ADD1, SEL1, RED3 // [d0 0 0 d0]

	VSLDB $12, ADD2, ADD1, T0 // ADD1 Free
	VSLDB $12, ZER, ADD2, T1  // ADD2 Free

	VACCQ  T0, ADD3, CAR1
	VAQ    T0, ADD3, T0       // ADD3 Free
	VACCCQ T1, ADD4, CAR1, T2
	VACQ   T1, ADD4, CAR1, T1 // ADD4 Free

	VL    48(CPOOL), SEL2
	VL    64(CPOOL), SEL3
	VL    80(CPOOL), SEL4
	VPERM RED3, T0, SEL2, RED1 // [d0  0 d1 d0]
	VPERM RED3, T0, SEL3, RED2 // [ 0 d1 d0 d1]
	VPERM RED3, T0, SEL4, RED3 // [ 0  0 d1 d0]
	VSQ   RED3, RED2, RED2     // Guaranteed not to underflow

	VSLDB $12, T1, T0, T0
	VSLDB $12, T2, T1, T1

	VACCQ  T0, ADD3H, CAR1
	VAQ    T0, ADD3H, T0
	VACCCQ T1, ADD4H, CAR1, T2
	VACQ   T1, ADD4H, CAR1, T1

	// ---------------------------------------------------

	VREPF  $1, Y0, YDIG
	VMALHF X0, YDIG, T0, ADD1H
	VMALHF X1, YDIG, T1, ADD2H
	VMALF  X0, YDIG, T0, ADD1  // T0 Free->ADD1
	VMALF  X1, YDIG, T1, ADD2  // T1 Free->ADD2

	VREPF  $0, Y0, YDIG
	VMALF  X0, YDIG, ADD1H, ADD3
	VMALF  X1, YDIG, ADD2H, ADD4
	VMALHF X0, YDIG, ADD1H, ADD3H // ADD1H Free->ADD3H
	VMALHF X1, YDIG, ADD2H, ADD4H // ADD2H Free->ADD4H , YDIG Free->ZER

	VZERO ZER
	VL    32(CPOOL), SEL1
	VPERM ZER, ADD1, SEL1, RED3 // [d0 0 0 d0]

	VSLDB $12, ADD2, ADD1, T0 // ADD1 Free->T0
	VSLDB $12, T2, ADD2, T1   // ADD2 Free->T1, T2 Free

	VACCQ  T0, RED1, CAR1
	VAQ    T0, RED1, T0
	VACCCQ T1, RED2, CAR1, T2
	VACQ   T1, RED2, CAR1, T1

	VACCQ  T0, ADD3, CAR1
	VAQ    T0, ADD3, T0
	VACCCQ T1, ADD4, CAR1, CAR2
	VACQ   T1, ADD4, CAR1, T1
	VAQ    T2, CAR2, T2

	VL    48(CPOOL), SEL2
	VL    64(CPOOL), SEL3
	VL    80(CPOOL), SEL4
	VPERM RED3, T0, SEL2, RED1 // [d0  0 d1 d0]
	VPERM RED3, T0, SEL3, RED2 // [ 0 d1 d0 d1]
	VPERM RED3, T0, SEL4, RED3 // [ 0  0 d1 d0]
	VSQ   RED3, RED2, RED2     // Guaranteed not to underflow

	VSLDB $12, T1, T0, T0
	VSLDB $12, T2, T1, T1

	VACCQ  T0, ADD3H, CAR1
	VAQ    T0, ADD3H, T0
	VACCCQ T1, ADD4H, CAR1, T2
	VACQ   T1, ADD4H, CAR1, T1

	// ---------------------------------------------------

	VREPF  $3, Y1, YDIG
	VMALHF X0, YDIG, T0, ADD1H
	VMALHF X1, YDIG, T1, ADD2H
	VMALF  X0, YDIG, T0, ADD1
	VMALF  X1, YDIG, T1, ADD2

	VREPF  $2, Y1, YDIG
	VMALF  X0, YDIG, ADD1H, ADD3
	VMALF  X1, YDIG, ADD2H, ADD4
	VMALHF X0, YDIG, ADD1H, ADD3H // ADD1H Free
	VMALHF X1, YDIG, ADD2H, ADD4H // ADD2H Free

	VZERO ZER
	VL    32(CPOOL), SEL1
	VPERM ZER, ADD1, SEL1, RED3 // [d0 0 0 d0]

	VSLDB $12, ADD2, ADD1, T0 // ADD1 Free
	VSLDB $12, T2, ADD2, T1   // ADD2 Free

	VACCQ  T0, RED1, CAR1
	VAQ    T0, RED1, T0
	VACCCQ T1, RED2, CAR1, T2
	VACQ   T1, RED2, CAR1, T1

	VACCQ  T0, ADD3, CAR1
	VAQ    T0, ADD3, T0
	VACCCQ T1, ADD4, CAR1, CAR2
	VACQ   T1, ADD4, CAR1, T1
	VAQ    T2, CAR2, T2

	VL    48(CPOOL), SEL2
	VL    64(CPOOL), SEL3
	VL    80(CPOOL), SEL4
	VPERM RED3, T0, SEL2, RED1 // [d0  0 d1 d0]
	VPERM RED3, T0, SEL3, RED2 // [ 0 d1 d0 d1]
	VPERM RED3, T0, SEL4, RED3 // [ 0  0 d1 d0]
	VSQ   RED3, RED2, RED2     // Guaranteed not to underflow

	VSLDB $12, T1, T0, T0
	VSLDB $12, T2, T1, T1

	VACCQ  T0, ADD3H, CAR1
	VAQ    T0, ADD3H, T0
	VACCCQ T1, ADD4H, CAR1, T2
	VACQ   T1, ADD4H, CAR1, T1

	// ---------------------------------------------------

	VREPF  $1, Y1, YDIG
	VMALHF X0, YDIG, T0, ADD1H
	VMALHF X1, YDIG, T1, ADD2H
	VMALF  X0, YDIG, T0, ADD1
	VMALF  X1, YDIG, T1, ADD2

	VREPF  $0, Y1, YDIG
	VMALF  X0, YDIG, ADD1H, ADD3
	VMALF  X1, YDIG, ADD2H, ADD4
	VMALHF X0, YDIG, ADD1H, ADD3H
	VMALHF X1, YDIG, ADD2H, ADD4H

	VZERO ZER
	VL    32(CPOOL), SEL1
	VPERM ZER, ADD1, SEL1, RED3 // [d0 0 0 d0]

	VSLDB $12, ADD2, ADD1, T0
	VSLDB $12, T2, ADD2, T1

	VACCQ  T0, RED1, CAR1
	VAQ    T0, RED1, T0
	VACCCQ T1, RED2, CAR1, T2
	VACQ   T1, RED2, CAR1, T1

	VACCQ  T0, ADD3, CAR1
	VAQ    T0, ADD3, T0
	VACCCQ T1, ADD4, CAR1, CAR2
	VACQ   T1, ADD4, CAR1, T1
	VAQ    T2, CAR2, T2

	VL    96(CPOOL), SEL5
	VL    112(CPOOL), SEL6
	VPERM T0, RED3, SEL5, RED2 // [d1 d0 d1 d0]
	VPERM T0, RED3, SEL6, RED1 // [ 0 d1 d0  0]
	VSQ   RED1, RED2, RED2     // Guaranteed not to underflow

	VSLDB $12, T1, T0, T0
	VSLDB $12, T2, T1, T1

	VACCQ  T0, ADD3H, CAR1
	VAQ    T0, ADD3H, T0
	VACCCQ T1, ADD4H, CAR1, T2
	VACQ   T1, ADD4H, CAR1, T1

	VACCQ  T0, RED1, CAR1
	VAQ    T0, RED1, T0
	VACCCQ T1, RED2, CAR1, CAR2
	VACQ   T1, RED2, CAR1, T1
	VAQ    T2, CAR2, T2

	// ---------------------------------------------------

	VZERO   RED3
	VSCBIQ  P0, T0, CAR1
	VSQ     P0, T0, ADD1H
	VSBCBIQ T1, P1, CAR1, CAR2
	VSBIQ   T1, P1, CAR1, ADD2H
	VSBIQ   T2, RED3, CAR2, T2

	// what output to use, ADD2H||ADD1H or T1||T0?
	VSEL T0, ADD1H, T2, T0
	VSEL T1, ADD2H, T2, T1
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

// ---------------------------------------

// Parameters
#define X0    V0
#define X1    V1
#define Y0    V2
#define Y1    V3

TEXT p256SqrInternal<>(SB), NOFRAME|NOSPLIT, $0
	VLR X0, Y0
	VLR X1, Y1
	BR  p256MulInternal<>(SB)

#undef X0
#undef X1
#undef Y0
#undef Y1

#define p256SubInternal(T1, T0, X1, X0, Y1, Y0) \
	VZERO   ZER                \
	VSCBIQ  Y0, X0, CAR1       \
	VSQ     Y0, X0, T0         \
	VSBCBIQ X1, Y1, CAR1, SEL1 \
	VSBIQ   X1, Y1, CAR1, T1   \
	VSQ     SEL1, ZER, SEL1    \
	                           \
	VACCQ   T0, PL, CAR1       \
	VAQ     T0, PL, TT0        \
	VACQ    T1, PH, CAR1, TT1  \
	                           \
	VSEL    T0, TT0, SEL1, T0  \
	VSEL    T1, TT1, SEL1, T1  \

#define p256AddInternal(T1, T0, X1, X0, Y1, Y0) \
	VACCQ   X0, Y0, CAR1        \
	VAQ     X0, Y0, T0          \
	VACCCQ  X1, Y1, CAR1, T2    \
	VACQ    X1, Y1, CAR1, T1    \
	                            \
	VZERO   ZER                 \
	VSCBIQ  PL, T0, CAR1        \
	VSQ     PL, T0, TT0         \
	VSBCBIQ T1, PH, CAR1, CAR2  \
	VSBIQ   T1, PH, CAR1, TT1   \
	VSBIQ   T2, ZER, CAR2, SEL1 \
	                            \
	VSEL    T0, TT0, SEL1, T0   \
	VSEL    T1, TT1, SEL1, T1

#define p256HalfInternal(T1, T0, X1, X0) \
	VZERO  ZER                \
	VSBIQ  ZER, ZER, X0, SEL1 \
	                          \
	VACCQ  X0, PL, CAR1       \
	VAQ    X0, PL, T0         \
	VACCCQ X1, PH, CAR1, T2   \
	VACQ   X1, PH, CAR1, T1   \
	                          \
	VSEL   X0, T0, SEL1, T0   \
	VSEL   X1, T1, SEL1, T1   \
	VSEL   ZER, T2, SEL1, T2  \
	                          \
	VSLDB  $15, T2, ZER, TT1  \
	VSLDB  $15, T1, ZER, TT0  \
	VREPIB $1, SEL1           \
	VSRL   SEL1, T0, T0       \
	VSRL   SEL1, T1, T1       \
	VREPIB $7, SEL1           \
	VSL    SEL1, TT0, TT0     \
	VSL    SEL1, TT1, TT1     \
	VO     T0, TT0, T0        \
	VO     T1, TT1, T1

// ---------------------------------------
// func p256Mul(res, in1, in2 *p256Element)
#define res_ptr R1
#define x_ptr   R2
#define y_ptr   R3
#define CPOOL   R4

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
TEXT ·p256Mul(SB), NOSPLIT, $0
	MOVD res+0(FP), res_ptr
	MOVD in1+8(FP), x_ptr
	MOVD in2+16(FP), y_ptr

	VL   (0*16)(x_ptr), X0
	VPDI $0x4, X0, X0, X0
	VL   (1*16)(x_ptr), X1
	VPDI $0x4, X1, X1, X1
	VL   (0*16)(y_ptr), Y0
	VPDI $0x4, Y0, Y0, Y0
	VL   (1*16)(y_ptr), Y1
	VPDI $0x4, Y1, Y1, Y1

	MOVD $p256mul<>+0x00(SB), CPOOL
	VL   16(CPOOL), P0
	VL   0(CPOOL), P1

	CALL p256MulInternal<>(SB)

	VPDI $0x4, T0, T0, T0
	VST  T0, (0*16)(res_ptr)
	VPDI $0x4, T1, T1, T1
	VST  T1, (1*16)(res_ptr)
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

// ---------------------------------------
//  func p256Sqr(res, in *p256Element, n int)
#define res_ptr R1
#define x_ptr   R2
#define y_ptr   R3
#define CPOOL   R4
#define COUNT   R5
#define N       R6

// Parameters
#define X0    V0
#define X1    V1
#define T0    V4
#define T1    V5

// Constants
#define P0    V30
#define P1    V31
TEXT ·p256Sqr(SB), NOSPLIT, $0
	MOVD res+0(FP), res_ptr
	MOVD in+8(FP), x_ptr

	VL   (0*16)(x_ptr), X0
	VPDI $0x4, X0, X0, X0
	VL   (1*16)(x_ptr), X1
	VPDI $0x4, X1, X1, X1

	MOVD $p256mul<>+0x00(SB), CPOOL
	MOVD $0, COUNT
	MOVD n+16(FP), N
	VL   16(CPOOL), P0
	VL   0(CPOOL), P1

loop:
	CALL p256SqrInternal<>(SB)
	VLR  T0, X0
	VLR  T1, X1
	ADDW $1, COUNT
	CMPW COUNT, N
	BLT  loop

	VPDI $0x4, T0, T0, T0
	VST  T0, (0*16)(res_ptr)
	VPDI $0x4, T1, T1, T1
	VST  T1, (1*16)(res_ptr)
	RET

#undef res_ptr
#undef x_ptr
#undef y_ptr
#undef CPOOL
#undef COUNT
#undef N

#undef X0
#undef X1
#undef T0
#undef T1
#undef P0
#undef P1

// Point add with P2 being affine point
// If sign == 1 -> P2 = -P2
// If sel == 0 -> P3 = P1
// if zero == 0 -> P3 = P2
// func p256PointAddAffineAsm(res, in1 *P256Point, in2 *p256AffinePoint, sign, sel, zero int)
#define P3ptr   R1
#define P1ptr   R2
#define P2ptr   R3
#define CPOOL   R4

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
#define Z3L    V28
#define Z3H    V29

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
TEXT ·p256PointAddAffineAsm(SB), NOSPLIT, $0
	MOVD res+0(FP), P3ptr
	MOVD in1+8(FP), P1ptr
	MOVD in2+16(FP), P2ptr

	MOVD $p256mul<>+0x00(SB), CPOOL
	VL   16(CPOOL), PL
	VL   0(CPOOL), PH

	//	if (sign == 1) {
	//		Y2 = fromBig(new(big.Int).Mod(new(big.Int).Sub(p256.P, new(big.Int).SetBytes(Y2)), p256.P)) // Y2  = P-Y2
	//	}

	VL   48(P2ptr), Y2H
	VPDI $0x4, Y2H, Y2H, Y2H
	VL   32(P2ptr), Y2L
	VPDI $0x4, Y2L, Y2L, Y2L

	VLREPG sign+24(FP), SEL1
	VZERO  ZER
	VCEQG  SEL1, ZER, SEL1

	VSCBIQ Y2L, PL, CAR1
	VSQ    Y2L, PL, T1L
	VSBIQ  PH, Y2H, CAR1, T1H

	VSEL Y2L, T1L, SEL1, Y2L
	VSEL Y2H, T1H, SEL1, Y2H

/* *
 * Three operand formula:
 * Source: 2004 Hankerson–Menezes–Vanstone, page 91.
 */
	// X=Z1; Y=Z1; MUL; T-   // T1 = Z1²      T1
	VL   80(P1ptr), X1       // Z1H
	VPDI $0x4, X1, X1, X1
	VL   64(P1ptr), X0       // Z1L
	VPDI $0x4, X0, X0, X0
	VLR  X0, Y0
	VLR  X1, Y1
	CALL p256SqrInternal<>(SB)

	// X=T ; Y-  ; MUL; T2=T // T2 = T1*Z1    T1   T2
	VLR  T0, X0
	VLR  T1, X1
	CALL p256MulInternal<>(SB)
	VLR  T0, T2L
	VLR  T1, T2H

	// X-  ; Y=X2; MUL; T1=T // T1 = T1*X2    T1   T2
	VL   16(P2ptr), Y1       // X2H
	VPDI $0x4, Y1, Y1, Y1
	VL   0(P2ptr), Y0        // X2L
	VPDI $0x4, Y0, Y0, Y0
	CALL p256MulInternal<>(SB)
	VLR  T0, T1L
	VLR  T1, T1H

	// X=T2; Y=Y2; MUL; T-   // T2 = T2*Y2    T1   T2
	VLR  T2L, X0
	VLR  T2H, X1
	VLR  Y2L, Y0
	VLR  Y2H, Y1
	CALL p256MulInternal<>(SB)

	// SUB(T2<T-Y1)          // T2 = T2-Y1    T1   T2
	VL   48(P1ptr), Y1H
	VPDI $0x4, Y1H, Y1H, Y1H
	VL   32(P1ptr), Y1L
	VPDI $0x4, Y1L, Y1L, Y1L
	p256SubInternal(T2H,T2L,T1,T0,Y1H,Y1L)

	// SUB(Y<T1-X1)          // T1 = T1-X1    T1   T2
	VL   16(P1ptr), X1H
	VPDI $0x4, X1H, X1H, X1H
	VL   0(P1ptr), X1L
	VPDI $0x4, X1L, X1L, X1L
	p256SubInternal(Y1,Y0,T1H,T1L,X1H,X1L)

	// X=Z1; Y- ;  MUL; Z3:=T// Z3 = Z1*T1         T2
	VL   80(P1ptr), X1       // Z1H
	VPDI $0x4, X1, X1, X1
	VL   64(P1ptr), X0       // Z1L
	VPDI $0x4, X0, X0, X0
	CALL p256MulInternal<>(SB)

	// VST T1, 64(P3ptr)
	// VST T0, 80(P3ptr)
	VLR T0, Z3L
	VLR T1, Z3H

	// X=Y;  Y- ;  MUL; X=T  // T3 = T1*T1         T2
	VLR  Y0, X0
	VLR  Y1, X1
	CALL p256SqrInternal<>(SB)
	VLR  T0, X0
	VLR  T1, X1

	// X- ;  Y- ;  MUL; T4=T // T4 = T3*T1         T2        T4
	CALL p256MulInternal<>(SB)
	VLR  T0, T4L
	VLR  T1, T4H

	// X- ;  Y=X1; MUL; T3=T // T3 = T3*X1         T2   T3   T4
	VL   16(P1ptr), Y1       // X1H
	VPDI $0x4, Y1, Y1, Y1
	VL   0(P1ptr), Y0        // X1L
	VPDI $0x4, Y0, Y0, Y0
	CALL p256MulInternal<>(SB)
	VLR  T0, T3L
	VLR  T1, T3H

	// ADD(T1<T+T)           // T1 = T3+T3    T1   T2   T3   T4
	p256AddInternal(T1H,T1L, T1,T0,T1,T0)

	// X=T2; Y=T2; MUL; T-   // X3 = T2*T2    T1   T2   T3   T4
	VLR  T2L, X0
	VLR  T2H, X1
	VLR  T2L, Y0
	VLR  T2H, Y1
	CALL p256SqrInternal<>(SB)

	// SUB(T<T-T1)           // X3 = X3-T1    T1   T2   T3   T4  (T1 = X3)
	p256SubInternal(T1,T0,T1,T0,T1H,T1L)

	// SUB(T<T-T4) X3:=T     // X3 = X3-T4         T2   T3   T4
	p256SubInternal(T1,T0,T1,T0,T4H,T4L)
	VLR T0, X3L
	VLR T1, X3H

	// SUB(X<T3-T)           // T3 = T3-X3         T2   T3   T4
	p256SubInternal(X1,X0,T3H,T3L,T1,T0)

	// X- ;  Y- ;  MUL; T3=T // T3 = T3*T2         T2   T3   T4
	CALL p256MulInternal<>(SB)
	VLR  T0, T3L
	VLR  T1, T3H

	// X=T4; Y=Y1; MUL; T-   // T4 = T4*Y1              T3   T4
	VLR  T4L, X0
	VLR  T4H, X1
	VL   48(P1ptr), Y1       // Y1H
	VPDI $0x4, Y1, Y1, Y1
	VL   32(P1ptr), Y0       // Y1L
	VPDI $0x4, Y0, Y0, Y0
	CALL p256MulInternal<>(SB)

	// SUB(T<T3-T) Y3:=T     // Y3 = T3-T4              T3   T4  (T3 = Y3)
	p256SubInternal(Y3H,Y3L,T3H,T3L,T1,T0)

	//	if (sel == 0) {
	//		copy(P3.x[:], X1)
	//		copy(P3.y[:], Y1)
	//		copy(P3.z[:], Z1)
	//	}

	VL   16(P1ptr), X1H
	VPDI $0x4, X1H, X1H, X1H
	VL   0(P1ptr), X1L
	VPDI $0x4, X1L, X1L, X1L

	// Y1 already loaded, left over from addition
	VL   80(P1ptr), Z1H
	VPDI $0x4, Z1H, Z1H, Z1H
	VL   64(P1ptr), Z1L
	VPDI $0x4, Z1L, Z1L, Z1L

	VLREPG sel+32(FP), SEL1
	VZERO  ZER
	VCEQG  SEL1, ZER, SEL1

	VSEL X1L, X3L, SEL1, X3L
	VSEL X1H, X3H, SEL1, X3H
	VSEL Y1L, Y3L, SEL1, Y3L
	VSEL Y1H, Y3H, SEL1, Y3H
	VSEL Z1L, Z3L, SEL1, Z3L
	VSEL Z1H, Z3H, SEL1, Z3H

	//	if (zero == 0) {
	//		copy(P3.x[:], X2)
	//		copy(P3.y[:], Y2)
	//		copy(P3.z[:], []byte{0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	//			0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01})  //(p256.z*2^256)%p
	//	}
	VL   16(P2ptr), X2H
	VPDI $0x4, X2H, X2H, X2H
	VL   0(P2ptr), X2L
	VPDI $0x4, X2L, X2L, X2L

	// Y2 already loaded
	VL 128(CPOOL), Z2H
	VL 144(CPOOL), Z2L

	VLREPG zero+40(FP), SEL1
	VZERO  ZER
	VCEQG  SEL1, ZER, SEL1

	VSEL X2L, X3L, SEL1, X3L
	VSEL X2H, X3H, SEL1, X3H
	VSEL Y2L, Y3L, SEL1, Y3L
	VSEL Y2H, Y3H, SEL1, Y3H
	VSEL Z2L, Z3L, SEL1, Z3L
	VSEL Z2H, Z3H, SEL1, Z3H

	// All done, store out the result!!!
	VPDI $0x4, X3H, X3H, X3H
	VST  X3H, 16(P3ptr)
	VPDI $0x4, X3L, X3L, X3L
	VST  X3L, 0(P3ptr)
	VPDI $0x4, Y3H, Y3H, Y3H
	VST  Y3H, 48(P3ptr)
	VPDI $0x4, Y3L, Y3L, Y3L
	VST  Y3L, 32(P3ptr)
	VPDI $0x4, Z3H, Z3H, Z3H
	VST  Z3H, 80(P3ptr)
	VPDI $0x4, Z3L, Z3L, Z3L
	VST  Z3L, 64(P3ptr)

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

// func p256PointDoubleAsm(res, in *P256Point)
// https://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian.html#doubling-dbl-2007-bl
// https://www.hyperelliptic.org/EFD/g1p/auto-shortw.html
// https://www.hyperelliptic.org/EFD/g1p/auto-shortw-projective-3.html
#define P3ptr   R1
#define P1ptr   R2
#define CPOOL   R4

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
 * https://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-3.html#doubling-dbl-2004-hmv
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

TEXT ·p256PointDoubleAsm(SB), NOSPLIT, $0
	MOVD res+0(FP), P3ptr
	MOVD in+8(FP), P1ptr

	MOVD $p256mul<>+0x00(SB), CPOOL
	VL   16(CPOOL), PL
	VL   0(CPOOL), PH

	// X=Z1; Y=Z1; MUL; T-    // T1 = Z1²
	VL   80(P1ptr), X1        // Z1H
	VPDI $0x4, X1, X1, X1
	VL   64(P1ptr), X0        // Z1L
	VPDI $0x4, X0, X0, X0
	VLR  X0, Y0
	VLR  X1, Y1
	CALL p256SqrInternal<>(SB)

	// SUB(X<X1-T)            // T2 = X1-T1
	VL   16(P1ptr), X1H
	VPDI $0x4, X1H, X1H, X1H
	VL   0(P1ptr), X1L
	VPDI $0x4, X1L, X1L, X1L
	p256SubInternal(X1,X0,X1H,X1L,T1,T0)

	// ADD(Y<X1+T)            // T1 = X1+T1
	p256AddInternal(Y1,Y0,X1H,X1L,T1,T0)

	// X-  ; Y-  ; MUL; T-    // T2 = T2*T1
	CALL p256MulInternal<>(SB)

	// ADD(T2<T+T); ADD(T2<T2+T)  // T2 = 3*T2
	p256AddInternal(T2H,T2L,T1,T0,T1,T0)
	p256AddInternal(T2H,T2L,T2H,T2L,T1,T0)

	// ADD(X<Y1+Y1)           // Y3 = 2*Y1
	VL   48(P1ptr), Y1H
	VPDI $0x4, Y1H, Y1H, Y1H
	VL   32(P1ptr), Y1L
	VPDI $0x4, Y1L, Y1L, Y1L
	p256AddInternal(X1,X0,Y1H,Y1L,Y1H,Y1L)

	// X-  ; Y=Z1; MUL; Z3:=T // Z3 = Y3*Z1
	VL   80(P1ptr), Y1        // Z1H
	VPDI $0x4, Y1, Y1, Y1
	VL   64(P1ptr), Y0        // Z1L
	VPDI $0x4, Y0, Y0, Y0
	CALL p256MulInternal<>(SB)
	VPDI $0x4, T1, T1, TT1
	VST  TT1, 80(P3ptr)
	VPDI $0x4, T0, T0, TT0
	VST  TT0, 64(P3ptr)

	// X-  ; Y=X ; MUL; T-    // Y3 = Y3²
	VLR  X0, Y0
	VLR  X1, Y1
	CALL p256SqrInternal<>(SB)

	// X=T ; Y=X1; MUL; T3=T  // T3 = Y3*X1
	VLR  T0, X0
	VLR  T1, X1
	VL   16(P1ptr), Y1
	VPDI $0x4, Y1, Y1, Y1
	VL   0(P1ptr), Y0
	VPDI $0x4, Y0, Y0, Y0
	CALL p256MulInternal<>(SB)
	VLR  T0, T3L
	VLR  T1, T3H

	// X-  ; Y=X ; MUL; T-    // Y3 = Y3²
	VLR  X0, Y0
	VLR  X1, Y1
	CALL p256SqrInternal<>(SB)

	// HAL(Y3<T)              // Y3 = half*Y3
	p256HalfInternal(Y3H,Y3L, T1,T0)

	// X=T2; Y=T2; MUL; T-    // X3 = T2²
	VLR  T2L, X0
	VLR  T2H, X1
	VLR  T2L, Y0
	VLR  T2H, Y1
	CALL p256SqrInternal<>(SB)

	// ADD(T1<T3+T3)          // T1 = 2*T3
	p256AddInternal(T1H,T1L,T3H,T3L,T3H,T3L)

	// SUB(X3<T-T1) X3:=X3    // X3 = X3-T1
	p256SubInternal(X3H,X3L,T1,T0,T1H,T1L)
	VPDI $0x4, X3H, X3H, TT1
	VST  TT1, 16(P3ptr)
	VPDI $0x4, X3L, X3L, TT0
	VST  TT0, 0(P3ptr)

	// SUB(X<T3-X3)           // T1 = T3-X3
	p256SubInternal(X1,X0,T3H,T3L,X3H,X3L)

	// X-  ; Y-  ; MUL; T-    // T1 = T1*T2
	CALL p256MulInternal<>(SB)

	// SUB(Y3<T-Y3)           // Y3 = T1-Y3
	p256SubInternal(Y3H,Y3L,T1,T0,Y3H,Y3L)

	VPDI $0x4, Y3H, Y3H, Y3H
	VST  Y3H, 48(P3ptr)
	VPDI $0x4, Y3L, Y3L, Y3L
	VST  Y3L, 32(P3ptr)
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

// func p256PointAddAsm(res, in1, in2 *P256Point) int
#define P3ptr  R1
#define P1ptr  R2
#define P2ptr  R3
#define CPOOL  R4
#define ISZERO R5
#define TRUE   R6

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
 * https://delta.cs.cinvestav.mx/~francisco/arith/julio.pdf "Software Implementation of the NIST Elliptic Curves Over Prime Fields"
 *
 * A = X₁×Z₂²
 * B = Y₁×Z₂³
 * C = X₂×Z₁²-A
 * D = Y₂×Z₁³-B
 * X₃ = D² - 2A×C² - C³
 * Y₃ = D×(A×C² - X₃) - B×C³
 * Z₃ = Z₁×Z₂×C
 *
 * Three-operand formula (adopted): https://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-3.html#addition-add-1998-cmo-2
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
TEXT ·p256PointAddAsm(SB), NOSPLIT, $0
	MOVD res+0(FP), P3ptr
	MOVD in1+8(FP), P1ptr
	MOVD in2+16(FP), P2ptr

	MOVD $p256mul<>+0x00(SB), CPOOL
	VL   16(CPOOL), PL
	VL   0(CPOOL), PH

	// X=Z1; Y=Z1; MUL; T-   // T1 = Z1*Z1
	VL   80(P1ptr), X1       // Z1H
	VPDI $0x4, X1, X1, X1
	VL   64(P1ptr), X0       // Z1L
	VPDI $0x4, X0, X0, X0
	VLR  X0, Y0
	VLR  X1, Y1
	CALL p256SqrInternal<>(SB)

	// X-  ; Y=T ; MUL; R=T  // R  = Z1*T1
	VLR  T0, Y0
	VLR  T1, Y1
	CALL p256MulInternal<>(SB)
	VLR  T0, RL
	VLR  T1, RH

	// X=X2; Y-  ; MUL; H=T  // H  = X2*T1
	VL   16(P2ptr), X1       // X2H
	VPDI $0x4, X1, X1, X1
	VL   0(P2ptr), X0        // X2L
	VPDI $0x4, X0, X0, X0
	CALL p256MulInternal<>(SB)
	VLR  T0, HL
	VLR  T1, HH

	// X=Z2; Y=Z2; MUL; T-   // T2 = Z2*Z2
	VL   80(P2ptr), X1       // Z2H
	VPDI $0x4, X1, X1, X1
	VL   64(P2ptr), X0       // Z2L
	VPDI $0x4, X0, X0, X0
	VLR  X0, Y0
	VLR  X1, Y1
	CALL p256SqrInternal<>(SB)

	// X-  ; Y=T ; MUL; S1=T // S1 = Z2*T2
	VLR  T0, Y0
	VLR  T1, Y1
	CALL p256MulInternal<>(SB)
	VLR  T0, S1L
	VLR  T1, S1H

	// X=X1; Y-  ; MUL; U1=T // U1 = X1*T2
	VL   16(P1ptr), X1       // X1H
	VPDI $0x4, X1, X1, X1
	VL   0(P1ptr), X0        // X1L
	VPDI $0x4, X0, X0, X0
	CALL p256MulInternal<>(SB)
	VLR  T0, U1L
	VLR  T1, U1H

	// SUB(H<H-T)            // H  = H-U1
	p256SubInternal(HH,HL,HH,HL,T1,T0)

	// if H == 0 or H^P == 0 then ret=1 else ret=0
	// clobbers T1H and T1L
	MOVD   $0, ISZERO
	MOVD   $1, TRUE
	VZERO  ZER
	VO     HL, HH, T1H
	VCEQGS ZER, T1H, T1H
	MOVDEQ TRUE, ISZERO
	VX     HL, PL, T1L
	VX     HH, PH, T1H
	VO     T1L, T1H, T1H
	VCEQGS ZER, T1H, T1H
	MOVDEQ TRUE, ISZERO
	MOVD   ISZERO, ret+24(FP)

	// X=Z1; Y=Z2; MUL; T-   // Z3 = Z1*Z2
	VL   80(P1ptr), X1       // Z1H
	VPDI $0x4, X1, X1, X1
	VL   64(P1ptr), X0       // Z1L
	VPDI $0x4, X0, X0, X0
	VL   80(P2ptr), Y1       // Z2H
	VPDI $0x4, Y1, Y1, Y1
	VL   64(P2ptr), Y0       // Z2L
	VPDI $0x4, Y0, Y0, Y0
	CALL p256MulInternal<>(SB)

	// X=T ; Y=H ; MUL; Z3:=T// Z3 = Z3*H
	VLR  T0, X0
	VLR  T1, X1
	VLR  HL, Y0
	VLR  HH, Y1
	CALL p256MulInternal<>(SB)
	VPDI $0x4, T1, T1, TT1
	VST  TT1, 80(P3ptr)
	VPDI $0x4, T0, T0, TT0
	VST  TT0, 64(P3ptr)

	// X=Y1; Y=S1; MUL; S1=T // S1 = Y1*S1
	VL   48(P1ptr), X1
	VPDI $0x4, X1, X1, X1
	VL   32(P1ptr), X0
	VPDI $0x4, X0, X0, X0
	VLR  S1L, Y0
	VLR  S1H, Y1
	CALL p256MulInternal<>(SB)
	VLR  T0, S1L
	VLR  T1, S1H

	// X=Y2; Y=R ; MUL; T-   // R  = Y2*R
	VL   48(P2ptr), X1
	VPDI $0x4, X1, X1, X1
	VL   32(P2ptr), X0
	VPDI $0x4, X0, X0, X0
	VLR  RL, Y0
	VLR  RH, Y1
	CALL p256MulInternal<>(SB)

	// SUB(R<T-S1)           // R  = T-S1
	p256SubInternal(RH,RL,T1,T0,S1H,S1L)

	// if R == 0 or R^P == 0 then ret=ret else ret=0
	// clobbers T1H and T1L
	MOVD   $0, ISZERO
	MOVD   $1, TRUE
	VZERO  ZER
	VO     RL, RH, T1H
	VCEQGS ZER, T1H, T1H
	MOVDEQ TRUE, ISZERO
	VX     RL, PL, T1L
	VX     RH, PH, T1H
	VO     T1L, T1H, T1H
	VCEQGS ZER, T1H, T1H
	MOVDEQ TRUE, ISZERO
	AND    ret+24(FP), ISZERO
	MOVD   ISZERO, ret+24(FP)

	// X=H ; Y=H ; MUL; T-   // T1 = H*H
	VLR  HL, X0
	VLR  HH, X1
	VLR  HL, Y0
	VLR  HH, Y1
	CALL p256SqrInternal<>(SB)

	// X-  ; Y=T ; MUL; T2=T // T2 = H*T1
	VLR  T0, Y0
	VLR  T1, Y1
	CALL p256MulInternal<>(SB)
	VLR  T0, T2L
	VLR  T1, T2H

	// X=U1; Y-  ; MUL; U1=T // U1 = U1*T1
	VLR  U1L, X0
	VLR  U1H, X1
	CALL p256MulInternal<>(SB)
	VLR  T0, U1L
	VLR  T1, U1H

	// X=R ; Y=R ; MUL; T-   // X3 = R*R
	VLR  RL, X0
	VLR  RH, X1
	VLR  RL, Y0
	VLR  RH, Y1
	CALL p256SqrInternal<>(SB)

	// SUB(T<T-T2)           // X3 = X3-T2
	p256SubInternal(T1,T0,T1,T0,T2H,T2L)

	// ADD(X<U1+U1)          // T1 = 2*U1
	p256AddInternal(X1,X0,U1H,U1L,U1H,U1L)

	// SUB(T<T-X) X3:=T      // X3 = X3-T1 << store-out X3 result reg
	p256SubInternal(T1,T0,T1,T0,X1,X0)
	VPDI $0x4, T1, T1, TT1
	VST  TT1, 16(P3ptr)
	VPDI $0x4, T0, T0, TT0
	VST  TT0, 0(P3ptr)

	// SUB(Y<U1-T)           // Y3 = U1-X3
	p256SubInternal(Y1,Y0,U1H,U1L,T1,T0)

	// X=R ; Y-  ; MUL; U1=T // Y3 = R*Y3
	VLR  RL, X0
	VLR  RH, X1
	CALL p256MulInternal<>(SB)
	VLR  T0, U1L
	VLR  T1, U1H

	// X=S1; Y=T2; MUL; T-   // T2 = S1*T2
	VLR  S1L, X0
	VLR  S1H, X1
	VLR  T2L, Y0
	VLR  T2H, Y1
	CALL p256MulInternal<>(SB)

	// SUB(T<U1-T); Y3:=T    // Y3 = Y3-T2 << store-out Y3 result reg
	p256SubInternal(T1,T0,U1H,U1L,T1,T0)
	VPDI $0x4, T1, T1, T1
	VST  T1, 48(P3ptr)
	VPDI $0x4, T0, T0, T0
	VST  T0, 32(P3ptr)

	RET
