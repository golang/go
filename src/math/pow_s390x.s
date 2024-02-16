// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

#define PosInf   0x7FF0000000000000
#define NaN      0x7FF8000000000001
#define NegInf   0xFFF0000000000000
#define PosOne   0x3FF0000000000000
#define NegOne   0xBFF0000000000000
#define NegZero  0x8000000000000000

// Minimax polynomial approximation
DATA ·powrodataL51<> + 0(SB)/8, $-1.0
DATA ·powrodataL51<> + 8(SB)/8, $1.0
DATA ·powrodataL51<> + 16(SB)/8, $0.24022650695910110361E+00
DATA ·powrodataL51<> + 24(SB)/8, $0.69314718055994686185E+00
DATA ·powrodataL51<> + 32(SB)/8, $0.96181291057109484809E-02
DATA ·powrodataL51<> + 40(SB)/8, $0.15403814778342868389E-03
DATA ·powrodataL51<> + 48(SB)/8, $0.55504108652095235601E-01
DATA ·powrodataL51<> + 56(SB)/8, $0.13333818813168698658E-02
DATA ·powrodataL51<> + 64(SB)/8, $0.68205322933914439200E-12
DATA ·powrodataL51<> + 72(SB)/8, $-.18466496523378731640E-01
DATA ·powrodataL51<> + 80(SB)/8, $0.19697596291603973706E-02
DATA ·powrodataL51<> + 88(SB)/8, $0.23083120654155209200E+00
DATA ·powrodataL51<> + 96(SB)/8, $0.55324356012093416771E-06
DATA ·powrodataL51<> + 104(SB)/8, $-.40340677224649339048E-05
DATA ·powrodataL51<> + 112(SB)/8, $0.30255507904062541562E-04
DATA ·powrodataL51<> + 120(SB)/8, $-.77453979912413008787E-07
DATA ·powrodataL51<> + 128(SB)/8, $-.23637115549923464737E-03
DATA ·powrodataL51<> + 136(SB)/8, $0.11016119077267717198E-07
DATA ·powrodataL51<> + 144(SB)/8, $0.22608272174486123035E-09
DATA ·powrodataL51<> + 152(SB)/8, $-.15895808101370190382E-08
DATA ·powrodataL51<> + 160(SB)/8, $0x4540190000000000
GLOBL ·powrodataL51<> + 0(SB), RODATA, $168

// Constants
DATA ·pow_x001a<> + 0(SB)/8, $0x1a000000000000
GLOBL ·pow_x001a<> + 0(SB), RODATA, $8
DATA ·pow_xinf<> + 0(SB)/8, $0x7ff0000000000000      //+Inf
GLOBL ·pow_xinf<> + 0(SB), RODATA, $8
DATA ·pow_xnan<> + 0(SB)/8, $0x7ff8000000000000      //NaN
GLOBL ·pow_xnan<> + 0(SB), RODATA, $8
DATA ·pow_x434<> + 0(SB)/8, $0x4340000000000000
GLOBL ·pow_x434<> + 0(SB), RODATA, $8
DATA ·pow_x433<> + 0(SB)/8, $0x4330000000000000
GLOBL ·pow_x433<> + 0(SB), RODATA, $8
DATA ·pow_x43f<> + 0(SB)/8, $0x43f0000000000000
GLOBL ·pow_x43f<> + 0(SB), RODATA, $8
DATA ·pow_xadd<> + 0(SB)/8, $0xc2f0000100003fef
GLOBL ·pow_xadd<> + 0(SB), RODATA, $8
DATA ·pow_xa<> + 0(SB)/8, $0x4019000000000000
GLOBL ·pow_xa<> + 0(SB), RODATA, $8

// Scale correction tables
DATA powiadd<> + 0(SB)/8, $0xf000000000000000
DATA powiadd<> + 8(SB)/8, $0x1000000000000000
GLOBL powiadd<> + 0(SB), RODATA, $16
DATA powxscale<> + 0(SB)/8, $0x4ff0000000000000
DATA powxscale<> + 8(SB)/8, $0x2ff0000000000000
GLOBL powxscale<> + 0(SB), RODATA, $16

// Fractional powers of 2 table
DATA ·powtexp<> + 0(SB)/8, $0.442737824274138381E-01
DATA ·powtexp<> + 8(SB)/8, $0.263602189790660309E-01
DATA ·powtexp<> + 16(SB)/8, $0.122565642281703586E-01
DATA ·powtexp<> + 24(SB)/8, $0.143757052860721398E-02
DATA ·powtexp<> + 32(SB)/8, $-.651375034121276075E-02
DATA ·powtexp<> + 40(SB)/8, $-.119317678849450159E-01
DATA ·powtexp<> + 48(SB)/8, $-.150868749549871069E-01
DATA ·powtexp<> + 56(SB)/8, $-.161992609578469234E-01
DATA ·powtexp<> + 64(SB)/8, $-.154492360403337917E-01
DATA ·powtexp<> + 72(SB)/8, $-.129850717389178721E-01
DATA ·powtexp<> + 80(SB)/8, $-.892902649276657891E-02
DATA ·powtexp<> + 88(SB)/8, $-.338202636596794887E-02
DATA ·powtexp<> + 96(SB)/8, $0.357266307045684762E-02
DATA ·powtexp<> + 104(SB)/8, $0.118665304327406698E-01
DATA ·powtexp<> + 112(SB)/8, $0.214434994118118914E-01
DATA ·powtexp<> + 120(SB)/8, $0.322580645161290314E-01
GLOBL ·powtexp<> + 0(SB), RODATA, $128

// Log multiplier tables
DATA ·powtl<> + 0(SB)/8, $0xbdf9723a80db6a05
DATA ·powtl<> + 8(SB)/8, $0x3e0cfe4a0babe862
DATA ·powtl<> + 16(SB)/8, $0xbe163b42dd33dada
DATA ·powtl<> + 24(SB)/8, $0xbe0cdf9de2a8429c
DATA ·powtl<> + 32(SB)/8, $0xbde9723a80db6a05
DATA ·powtl<> + 40(SB)/8, $0xbdb37fcae081745e
DATA ·powtl<> + 48(SB)/8, $0xbdd8b2f901ac662c
DATA ·powtl<> + 56(SB)/8, $0xbde867dc68c36cc9
DATA ·powtl<> + 64(SB)/8, $0xbdd23e36b47256b7
DATA ·powtl<> + 72(SB)/8, $0xbde4c9b89fcc7933
DATA ·powtl<> + 80(SB)/8, $0xbdd16905cad7cf66
DATA ·powtl<> + 88(SB)/8, $0x3ddb417414aa5529
DATA ·powtl<> + 96(SB)/8, $0xbdce046f2889983c
DATA ·powtl<> + 104(SB)/8, $0x3dc2c3865d072897
DATA ·powtl<> + 112(SB)/8, $0x8000000000000000
DATA ·powtl<> + 120(SB)/8, $0x3dc1ca48817f8afe
DATA ·powtl<> + 128(SB)/8, $0xbdd703518a88bfb7
DATA ·powtl<> + 136(SB)/8, $0x3dc64afcc46942ce
DATA ·powtl<> + 144(SB)/8, $0xbd9d79191389891a
DATA ·powtl<> + 152(SB)/8, $0x3ddd563044da4fa0
DATA ·powtl<> + 160(SB)/8, $0x3e0f42b5e5f8f4b6
DATA ·powtl<> + 168(SB)/8, $0x3e0dfa2c2cbf6ead
DATA ·powtl<> + 176(SB)/8, $0x3e14e25e91661293
DATA ·powtl<> + 184(SB)/8, $0x3e0aac461509e20c
GLOBL ·powtl<> + 0(SB), RODATA, $192

DATA ·powtm<> + 0(SB)/8, $0x3da69e13
DATA ·powtm<> + 8(SB)/8, $0x100003d66fcb6
DATA ·powtm<> + 16(SB)/8, $0x200003d1538df
DATA ·powtm<> + 24(SB)/8, $0x300003cab729e
DATA ·powtm<> + 32(SB)/8, $0x400003c1a784c
DATA ·powtm<> + 40(SB)/8, $0x500003ac9b074
DATA ·powtm<> + 48(SB)/8, $0x60000bb498d22
DATA ·powtm<> + 56(SB)/8, $0x68000bb8b29a2
DATA ·powtm<> + 64(SB)/8, $0x70000bb9a32d4
DATA ·powtm<> + 72(SB)/8, $0x74000bb9946bb
DATA ·powtm<> + 80(SB)/8, $0x78000bb92e34b
DATA ·powtm<> + 88(SB)/8, $0x80000bb6c57dc
DATA ·powtm<> + 96(SB)/8, $0x84000bb4020f7
DATA ·powtm<> + 104(SB)/8, $0x8c000ba93832d
DATA ·powtm<> + 112(SB)/8, $0x9000080000000
DATA ·powtm<> + 120(SB)/8, $0x940003aa66c4c
DATA ·powtm<> + 128(SB)/8, $0x980003b2fb12a
DATA ·powtm<> + 136(SB)/8, $0xa00003bc1def6
DATA ·powtm<> + 144(SB)/8, $0xa80003c1eb0eb
DATA ·powtm<> + 152(SB)/8, $0xb00003c64dcec
DATA ·powtm<> + 160(SB)/8, $0xc00003cc49e4e
DATA ·powtm<> + 168(SB)/8, $0xd00003d12f1de
DATA ·powtm<> + 176(SB)/8, $0xe00003d4a9c6f
DATA ·powtm<> + 184(SB)/8, $0xf00003d846c66
GLOBL ·powtm<> + 0(SB), RODATA, $192

// Table of indices into multiplier tables
// Adjusted from asm to remove offset and convert
DATA ·powtabi<> + 0(SB)/8, $0x1010101
DATA ·powtabi<> + 8(SB)/8, $0x101020202020203
DATA ·powtabi<> + 16(SB)/8, $0x303030404040405
DATA ·powtabi<> + 24(SB)/8, $0x505050606060708
DATA ·powtabi<> + 32(SB)/8, $0x90a0b0c0d0e0f10
DATA ·powtabi<> + 40(SB)/8, $0x1011111212121313
DATA ·powtabi<> + 48(SB)/8, $0x1314141414151515
DATA ·powtabi<> + 56(SB)/8, $0x1516161617171717
GLOBL ·powtabi<> + 0(SB), RODATA, $64

// Pow returns x**y, the base-x exponential of y.
//
// Special cases are (in order):
//      Pow(x, ±0) = 1 for any x
//      Pow(1, y) = 1 for any y
//      Pow(x, 1) = x for any x
//      Pow(NaN, y) = NaN
//      Pow(x, NaN) = NaN
//      Pow(±0, y) = ±Inf for y an odd integer < 0
//      Pow(±0, -Inf) = +Inf
//      Pow(±0, +Inf) = +0
//      Pow(±0, y) = +Inf for finite y < 0 and not an odd integer
//      Pow(±0, y) = ±0 for y an odd integer > 0
//      Pow(±0, y) = +0 for finite y > 0 and not an odd integer
//      Pow(-1, ±Inf) = 1
//      Pow(x, +Inf) = +Inf for |x| > 1
//      Pow(x, -Inf) = +0 for |x| > 1
//      Pow(x, +Inf) = +0 for |x| < 1
//      Pow(x, -Inf) = +Inf for |x| < 1
//      Pow(+Inf, y) = +Inf for y > 0
//      Pow(+Inf, y) = +0 for y < 0
//      Pow(-Inf, y) = Pow(-0, -y)
//      Pow(x, y) = NaN for finite x < 0 and finite non-integer y

TEXT	·powAsm(SB), NOSPLIT, $0-24
	// special case
	MOVD	x+0(FP), R1
	MOVD	y+8(FP), R2

	// special case Pow(1, y) = 1 for any y
	MOVD	$PosOne, R3
	CMPUBEQ	R1, R3, xIsOne

	// special case Pow(x, 1) = x for any x
	MOVD	$PosOne, R4
	CMPUBEQ	R2, R4, yIsOne

	// special case Pow(x, NaN) = NaN for any x
	MOVD	$~(1<<63), R5
	AND	R2, R5    // y = |y|
	MOVD	$PosInf, R4
	CMPUBLT R4, R5, yIsNan

	MOVD	$NegInf, R3
	CMPUBEQ	R1, R3, xIsNegInf

	MOVD	$NegOne, R3
	CMPUBEQ	R1, R3, xIsNegOne

	MOVD	$PosInf, R3
	CMPUBEQ	R1, R3, xIsPosInf

	MOVD	$NegZero, R3
	CMPUBEQ	R1, R3, xIsNegZero

	MOVD	$PosInf, R4
	CMPUBEQ	R2, R4, yIsPosInf

	MOVD	$0x0, R3
	CMPUBEQ	R1, R3, xIsPosZero
	CMPBLT	R1, R3, xLtZero
	BR	Normal
xIsPosInf:
	// special case Pow(+Inf, y) = +Inf for y > 0
	MOVD	$0x0, R4
	CMPBGT	R2, R4, posInfGeZero
	BR	Normal
xIsNegInf:
	//Pow(-Inf, y) = Pow(-0, -y)
	FMOVD y+8(FP), F2
	FNEG F2, F2			// y = -y
	BR negZeroNegY		// call Pow(-0, -y)
xIsNegOne:
	// special case Pow(-1, ±Inf) = 1
	MOVD	$PosInf, R4
	CMPUBEQ	R2, R4, negOnePosInf
	MOVD	$NegInf, R4
	CMPUBEQ	R2, R4, negOneNegInf
	BR	Normal
xIsPosZero:
	// special case Pow(+0, -Inf) = +Inf
	MOVD	$NegInf, R4
	CMPUBEQ	R2, R4, zeroNegInf

	// special case Pow(+0, y < 0) = +Inf
	FMOVD	y+8(FP), F2
	FMOVD	$(0.0), F4
	FCMPU	F2, F4
	BLT	posZeroLtZero				//y < 0.0
	BR	Normal
xIsNegZero:
	// special case Pow(-0, -Inf) = +Inf
	MOVD	$NegInf, R4
	CMPUBEQ	R2, R4, zeroNegInf
	FMOVD	y+8(FP), F2
negZeroNegY:
	// special case Pow(x, ±0) = 1 for any x
	FMOVD	$(0.0), F4
	FCMPU	F4, F2
	BLT	negZeroGtZero		// y > 0.0
	BEQ yIsZero				// y = 0.0

	FMOVD $(-0.0), F4
	FCMPU F4, F2
	BLT negZeroGtZero				// y > -0.0
	BEQ yIsZero				// y = -0.0

	// special case Pow(-0, y) = -Inf for y an odd integer < 0
	// special case Pow(-0, y) = +Inf for finite y < 0 and not an odd integer
	FIDBR	$5, F2, F4		//F2 translate to integer F4
	FCMPU	F2, F4
	BNE	zeroNotOdd			// y is not an (odd) integer and y < 0
	FMOVD	$(2.0), F4
	FDIV	F4, F2			// F2 = F2 / 2.0
	FIDBR	$5, F2, F4		//F2 translate to integer F4
	FCMPU	F2, F4
	BNE	negZeroOddInt		// y is an odd integer and y < 0
	BR	zeroNotOdd			// y is not an (odd) integer and y < 0

negZeroGtZero:
	// special case Pow(-0, y) = -0 for y an odd integer > 0
	// special case Pow(±0, y) = +0 for finite y > 0 and not an odd integer
	FIDBR	$5, F2, F4      //F2 translate to integer F4
	FCMPU	F2, F4
	BNE	zeroNotOddGtZero    // y is not an (odd) integer and y > 0
	FMOVD	$(2.0), F4
	FDIV	F4, F2          // F2 = F2 / 2.0
	FIDBR	$5, F2, F4      //F2 translate to integer F4
	FCMPU	F2, F4
	BNE	negZeroOddIntGtZero       // y is an odd integer and y > 0
	BR	zeroNotOddGtZero          // y is not an (odd) integer

xLtZero:
	// special case Pow(x, y) = NaN for finite x < 0 and finite non-integer y
	FMOVD	y+8(FP), F2
	FIDBR	$5, F2, F4
	FCMPU	F2, F4
	BNE	ltZeroInt
	BR	Normal
yIsPosInf:
	// special case Pow(x, +Inf) = +Inf for |x| > 1
	FMOVD	x+0(FP), F1
	FMOVD	$(1.0), F3
	FCMPU	F1, F3
	BGT	gtOnePosInf
	FMOVD	$(-1.0), F3
	FCMPU	F1, F3
	BLT	ltNegOnePosInf
Normal:
	FMOVD	x+0(FP), F0
	FMOVD	y+8(FP), F2
	MOVD	$·powrodataL51<>+0(SB), R9
	LGDR	F0, R3
	WORD	$0xC0298009	//iilf	%r2,2148095317
	BYTE	$0x55
	BYTE	$0x55
	RISBGNZ	$32, $63, $32, R3, R1
	SUBW	R1, R2
	RISBGNZ	$58, $63, $50, R2, R3
	BYTE	$0x18	//lr	%r5,%r1
	BYTE	$0x51
	MOVD	$·powtabi<>+0(SB), R12
	WORD	$0xE303C000	//llgc	%r0,0(%r3,%r12)
	BYTE	$0x00
	BYTE	$0x90
	SUBW	$0x1A0000, R5
	SLD	$3, R0, R3
	MOVD	$·powtm<>+0(SB), R4
	MOVH	$0x0, R8
	ANDW	$0x7FF00000, R2
	ORW	R5, R1
	WORD	$0x5A234000	//a	%r2,0(%r3,%r4)
	MOVD	$0x3FF0000000000000, R5
	RISBGZ	$40, $63, $56, R2, R3
	RISBGN	$0, $31, $32, R2, R8
	ORW	$0x45000000, R3
	MOVW	R1, R6
	CMPBLT	R6, $0, L42
	FMOVD	F0, F4
L2:
	VLVGF	$0, R3, V1
	MOVD	$·pow_xa<>+0(SB), R2
	WORD	$0xED3090A0	//lde	%f3,.L52-.L51(%r9)
	BYTE	$0x00
	BYTE	$0x24
	FMOVD	0(R2), F6
	FSUBS	F1, F3
	LDGR	R8, F1
	WFMSDB	V4, V1, V6, V4
	FMOVD	152(R9), F6
	WFMDB	V4, V4, V7
	FMOVD	144(R9), F1
	FMOVD	136(R9), F5
	WFMADB	V4, V1, V6, V1
	VLEG	$0, 128(R9), V16
	FMOVD	120(R9), F6
	WFMADB	V4, V5, V6, V5
	FMOVD	112(R9), F6
	WFMADB	V1, V7, V5, V1
	WFMADB	V4, V6, V16, V16
	SLD	$3, R0, R2
	FMOVD	104(R9), F5
	WORD	$0xED824004	//ldeb	%f8,4(%r2,%r4)
	BYTE	$0x00
	BYTE	$0x04
	LDEBR	F3, F3
	FMOVD	96(R9), F6
	WFMADB	V4, V6, V5, V6
	FADD	F8, F3
	WFMADB	V7, V6, V16, V6
	FMUL	F7, F7
	FMOVD	88(R9), F5
	FMADD	F7, F1, F6
	WFMADB	V4, V5, V3, V16
	FMOVD	80(R9), F1
	WFSDB	V16, V3, V3
	MOVD	$·powtl<>+0(SB), R3
	WFMADB	V4, V6, V1, V6
	FMADD	F5, F4, F3
	FMOVD	72(R9), F1
	WFMADB	V4, V6, V1, V6
	WORD	$0xED323000	//adb	%f3,0(%r2,%r3)
	BYTE	$0x00
	BYTE	$0x1A
	FMOVD	64(R9), F1
	WFMADB	V4, V6, V1, V6
	MOVD	$·pow_xadd<>+0(SB), R2
	WFMADB	V4, V6, V3, V4
	FMOVD	0(R2), F5
	WFADB	V4, V16, V3
	VLEG	$0, 56(R9), V20
	WFMSDB	V2, V3, V5, V3
	VLEG	$0, 48(R9), V18
	WFADB	V3, V5, V6
	LGDR	F3, R2
	WFMSDB	V2, V16, V6, V16
	FMOVD	40(R9), F1
	WFMADB	V2, V4, V16, V4
	FMOVD	32(R9), F7
	WFMDB	V4, V4, V3
	WFMADB	V4, V1, V20, V1
	WFMADB	V4, V7, V18, V7
	VLEG	$0, 24(R9), V16
	WFMADB	V1, V3, V7, V1
	FMOVD	16(R9), F5
	WFMADB	V4, V5, V16, V5
	RISBGZ	$57, $60, $3, R2, R4
	WFMADB	V3, V1, V5, V1
	MOVD	$·powtexp<>+0(SB), R3
	WORD	$0x68343000	//ld	%f3,0(%r4,%r3)
	FMADD	F3, F4, F4
	RISBGN	$0, $15, $48, R2, R5
	WFMADB	V4, V1, V3, V4
	LGDR	F6, R2
	LDGR	R5, F1
	SRAD	$48, R2, R2
	FMADD	F1, F4, F1
	RLL	$16, R2, R2
	ANDW	$0x7FFF0000, R2
	WORD	$0xC22B3F71	//alfi	%r2,1064370176
	BYTE	$0x00
	BYTE	$0x00
	ORW	R2, R1, R3
	MOVW	R3, R6
	CMPBLT	R6, $0, L43
L1:
	FMOVD	F1, ret+16(FP)
	RET
L43:
	LTDBR	F0, F0
	BLTU	L44
	FMOVD	F0, F3
L7:
	MOVD	$·pow_xinf<>+0(SB), R3
	FMOVD	0(R3), F5
	WFCEDBS	V3, V5, V7
	BVS	L8
	WFMDB	V3, V2, V6
L8:
	WFCEDBS	V2, V2, V3
	BVS	L9
	LTDBR	F2, F2
	BEQ	L26
	MOVW	R1, R6
	CMPBLT	R6, $0, L45
L11:
	WORD	$0xC0190003	//iilf	%r1,262143
	BYTE	$0xFF
	BYTE	$0xFF
	MOVW	R2, R7
	MOVW	R1, R6
	CMPBLE	R7, R6, L34
	RISBGNZ	$32, $63, $32, R5, R1
	LGDR	F6, R2
	MOVD	$powiadd<>+0(SB), R3
	RISBGZ	$60, $60, $4, R2, R2
	WORD	$0x5A123000	//a	%r1,0(%r2,%r3)
	RISBGN	$0, $31, $32, R1, R5
	LDGR	R5, F1
	FMADD	F1, F4, F1
	MOVD	$powxscale<>+0(SB), R1
	WORD	$0xED121000	//mdb	%f1,0(%r2,%r1)
	BYTE	$0x00
	BYTE	$0x1C
	BR	L1
L42:
	LTDBR	F0, F0
	BLTU	L46
	FMOVD	F0, F4
L3:
	MOVD	$·pow_x001a<>+0(SB), R2
	WORD	$0xED402000	//cdb	%f4,0(%r2)
	BYTE	$0x00
	BYTE	$0x19
	BGE	L2
	BVS	L2
	MOVD	$·pow_x43f<>+0(SB), R2
	WORD	$0xED402000	//mdb	%f4,0(%r2)
	BYTE	$0x00
	BYTE	$0x1C
	WORD	$0xC0298009	//iilf	%r2,2148095317
	BYTE	$0x55
	BYTE	$0x55
	LGDR	F4, R3
	RISBGNZ	$32, $63, $32, R3, R3
	SUBW	R3, R2, R3
	RISBGZ	$33, $43, $0, R3, R2
	RISBGNZ	$58, $63, $50, R3, R3
	WORD	$0xE303C000	//llgc	%r0,0(%r3,%r12)
	BYTE	$0x00
	BYTE	$0x90
	SLD	$3, R0, R3
	WORD	$0x5A234000	//a	%r2,0(%r3,%r4)
	BYTE	$0x18	//lr	%r3,%r2
	BYTE	$0x32
	RISBGN	$0, $31, $32, R3, R8
	ADDW	$0x4000000, R3
	BLEU	L5
	RISBGZ	$40, $63, $56, R3, R3
	ORW	$0x45000000, R3
	BR	L2
L9:
	WFCEDBS	V0, V0, V4
	BVS	L35
	FMOVD	F2, F1
	BR	L1
L46:
	WORD	$0xB3130040	//lcdbr	%f4,%f0
	BR	L3
L44:
	WORD	$0xB3130030	//lcdbr	%f3,%f0
	BR	L7
L35:
	FMOVD	F0, F1
	BR	L1
L26:
	FMOVD	8(R9), F1
	BR	L1
L34:
	FMOVD	8(R9), F4
L19:
	LTDBR	F6, F6
	BLEU	L47
L18:
	WFMDB	V4, V5, V1
	BR	L1
L5:
	RISBGZ	$33, $50, $63, R3, R3
	WORD	$0xC23B4000	//alfi	%r3,1073741824
	BYTE	$0x00
	BYTE	$0x00
	RLL	$24, R3, R3
	ORW	$0x45000000, R3
	BR	L2
L45:
	WFCEDBS	V0, V0, V4
	BVS	L35
	LTDBR	F0, F0
	BLEU	L48
	FMOVD	8(R9), F4
L12:
	MOVW	R2, R6
	CMPBLT	R6, $0, L19
	FMUL	F4, F1
	BR	L1
L47:
	BLT	L40
	WFCEDBS	V0, V0, V2
	BVS	L49
L16:
	MOVD	·pow_xnan<>+0(SB), R1
	LDGR	R1, F0
	WFMDB	V4, V0, V1
	BR	L1
L48:
	LGDR	F0, R3
	RISBGNZ	$32, $63, $32, R3, R1
	MOVW	R1, R6
	CMPBEQ	R6, $0, L29
	LTDBR	F2, F2
	BLTU	L50
	FMOVD	F2, F4
L14:
	MOVD	$·pow_x433<>+0(SB), R1
	FMOVD	0(R1), F7
	WFCHDBS	V4, V7, V3
	BEQ	L15
	WFADB	V7, V4, V3
	FSUB	F7, F3
	WFCEDBS	V4, V3, V3
	BEQ	L15
	LTDBR	F0, F0
	FMOVD	8(R9), F4
	BNE	L16
L13:
	LTDBR	F2, F2
	BLT	L18
L40:
	FMOVD	$0, F0
	WFMDB	V4, V0, V1
	BR	L1
L49:
	WFMDB	V0, V4, V1
	BR	L1
L29:
	FMOVD	8(R9), F4
	BR	L13
L15:
	MOVD	$·pow_x434<>+0(SB), R1
	FMOVD	0(R1), F7
	WFCHDBS	V4, V7, V3
	BEQ	L32
	WFADB	V7, V4, V3
	FSUB	F7, F3
	WFCEDBS	V4, V3, V4
	BEQ	L32
	FMOVD	0(R9), F4
L17:
	LTDBR	F0, F0
	BNE	L12
	BR	L13
L32:
	FMOVD	8(R9), F4
	BR	L17
L50:
	WORD	$0xB3130042	//lcdbr	%f4,%f2
	BR	L14
xIsOne:			// Pow(1, y) = 1 for any y
yIsOne:			// Pow(x, 1) = x for any x
posInfGeZero:	// Pow(+Inf, y) = +Inf for y > 0
	MOVD	R1, ret+16(FP)
	RET
yIsNan:			//  Pow(NaN, y) = NaN
ltZeroInt:		// Pow(x, y) = NaN for finite x < 0 and finite non-integer y
	MOVD	$NaN, R2
	MOVD	R2, ret+16(FP)
	RET
negOnePosInf:	// Pow(-1, ±Inf) = 1
negOneNegInf:
	MOVD	$PosOne, R3
	MOVD	R3, ret+16(FP)
	RET
negZeroOddInt:
	MOVD	$NegInf, R3
	MOVD	R3, ret+16(FP)
	RET
zeroNotOdd:		// Pow(±0, y) = +Inf for finite y < 0 and not an odd integer
posZeroLtZero:	// special case Pow(+0, y < 0) = +Inf
zeroNegInf:		// Pow(±0, -Inf) = +Inf
	MOVD	$PosInf, R3
	MOVD	R3, ret+16(FP)
	RET
gtOnePosInf:	//Pow(x, +Inf) = +Inf for |x| > 1
ltNegOnePosInf:
	MOVD	R2, ret+16(FP)
	RET
yIsZero:		//Pow(x, ±0) = 1 for any x
	MOVD	$PosOne, R4
	MOVD	R4, ret+16(FP)
	RET
negZeroOddIntGtZero:        // Pow(-0, y) = -0 for y an odd integer > 0
	MOVD	$NegZero, R3
	MOVD	R3, ret+16(FP)
	RET
zeroNotOddGtZero:        // Pow(±0, y) = +0 for finite y > 0 and not an odd integer
	MOVD	$0, ret+16(FP)
	RET
