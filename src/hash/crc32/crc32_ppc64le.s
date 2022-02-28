// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The vectorized implementation found below is a derived work
// from code written by Anton Blanchard <anton@au.ibm.com> found
// at https://github.com/antonblanchard/crc32-vpmsum.  The original
// is dual licensed under GPL and Apache 2.  As the copyright holder
// for the work, IBM has contributed this new work under
// the golang license.

// Changes include porting to Go assembler with modifications for
// the Go ABI for ppc64le.

#include "textflag.h"

#define POWER8_OFFSET 132

#define off16	R16
#define off32	R17
#define off48	R18
#define off64	R19
#define off80	R20
#define off96	R21
#define	off112	R22

#define const1	V24
#define const2	V25

#define byteswap	V26
#define mask_32bit	V27
#define mask_64bit	V28
#define zeroes		V29

#define MAX_SIZE	32*1024
#define REFLECT

TEXT ·ppc64SlicingUpdateBy8(SB), NOSPLIT|NOFRAME, $0-44
	MOVWZ	crc+0(FP), R3   // incoming crc
	MOVD    table8+8(FP), R4   // *Table
	MOVD    p+16(FP), R5
	MOVD    p_len+24(FP), R6 // p len

	CMP     $0,R6           // len == 0?
	BNE     start
	MOVW    R3,ret+40(FP)   // return crc
	RET

start:
	NOR     R3,R3,R7        // ^crc
	MOVWZ	R7,R7		// 32 bits
	CMP	R6,$16
	MOVD	R6,CTR
	BLT	short
	SRAD    $3,R6,R8        // 8 byte chunks
	MOVD    R8,CTR

loop:
	MOVWZ	0(R5),R8	// 0-3 bytes of p ?Endian?
	MOVWZ	4(R5),R9	// 4-7 bytes of p
	MOVD	R4,R10		// &tab[0]
	XOR	R7,R8,R7	// crc ^= byte[0:3]
	RLDICL	$40,R9,$56,R17	// p[7]
	SLD	$2,R17,R17	// p[7]*4
	RLDICL	$40,R7,$56,R8	// crc>>24
	ADD	R17,R10,R17	// &tab[0][p[7]]
	SLD	$2,R8,R8	// crc>>24*4
	RLDICL	$48,R9,$56,R18	// p[6]
	SLD	$2,R18,R18	// p[6]*4
	ADD	$1024,R10,R10	// tab[1]
	MOVWZ	0(R17),R21	// tab[0][p[7]]
	RLDICL	$56,R9,$56,R19	// p[5]
	ADD	R10,R18,R18	// &tab[1][p[6]]
	SLD	$2,R19,R19	// p[5]*4:1
	MOVWZ	0(R18),R22	// tab[1][p[6]]
	ADD	$1024,R10,R10	// tab[2]
	XOR	R21,R22,R21	// xor done R22
	ADD	R19,R10,R19	// &tab[2][p[5]]
	ANDCC	$255,R9,R20	// p[4] ??
	SLD	$2,R20,R20	// p[4]*4
	MOVWZ	0(R19),R23	// tab[2][p[5]]
	ADD	$1024,R10,R10	// &tab[3]
	ADD	R20,R10,R20	// tab[3][p[4]]
	XOR	R21,R23,R21	// xor done R23
	ADD	$1024,R10,R10	// &tab[4]
	MOVWZ	0(R20),R24	// tab[3][p[4]]
	ADD	R10,R8,R23	// &tab[4][crc>>24]
	XOR	R21,R24,R21	// xor done R24
	MOVWZ	0(R23),R25	// tab[4][crc>>24]
	RLDICL	$48,R7,$56,R24	// crc>>16&0xFF
	XOR	R21,R25,R21	// xor done R25
	ADD	$1024,R10,R10	// &tab[5]
	SLD	$2,R24,R24	// crc>>16&0xFF*4
	ADD	R24,R10,R24	// &tab[5][crc>>16&0xFF]
	MOVWZ	0(R24),R26	// tab[5][crc>>16&0xFF]
	XOR	R21,R26,R21	// xor done R26
	RLDICL	$56,R7,$56,R25	// crc>>8
	ADD	$1024,R10,R10	// &tab[6]
	SLD	$2,R25,R25	// crc>>8&FF*2
	ADD	R25,R10,R25	// &tab[6][crc>>8&0xFF]
	MOVBZ   R7,R26          // crc&0xFF
	ADD     $1024,R10,R10   // &tab[7]
	MOVWZ	0(R25),R27	// tab[6][crc>>8&0xFF]
	SLD	$2,R26,R26	// crc&0xFF*2
	XOR	R21,R27,R21	// xor done R27
	ADD	R26,R10,R26	// &tab[7][crc&0xFF]
	ADD     $8,R5           // p = p[8:]
	MOVWZ	0(R26),R28	// tab[7][crc&0xFF]
	XOR	R21,R28,R21	// xor done R28
	MOVWZ	R21,R7		// crc for next round
	BC	16,0,loop	// next 8 bytes
	ANDCC	$7,R6,R8	// any leftover bytes
	BEQ	done		// none --> done
	MOVD	R8,CTR		// byte count
        PCALIGN $16             // align short loop
short:
	MOVBZ   0(R5),R8        // get v
	MOVBZ   R7,R9           // byte(crc) -> R8 BE vs LE?
        SRD     $8,R7,R14       // crc>>8
	XOR     R8,R9,R8        // byte(crc)^v -> R8
	ADD	$1,R5		// ptr to next v
	SLD     $2,R8           // convert index-> bytes
	ADD     R8,R4,R9        // &tab[byte(crc)^v]
	MOVWZ   0(R9),R10       // tab[byte(crc)^v]
	XOR     R10,R14,R7       // loop crc in R7
	BC      16,0,short
done:
	NOR     R7,R7,R7        // ^crc
	MOVW    R7,ret+40(FP)   // return crc
	RET

#ifdef BYTESWAP_DATA
DATA ·byteswapcons+0(SB)/8,$0x0706050403020100
DATA ·byteswapcons+8(SB)/8,$0x0f0e0d0c0b0a0908

GLOBL ·byteswapcons+0(SB),RODATA,$16
#endif

TEXT ·vectorCrc32(SB), NOSPLIT|NOFRAME, $0-36
	MOVWZ	crc+0(FP), R3   // incoming crc
	MOVWZ	ctab+4(FP), R14   // crc poly id
	MOVD    p+8(FP), R4
	MOVD    p_len+16(FP), R5 // p len

	// R3 = incoming crc
	// R14 = constant table identifier
	// R5 = address of bytes
	// R6 = length of bytes

	// defines for index loads

	MOVD	$16,off16
	MOVD	$32,off32
	MOVD	$48,off48
	MOVD	$64,off64
	MOVD	$80,off80
	MOVD	$96,off96
	MOVD	$112,off112
	MOVD	$0,R15

	MOVD	R3,R10	// save initial crc

	NOR	R3,R3,R3  // ^crc
	MOVWZ	R3,R3	// 32 bits
	VXOR	zeroes,zeroes,zeroes  // clear the V reg
	VSPLTISW $-1,V0
	VSLDOI	$4,V29,V0,mask_32bit
	VSLDOI	$8,V29,V0,mask_64bit

	VXOR	V8,V8,V8
	MTVSRD	R3,VS40	// crc initial value VS40 = V8

#ifdef REFLECT
	VSLDOI	$8,zeroes,V8,V8  // or: VSLDOI V29,V8,V27,4 for top 32 bits?
#else
	VSLDOI	$4,V8,zeroes,V8
#endif

#ifdef BYTESWAP_DATA
	MOVD    $·byteswapcons(SB),R3
	LVX	(R3),byteswap
#endif

	CMPU	R5,$256		// length of bytes
	BLT	short

	RLDICR	$0,R5,$56,R6 // chunk to process

	// First step for larger sizes
l1:	MOVD	$32768,R7
	MOVD	R7,R9
	CMP	R6,R7   // compare R6, R7 (MAX SIZE)
	BGT	top	// less than MAX, just do remainder
	MOVD	R6,R7
top:
	SUB	R7,R6,R6

	// mainloop does 128 bytes at a time
	SRD	$7,R7

	// determine the offset into the constants table to start with.
	// Each constant is 128 bytes, used against 16 bytes of data.
	SLD	$4,R7,R8
	SRD	$3,R9,R9
	SUB	R8,R9,R8

	// The last iteration is reduced in a separate step
	ADD	$-1,R7
	MOVD	R7,CTR

	// Determine which constant table (depends on poly)
	CMP	R14,$1
	BNE	castTable
	MOVD	$·IEEEConst(SB),R3
	BR	startConst
castTable:
	MOVD	$·CastConst(SB),R3

startConst:
	ADD	R3,R8,R3	// starting point in constants table

	VXOR	V0,V0,V0	// clear the V regs
	VXOR	V1,V1,V1
	VXOR	V2,V2,V2
	VXOR	V3,V3,V3
	VXOR	V4,V4,V4
	VXOR	V5,V5,V5
	VXOR	V6,V6,V6
	VXOR	V7,V7,V7

	LVX	(R3),const1	// loading constant values

	CMP	R15,$1		// Identify warm up pass
	BEQ	next

	// First warm up pass: load the bytes to process
	LVX	(R4),V16
	LVX	(R4+off16),V17
	LVX	(R4+off32),V18
	LVX	(R4+off48),V19
	LVX	(R4+off64),V20
	LVX	(R4+off80),V21
	LVX	(R4+off96),V22
	LVX	(R4+off112),V23
	ADD	$128,R4		// bump up to next 128 bytes in buffer

	VXOR	V16,V8,V16	// xor in initial CRC in V8

next:
	BC	18,0,first_warm_up_done

	ADD	$16,R3		// bump up to next constants
	LVX	(R3),const2	// table values

	VPMSUMD	V16,const1,V8 // second warm up pass
	LVX	(R4),V16	// load from buffer
	OR	$0,R2,R2

	VPMSUMD	V17,const1,V9	// vpmsumd with constants
	LVX	(R4+off16),V17	// load next from buffer
	OR	$0,R2,R2

	VPMSUMD	V18,const1,V10	// vpmsumd with constants
	LVX	(R4+off32),V18	// load next from buffer
	OR	$0,R2,R2

	VPMSUMD	V19,const1,V11	// vpmsumd with constants
	LVX	(R4+off48),V19	// load next from buffer
	OR	$0,R2,R2

	VPMSUMD	V20,const1,V12	// vpmsumd with constants
	LVX	(R4+off64),V20	// load next from buffer
	OR	$0,R2,R2

	VPMSUMD	V21,const1,V13	// vpmsumd with constants
	LVX	(R4+off80),V21	// load next from buffer
	OR	$0,R2,R2

	VPMSUMD	V22,const1,V14	// vpmsumd with constants
	LVX	(R4+off96),V22	// load next from buffer
	OR	$0,R2,R2

	VPMSUMD	V23,const1,V15	// vpmsumd with constants
	LVX	(R4+off112),V23	// load next from buffer

	ADD	$128,R4		// bump up to next 128 bytes in buffer

	BC	18,0,first_cool_down

cool_top:
	LVX	(R3),const1	// constants
	ADD	$16,R3		// inc to next constants
	OR	$0,R2,R2

	VXOR	V0,V8,V0	// xor in previous vpmsumd
	VPMSUMD	V16,const2,V8	// vpmsumd with constants
	LVX	(R4),V16	// buffer
	OR	$0,R2,R2

	VXOR	V1,V9,V1	// xor in previous
	VPMSUMD	V17,const2,V9	// vpmsumd with constants
	LVX	(R4+off16),V17	// next in buffer
	OR	$0,R2,R2

	VXOR	V2,V10,V2	// xor in previous
	VPMSUMD	V18,const2,V10	// vpmsumd with constants
	LVX	(R4+off32),V18	// next in buffer
	OR	$0,R2,R2

	VXOR	V3,V11,V3	// xor in previous
	VPMSUMD	V19,const2,V11	// vpmsumd with constants
	LVX	(R4+off48),V19	// next in buffer
	LVX	(R3),const2	// get next constant
	OR	$0,R2,R2

	VXOR	V4,V12,V4	// xor in previous
	VPMSUMD	V20,const1,V12	// vpmsumd with constants
	LVX	(R4+off64),V20	// next in buffer
	OR	$0,R2,R2

	VXOR	V5,V13,V5	// xor in previous
	VPMSUMD	V21,const1,V13	// vpmsumd with constants
	LVX	(R4+off80),V21	// next in buffer
	OR	$0,R2,R2

	VXOR	V6,V14,V6	// xor in previous
	VPMSUMD	V22,const1,V14	// vpmsumd with constants
	LVX	(R4+off96),V22	// next in buffer
	OR	$0,R2,R2

	VXOR	V7,V15,V7	// xor in previous
	VPMSUMD	V23,const1,V15	// vpmsumd with constants
	LVX	(R4+off112),V23	// next in buffer

	ADD	$128,R4		// bump up buffer pointer
	BC	16,0,cool_top	// are we done?

first_cool_down:

	// load the constants
	// xor in the previous value
	// vpmsumd the result with constants

	LVX	(R3),const1
	ADD	$16,R3

	VXOR	V0,V8,V0
	VPMSUMD V16,const1,V8
	OR	$0,R2,R2

	VXOR	V1,V9,V1
	VPMSUMD	V17,const1,V9
	OR	$0,R2,R2

	VXOR	V2,V10,V2
	VPMSUMD	V18,const1,V10
	OR	$0,R2,R2

	VXOR	V3,V11,V3
	VPMSUMD	V19,const1,V11
	OR	$0,R2,R2

	VXOR	V4,V12,V4
	VPMSUMD	V20,const1,V12
	OR	$0,R2,R2

	VXOR	V5,V13,V5
	VPMSUMD	V21,const1,V13
	OR	$0,R2,R2

	VXOR	V6,V14,V6
	VPMSUMD	V22,const1,V14
	OR	$0,R2,R2

	VXOR	V7,V15,V7
	VPMSUMD	V23,const1,V15
	OR	$0,R2,R2

second_cool_down:

	VXOR    V0,V8,V0
	VXOR    V1,V9,V1
	VXOR    V2,V10,V2
	VXOR    V3,V11,V3
	VXOR    V4,V12,V4
	VXOR    V5,V13,V5
	VXOR    V6,V14,V6
	VXOR    V7,V15,V7

#ifdef REFLECT
	VSLDOI  $4,V0,zeroes,V0
	VSLDOI  $4,V1,zeroes,V1
	VSLDOI  $4,V2,zeroes,V2
	VSLDOI  $4,V3,zeroes,V3
	VSLDOI  $4,V4,zeroes,V4
	VSLDOI  $4,V5,zeroes,V5
	VSLDOI  $4,V6,zeroes,V6
	VSLDOI  $4,V7,zeroes,V7
#endif

	LVX	(R4),V8
	LVX	(R4+off16),V9
	LVX	(R4+off32),V10
	LVX	(R4+off48),V11
	LVX	(R4+off64),V12
	LVX	(R4+off80),V13
	LVX	(R4+off96),V14
	LVX	(R4+off112),V15

	ADD	$128,R4

	VXOR	V0,V8,V16
	VXOR	V1,V9,V17
	VXOR	V2,V10,V18
	VXOR	V3,V11,V19
	VXOR	V4,V12,V20
	VXOR	V5,V13,V21
	VXOR	V6,V14,V22
	VXOR	V7,V15,V23

	MOVD    $1,R15
	CMP     $0,R6
	ADD     $128,R6

	BNE	l1
	ANDCC   $127,R5
	SUBC	R5,$128,R6
	ADD	R3,R6,R3

	SRD	$4,R5,R7
	MOVD	R7,CTR
	LVX	(R3),V0
	LVX	(R3+off16),V1
	LVX	(R3+off32),V2
	LVX	(R3+off48),V3
	LVX	(R3+off64),V4
	LVX	(R3+off80),V5
	LVX	(R3+off96),V6
	LVX	(R3+off112),V7

	ADD	$128,R3

	VPMSUMW	V16,V0,V0
	VPMSUMW	V17,V1,V1
	VPMSUMW	V18,V2,V2
	VPMSUMW	V19,V3,V3
	VPMSUMW	V20,V4,V4
	VPMSUMW	V21,V5,V5
	VPMSUMW	V22,V6,V6
	VPMSUMW	V23,V7,V7

	// now reduce the tail

	CMP	$0,R7
	BEQ	next1

	LVX	(R4),V16
	LVX	(R3),V17
	VPMSUMW	V16,V17,V16
	VXOR	V0,V16,V0
	BC	18,0,next1

	LVX	(R4+off16),V16
	LVX	(R3+off16),V17
	VPMSUMW	V16,V17,V16
	VXOR	V0,V16,V0
	BC	18,0,next1

	LVX	(R4+off32),V16
	LVX	(R3+off32),V17
	VPMSUMW	V16,V17,V16
	VXOR	V0,V16,V0
	BC	18,0,next1

	LVX	(R4+off48),V16
	LVX	(R3+off48),V17
	VPMSUMW	V16,V17,V16
	VXOR	V0,V16,V0
	BC	18,0,next1

	LVX	(R4+off64),V16
	LVX	(R3+off64),V17
	VPMSUMW	V16,V17,V16
	VXOR	V0,V16,V0
	BC	18,0,next1

	LVX	(R4+off80),V16
	LVX	(R3+off80),V17
	VPMSUMW	V16,V17,V16
	VXOR	V0,V16,V0
	BC	18,0,next1

	LVX	(R4+off96),V16
	LVX	(R3+off96),V17
	VPMSUMW	V16,V17,V16
	VXOR	V0,V16,V0

next1:
	VXOR	V0,V1,V0
	VXOR	V2,V3,V2
	VXOR	V4,V5,V4
	VXOR	V6,V7,V6
	VXOR	V0,V2,V0
	VXOR	V4,V6,V4
	VXOR	V0,V4,V0

barrett_reduction:

	CMP	R14,$1
	BNE	barcstTable
	MOVD	$·IEEEBarConst(SB),R3
	BR	startbarConst
barcstTable:
	MOVD    $·CastBarConst(SB),R3

startbarConst:
	LVX	(R3),const1
	LVX	(R3+off16),const2

	VSLDOI	$8,V0,V0,V1
	VXOR	V0,V1,V0

#ifdef REFLECT
	VSPLTISB $1,V1
	VSL	V0,V1,V0
#endif

	VAND	V0,mask_64bit,V0

#ifndef	REFLECT

	VPMSUMD	V0,const1,V1
	VSLDOI	$8,zeroes,V1,V1
	VPMSUMD	V1,const2,V1
	VXOR	V0,V1,V0
	VSLDOI	$8,V0,zeroes,V0

#else

	VAND	V0,mask_32bit,V1
	VPMSUMD	V1,const1,V1
	VAND	V1,mask_32bit,V1
	VPMSUMD	V1,const2,V1
	VXOR	V0,V1,V0
	VSLDOI  $4,V0,zeroes,V0

#endif

	MFVSRD	VS32,R3 // VS32 = V0

	NOR	R3,R3,R3 // return ^crc
	MOVW	R3,ret+32(FP)
	RET

first_warm_up_done:

	LVX	(R3),const1
	ADD	$16,R3

	VPMSUMD	V16,const1,V8
	VPMSUMD	V17,const1,V9
	VPMSUMD	V18,const1,V10
	VPMSUMD	V19,const1,V11
	VPMSUMD	V20,const1,V12
	VPMSUMD	V21,const1,V13
	VPMSUMD	V22,const1,V14
	VPMSUMD	V23,const1,V15

	BR	second_cool_down

short:
	CMP	$0,R5
	BEQ	zero

	// compute short constants

	CMP     R14,$1
	BNE     castshTable
	MOVD    $·IEEEConst(SB),R3
	ADD	$4080,R3
	BR      startshConst
castshTable:
	MOVD    $·CastConst(SB),R3
	ADD	$4080,R3

startshConst:
	SUBC	R5,$256,R6	// sub from 256
	ADD	R3,R6,R3

	// calculate where to start

	SRD	$4,R5,R7
	MOVD	R7,CTR

	VXOR	V19,V19,V19
	VXOR	V20,V20,V20

	LVX	(R4),V0
	LVX	(R3),V16
	VXOR	V0,V8,V0
	VPMSUMW	V0,V16,V0
	BC	18,0,v0

	LVX	(R4+off16),V1
	LVX	(R3+off16),V17
	VPMSUMW	V1,V17,V1
	BC	18,0,v1

	LVX	(R4+off32),V2
	LVX	(R3+off32),V16
	VPMSUMW	V2,V16,V2
	BC	18,0,v2

	LVX	(R4+off48),V3
	LVX	(R3+off48),V17
	VPMSUMW	V3,V17,V3
	BC	18,0,v3

	LVX	(R4+off64),V4
	LVX	(R3+off64),V16
	VPMSUMW	V4,V16,V4
	BC	18,0,v4

	LVX	(R4+off80),V5
	LVX	(R3+off80),V17
	VPMSUMW	V5,V17,V5
	BC	18,0,v5

	LVX	(R4+off96),V6
	LVX	(R3+off96),V16
	VPMSUMW	V6,V16,V6
	BC	18,0,v6

	LVX	(R4+off112),V7
	LVX	(R3+off112),V17
	VPMSUMW	V7,V17,V7
	BC	18,0,v7

	ADD	$128,R3
	ADD	$128,R4

	LVX	(R4),V8
	LVX	(R3),V16
	VPMSUMW	V8,V16,V8
	BC	18,0,v8

	LVX	(R4+off16),V9
	LVX	(R3+off16),V17
	VPMSUMW	V9,V17,V9
	BC	18,0,v9

	LVX	(R4+off32),V10
	LVX	(R3+off32),V16
	VPMSUMW	V10,V16,V10
	BC	18,0,v10

	LVX	(R4+off48),V11
	LVX	(R3+off48),V17
	VPMSUMW	V11,V17,V11
	BC	18,0,v11

	LVX	(R4+off64),V12
	LVX	(R3+off64),V16
	VPMSUMW	V12,V16,V12
	BC	18,0,v12

	LVX	(R4+off80),V13
	LVX	(R3+off80),V17
	VPMSUMW	V13,V17,V13
	BC	18,0,v13

	LVX	(R4+off96),V14
	LVX	(R3+off96),V16
	VPMSUMW	V14,V16,V14
	BC	18,0,v14

	LVX	(R4+off112),V15
	LVX	(R3+off112),V17
	VPMSUMW	V15,V17,V15

	VXOR	V19,V15,V19
v14:	VXOR	V20,V14,V20
v13:	VXOR	V19,V13,V19
v12:	VXOR	V20,V12,V20
v11:	VXOR	V19,V11,V19
v10:	VXOR	V20,V10,V20
v9:	VXOR	V19,V9,V19
v8:	VXOR	V20,V8,V20
v7:	VXOR	V19,V7,V19
v6:	VXOR	V20,V6,V20
v5:	VXOR	V19,V5,V19
v4:	VXOR	V20,V4,V20
v3:	VXOR	V19,V3,V19
v2:	VXOR	V20,V2,V20
v1:	VXOR	V19,V1,V19
v0:	VXOR	V20,V0,V20

	VXOR	V19,V20,V0

	BR	barrett_reduction

zero:
	// This case is the original crc, so just return it
	MOVW    R10,ret+32(FP)
	RET
