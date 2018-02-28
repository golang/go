// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !math_big_pure_go,s390x

#include "textflag.h"

// This file provides fast assembly versions for the elementary
// arithmetic operations on vectors implemented in arith.go.

TEXT ·hasVectorFacility(SB),NOSPLIT,$24-1
        MOVD    $x-24(SP), R1
        XC      $24, 0(R1), 0(R1) // clear the storage
        MOVD    $2, R0            // R0 is the number of double words stored -1
        WORD    $0xB2B01000       // STFLE 0(R1)
        XOR     R0, R0            // reset the value of R0
        MOVBZ   z-8(SP), R1
        AND     $0x40, R1
        BEQ     novector
vectorinstalled:
        // check if the vector instruction has been enabled
        VLEIB   $0, $0xF, V16
        VLGVB   $0, V16, R1
        CMPBNE  R1, $0xF, novector
        MOVB    $1, ret+0(FP) // have vx
        RET
novector:
        MOVB    $0, ret+0(FP) // no vx
        RET

TEXT ·mulWW(SB),NOSPLIT,$0
	MOVD	x+0(FP), R3
	MOVD	y+8(FP), R4
	MULHDU	R3, R4
	MOVD	R10, z1+16(FP)
	MOVD	R11, z0+24(FP)
	RET

// func divWW(x1, x0, y Word) (q, r Word)
TEXT ·divWW(SB),NOSPLIT,$0
	MOVD	x1+0(FP), R10
	MOVD	x0+8(FP), R11
	MOVD	y+16(FP), R5
	WORD	$0xb98700a5 // dlgr r10,r5
	MOVD	R11, q+24(FP)
	MOVD	R10, r+32(FP)
	RET

// DI = R3, CX = R4, SI = r10, r8 = r8, r9=r9, r10 = r2 , r11 = r5, r12 = r6, r13 = r7, r14 = r1 (R0 set to 0) + use R11
// func addVV(z, x, y []Word) (c Word)


TEXT ·addVV(SB),NOSPLIT,$0
	MOVD	addvectorfacility+0x00(SB),R1
	BR	(R1)
	
TEXT ·addVV_check(SB),NOSPLIT, $0
	MOVB	·hasVX(SB), R1
	CMPBEQ	R1, $1, vectorimpl      // vectorfacility = 1, vector supported
	MOVD	$addvectorfacility+0x00(SB), R1
	MOVD	$·addVV_novec(SB), R2
	MOVD	R2, 0(R1)
	//MOVD	$·addVV_novec(SB), 0(R1)
	BR	·addVV_novec(SB)
vectorimpl:
	MOVD	$addvectorfacility+0x00(SB), R1
	MOVD	$·addVV_vec(SB), R2
	MOVD	R2, 0(R1)
	//MOVD	$·addVV_vec(SB), 0(R1)
	BR	·addVV_vec(SB)

GLOBL addvectorfacility+0x00(SB), NOPTR, $8
DATA addvectorfacility+0x00(SB)/8, $·addVV_check(SB)

TEXT ·addVV_vec(SB),NOSPLIT,$0
	MOVD	z_len+8(FP), R3
	MOVD	x+24(FP), R8
	MOVD	y+48(FP), R9
	MOVD	z+0(FP), R2

	MOVD	$0, R4		// c = 0
	MOVD	$0, R0		// make sure it's zero
	MOVD	$0, R10		// i = 0


	// s/JL/JMP/ below to disable the unrolled loop
	SUB	$4, R3
	BLT	v1
	SUB     $12, R3                 // n -= 16
        BLT     A1                      // if n < 0 goto A1
       
	MOVD	R8, R5
	MOVD	R9, R6
	MOVD	R2, R7
	// n >= 0
	// regular loop body unrolled 16x
	VZERO	V0			// c = 0
UU1:	VLM	0(R5), V1, V4		// 64-bytes into V1..V8
	ADD	$64, R5
	VPDI	$0x4,V1,V1,V1		// flip the doublewords to big-endian order
	VPDI	$0x4,V2,V2,V2		// flip the doublewords to big-endian order


	VLM	0(R6), V9, V12  	// 64-bytes into V9..V16
	ADD	$64, R6
	VPDI	$0x4,V9,V9,V9		// flip the doublewords to big-endian order
	VPDI	$0x4,V10,V10,V10	// flip the doublewords to big-endian order

	VACCCQ	V1, V9, V0, V25
	VACQ	V1, V9, V0, V17
	VACCCQ	V2, V10, V25, V26
	VACQ	V2, V10, V25, V18


	VLM	0(R5), V5, V6		// 32-bytes into V1..V8
	VLM	0(R6), V13, V14  	// 32-bytes into V9..V16
	ADD	$32, R5
	ADD	$32, R6

	VPDI	$0x4,V3,V3,V3		// flip the doublewords to big-endian order
	VPDI	$0x4,V4,V4,V4		// flip the doublewords to big-endian order
	VPDI	$0x4,V11,V11,V11	// flip the doublewords to big-endian order
	VPDI	$0x4,V12,V12,V12	// flip the doublewords to big-endian order

	VACCCQ	V3, V11, V26, V27
	VACQ	V3, V11, V26, V19
	VACCCQ	V4, V12, V27, V28
	VACQ	V4, V12, V27, V20

	VLM	0(R5), V7, V8		// 32-bytes into V1..V8
	VLM	0(R6), V15, V16  	// 32-bytes into V9..V16
	ADD	$32, R5
	ADD	$32, R6

	VPDI	$0x4,V5,V5,V5		// flip the doublewords to big-endian order
	VPDI	$0x4,V6,V6,V6		// flip the doublewords to big-endian order
	VPDI	$0x4,V13,V13,V13	// flip the doublewords to big-endian order
	VPDI	$0x4,V14,V14,V14	// flip the doublewords to big-endian order

	VACCCQ	V5, V13, V28, V29
	VACQ	V5, V13, V28, V21
	VACCCQ	V6, V14, V29, V30
	VACQ	V6, V14, V29, V22

	VPDI	$0x4,V7,V7,V7		// flip the doublewords to big-endian order
	VPDI	$0x4,V8,V8,V8		// flip the doublewords to big-endian order
	VPDI	$0x4,V15,V15,V15	// flip the doublewords to big-endian order
	VPDI	$0x4,V16,V16,V16	// flip the doublewords to big-endian order

	VACCCQ	V7, V15, V30, V31
	VACQ	V7, V15, V30, V23
	VACCCQ	V8, V16, V31, V0	//V0 has carry-over
	VACQ	V8, V16, V31, V24

	VPDI	$0x4,V17,V17,V17	// flip the doublewords to big-endian order
	VPDI	$0x4,V18,V18,V18	// flip the doublewords to big-endian order
	VPDI	$0x4,V19,V19,V19	// flip the doublewords to big-endian order
	VPDI	$0x4,V20,V20,V20	// flip the doublewords to big-endian order
	VPDI	$0x4,V21,V21,V21	// flip the doublewords to big-endian order
	VPDI	$0x4,V22,V22,V22	// flip the doublewords to big-endian order
	VPDI	$0x4,V23,V23,V23	// flip the doublewords to big-endian order
	VPDI	$0x4,V24,V24,V24	// flip the doublewords to big-endian order
	VSTM	V17, V24, 0(R7)  	// 128-bytes into z
	ADD	$128, R7
	ADD	$128, R10	// i += 16
	SUB	$16,  R3	// n -= 16
	BGE	UU1		// if n >= 0 goto U1
	VLGVG	$1, V0, R4	// put cf into R4
	NEG	R4, R4		// save cf

A1:	ADD	$12, R3		// n += 16


	// s/JL/JMP/ below to disable the unrolled loop
	BLT	v1		// if n < 0 goto v1

U1:	// n >= 0
	// regular loop body unrolled 4x
	MOVD	0(R8)(R10*1), R5
	MOVD	8(R8)(R10*1), R6
	MOVD	16(R8)(R10*1), R7
	MOVD	24(R8)(R10*1), R1
	ADDC	R4, R4		// restore CF
	MOVD	0(R9)(R10*1), R11
	ADDE	R11, R5
	MOVD	8(R9)(R10*1), R11
	ADDE	R11, R6
	MOVD	16(R9)(R10*1), R11
	ADDE	R11, R7
	MOVD	24(R9)(R10*1), R11
	ADDE	R11, R1
	MOVD	R0, R4
	ADDE	R4, R4		// save CF
	NEG	R4, R4
	MOVD	R5, 0(R2)(R10*1)
	MOVD	R6, 8(R2)(R10*1)
	MOVD	R7, 16(R2)(R10*1)
	MOVD	R1, 24(R2)(R10*1)


	ADD	$32, R10	// i += 4
	SUB	$4,  R3		// n -= 4
	BGE	U1		// if n >= 0 goto U1

v1:	ADD	$4, R3		// n += 4
	BLE	E1		// if n <= 0 goto E1

L1:	// n > 0
	ADDC	R4, R4		// restore CF
	MOVD	0(R8)(R10*1), R5
	MOVD	0(R9)(R10*1), R11
	ADDE	R11, R5
	MOVD	R5, 0(R2)(R10*1)
	MOVD	R0, R4
	ADDE	R4, R4		// save CF
	NEG 	R4, R4

	ADD	$8, R10		// i++
	SUB	$1, R3		// n--
	BGT	L1		// if n > 0 goto L1

E1:	NEG	R4, R4
	MOVD	R4, c+72(FP)	// return c
	RET

TEXT ·addVV_novec(SB),NOSPLIT,$0
novec:
	MOVD	z_len+8(FP), R3
	MOVD	x+24(FP), R8
	MOVD	y+48(FP), R9
	MOVD	z+0(FP), R2

	MOVD	$0, R4		// c = 0
	MOVD	$0, R0		// make sure it's zero
	MOVD	$0, R10		// i = 0

	// s/JL/JMP/ below to disable the unrolled loop
	SUB	$4, R3		// n -= 4
	BLT	v1n		// if n < 0 goto v1n
U1n:	// n >= 0
	// regular loop body unrolled 4x
	MOVD	0(R8)(R10*1), R5
	MOVD	8(R8)(R10*1), R6
	MOVD	16(R8)(R10*1), R7
	MOVD	24(R8)(R10*1), R1
	ADDC	R4, R4		// restore CF
	MOVD	0(R9)(R10*1), R11
	ADDE	R11, R5
	MOVD	8(R9)(R10*1), R11
	ADDE	R11, R6
	MOVD	16(R9)(R10*1), R11
	ADDE	R11, R7
	MOVD	24(R9)(R10*1), R11
	ADDE	R11, R1
	MOVD	R0, R4
	ADDE	R4, R4		// save CF
	NEG	R4, R4
	MOVD	R5, 0(R2)(R10*1)
	MOVD	R6, 8(R2)(R10*1)
	MOVD	R7, 16(R2)(R10*1)
	MOVD	R1, 24(R2)(R10*1)


	ADD	$32, R10	// i += 4
	SUB	$4,  R3		// n -= 4
	BGE	U1n		// if n >= 0 goto U1n

v1n:	ADD	$4, R3		// n += 4
	BLE	E1n		// if n <= 0 goto E1n

L1n:	// n > 0
	ADDC	R4, R4		// restore CF
	MOVD	0(R8)(R10*1), R5
	MOVD	0(R9)(R10*1), R11
	ADDE	R11, R5
	MOVD	R5, 0(R2)(R10*1)
	MOVD	R0, R4
	ADDE	R4, R4		// save CF
	NEG 	R4, R4

	ADD	$8, R10		// i++
	SUB	$1, R3		// n--
	BGT L1n			// if n > 0 goto L1n

E1n:	NEG	R4, R4
	MOVD	R4, c+72(FP)	// return c
	RET


TEXT ·subVV(SB),NOSPLIT,$0
	MOVD	subvectorfacility+0x00(SB),R1
	BR	(R1)
	
TEXT ·subVV_check(SB),NOSPLIT,$0
	MOVB	·hasVX(SB), R1
	CMPBEQ	R1, $1, vectorimpl      // vectorfacility = 1, vector supported
	MOVD	$subvectorfacility+0x00(SB), R1
	MOVD	$·subVV_novec(SB), R2
	MOVD	R2, 0(R1)
	//MOVD	$·subVV_novec(SB), 0(R1)
	BR	·subVV_novec(SB)
vectorimpl:
	MOVD	$subvectorfacility+0x00(SB), R1
	MOVD    $·subVV_vec(SB), R2
        MOVD    R2, 0(R1)
	//MOVD	$·subVV_vec(SB), 0(R1)
	BR	·subVV_vec(SB)

GLOBL subvectorfacility+0x00(SB), NOPTR, $8
DATA subvectorfacility+0x00(SB)/8, $·subVV_check(SB)

// DI = R3, CX = R4, SI = r10, r8 = r8, r9=r9, r10 = r2 , r11 = r5, r12 = r6, r13 = r7, r14 = r1 (R0 set to 0) + use R11
// func subVV(z, x, y []Word) (c Word)
// (same as addVV except for SUBC/SUBE instead of ADDC/ADDE and label names)
TEXT ·subVV_vec(SB),NOSPLIT,$0
	MOVD	z_len+8(FP), R3
	MOVD	x+24(FP), R8
	MOVD	y+48(FP), R9
	MOVD	z+0(FP), R2
	MOVD	$0, R4		// c = 0
	MOVD	$0, R0		// make sure it's zero
	MOVD	$0, R10		// i = 0
	
	// s/JL/JMP/ below to disable the unrolled loop
	SUB	$4, R3		// n -= 4
	BLT	v1		// if n < 0 goto v1
	SUB     $12, R3         // n -= 16
        BLT     A1              // if n < 0 goto A1

	MOVD	R8, R5
	MOVD	R9, R6
	MOVD	R2, R7

	// n >= 0
	// regular loop body unrolled 16x
	VZERO	V0		// cf = 0
	MOVD	$1, R4		// for 390 subtraction cf starts as 1 (no borrow)
	VLVGG	$1, R4, V0	//put carry into V0

UU1:	VLM	0(R5), V1, V4		// 64-bytes into V1..V8
	ADD	$64, R5
	VPDI	$0x4,V1,V1,V1		// flip the doublewords to big-endian order
	VPDI	$0x4,V2,V2,V2		// flip the doublewords to big-endian order


	VLM	0(R6), V9, V12  	// 64-bytes into V9..V16
	ADD	$64, R6
	VPDI	$0x4,V9,V9,V9		// flip the doublewords to big-endian order
	VPDI	$0x4,V10,V10,V10	// flip the doublewords to big-endian order

	VSBCBIQ	V1, V9, V0, V25
	VSBIQ	V1, V9, V0, V17
	VSBCBIQ	V2, V10, V25, V26
	VSBIQ	V2, V10, V25, V18


	VLM	0(R5), V5, V6		// 32-bytes into V1..V8
	VLM	0(R6), V13, V14  	// 32-bytes into V9..V16
	ADD	$32, R5
	ADD	$32, R6

	VPDI	$0x4,V3,V3,V3		// flip the doublewords to big-endian order
	VPDI	$0x4,V4,V4,V4		// flip the doublewords to big-endian order
	VPDI	$0x4,V11,V11,V11	// flip the doublewords to big-endian order
	VPDI	$0x4,V12,V12,V12	// flip the doublewords to big-endian order

	VSBCBIQ	V3, V11, V26, V27
	VSBIQ	V3, V11, V26, V19
	VSBCBIQ	V4, V12, V27, V28
	VSBIQ	V4, V12, V27, V20

	VLM	0(R5), V7, V8		// 32-bytes into V1..V8
	VLM	0(R6), V15, V16  	// 32-bytes into V9..V16
	ADD	$32, R5
	ADD	$32, R6

	VPDI	$0x4,V5,V5,V5		// flip the doublewords to big-endian order
	VPDI	$0x4,V6,V6,V6		// flip the doublewords to big-endian order
	VPDI	$0x4,V13,V13,V13	// flip the doublewords to big-endian order
	VPDI	$0x4,V14,V14,V14	// flip the doublewords to big-endian order

	VSBCBIQ	V5, V13, V28, V29
	VSBIQ	V5, V13, V28, V21
	VSBCBIQ	V6, V14, V29, V30
	VSBIQ	V6, V14, V29, V22

	VPDI	$0x4,V7,V7,V7		// flip the doublewords to big-endian order
	VPDI	$0x4,V8,V8,V8		// flip the doublewords to big-endian order
	VPDI	$0x4,V15,V15,V15	// flip the doublewords to big-endian order
	VPDI	$0x4,V16,V16,V16	// flip the doublewords to big-endian order

	VSBCBIQ	V7, V15, V30, V31
	VSBIQ	V7, V15, V30, V23
	VSBCBIQ	V8, V16, V31, V0	//V0 has carry-over
	VSBIQ	V8, V16, V31, V24

	VPDI	$0x4,V17,V17,V17	// flip the doublewords to big-endian order
	VPDI	$0x4,V18,V18,V18	// flip the doublewords to big-endian order
	VPDI	$0x4,V19,V19,V19	// flip the doublewords to big-endian order
	VPDI	$0x4,V20,V20,V20	// flip the doublewords to big-endian order
	VPDI	$0x4,V21,V21,V21	// flip the doublewords to big-endian order
	VPDI	$0x4,V22,V22,V22	// flip the doublewords to big-endian order
	VPDI	$0x4,V23,V23,V23	// flip the doublewords to big-endian order
	VPDI	$0x4,V24,V24,V24	// flip the doublewords to big-endian order
	VSTM	V17, V24, 0(R7)   // 128-bytes into z
	ADD	$128, R7
	ADD	$128, R10	// i += 16
	SUB	$16,  R3	// n -= 16
	BGE	UU1		// if n >= 0 goto U1
	VLGVG	$1, V0, R4	// put cf into R4
	SUB	$1, R4		// save cf

A1:	ADD	$12, R3		// n += 16
	BLT	v1		// if n < 0 goto v1
	
U1:	// n >= 0
	// regular loop body unrolled 4x
	MOVD	0(R8)(R10*1), R5
	MOVD	8(R8)(R10*1), R6
	MOVD	16(R8)(R10*1), R7
	MOVD	24(R8)(R10*1), R1
	MOVD	R0, R11
	SUBC	R4, R11		// restore CF
	MOVD	0(R9)(R10*1), R11
	SUBE	R11, R5
	MOVD	8(R9)(R10*1), R11
	SUBE	R11, R6
	MOVD	16(R9)(R10*1), R11
	SUBE	R11, R7
	MOVD	24(R9)(R10*1), R11
	SUBE	R11, R1
	MOVD	R0, R4
	SUBE	R4, R4		// save CF
	MOVD	R5, 0(R2)(R10*1)
	MOVD	R6, 8(R2)(R10*1)
	MOVD	R7, 16(R2)(R10*1)
	MOVD	R1, 24(R2)(R10*1)

	ADD	$32, R10	// i += 4
	SUB	$4,  R3		// n -= 4
	BGE	U1		// if n >= 0 goto U1n

v1:	ADD	$4, R3		// n += 4
	BLE	E1		// if n <= 0 goto E1

L1:	// n > 0
	MOVD	R0, R11
	SUBC	R4, R11		// restore CF
	MOVD	0(R8)(R10*1), R5
	MOVD	0(R9)(R10*1), R11
	SUBE	R11, R5
	MOVD	R5, 0(R2)(R10*1)
	MOVD	R0, R4
	SUBE	R4, R4		// save CF

	ADD	$8, R10		// i++
	SUB	$1, R3		// n--
	BGT	L1		// if n > 0 goto L1n

E1:	NEG	R4, R4
	MOVD	R4, c+72(FP)	// return c
	RET


// DI = R3, CX = R4, SI = r10, r8 = r8, r9=r9, r10 = r2 , r11 = r5, r12 = r6, r13 = r7, r14 = r1 (R0 set to 0) + use R11
// func subVV(z, x, y []Word) (c Word)
// (same as addVV except for SUBC/SUBE instead of ADDC/ADDE and label names)
TEXT ·subVV_novec(SB),NOSPLIT,$0
	MOVD z_len+8(FP), R3
	MOVD x+24(FP), R8
	MOVD y+48(FP), R9
	MOVD z+0(FP), R2

	MOVD $0, R4		// c = 0
	MOVD $0, R0		// make sure it's zero
	MOVD $0, R10		// i = 0

	// s/JL/JMP/ below to disable the unrolled loop
	SUB  $4, R3		// n -= 4
	BLT v1			// if n < 0 goto v1

U1:	// n >= 0
	// regular loop body unrolled 4x
	MOVD 0(R8)(R10*1), R5
	MOVD 8(R8)(R10*1), R6
	MOVD 16(R8)(R10*1), R7
	MOVD 24(R8)(R10*1), R1
	MOVD R0, R11
	SUBC R4, R11		// restore CF
	MOVD 0(R9)(R10*1), R11
	SUBE R11, R5
	MOVD 8(R9)(R10*1), R11
	SUBE R11, R6
	MOVD 16(R9)(R10*1), R11
	SUBE R11, R7
	MOVD 24(R9)(R10*1), R11
	SUBE R11, R1
	MOVD R0, R4
	SUBE R4, R4		// save CF
	MOVD R5, 0(R2)(R10*1)
	MOVD R6, 8(R2)(R10*1)
	MOVD R7, 16(R2)(R10*1)
	MOVD R1, 24(R2)(R10*1)


	ADD  $32, R10		// i += 4
	SUB  $4,  R3		// n -= 4
	BGE  U1			// if n >= 0 goto U1

v1:	ADD  $4, R3		// n += 4
	BLE E1			// if n <= 0 goto E1

L1:	// n > 0
	MOVD R0, R11
	SUBC R4, R11		// restore CF
	MOVD 0(R8)(R10*1), R5
	MOVD 0(R9)(R10*1), R11
	SUBE R11, R5
	MOVD R5, 0(R2)(R10*1)
	MOVD R0, R4
	SUBE R4, R4		// save CF

	ADD  $8, R10		// i++
	SUB  $1, R3		// n--
	BGT L1			// if n > 0 goto L1

E1:	NEG  R4, R4
	MOVD R4, c+72(FP)	// return c
	RET

TEXT ·addVW(SB),NOSPLIT,$0
	MOVD	addwvectorfacility+0x00(SB),R1
	BR	(R1)
	
TEXT ·addVW_check(SB),NOSPLIT,$0
	MOVB	·hasVX(SB), R1
	CMPBEQ	R1, $1, vectorimpl      // vectorfacility = 1, vector supported
	MOVD	$addwvectorfacility+0x00(SB), R1
	MOVD    $·addVW_novec(SB), R2
        MOVD    R2, 0(R1)
	//MOVD	$·addVW_novec(SB), 0(R1)
	BR	·addVW_novec(SB)
vectorimpl:
	MOVD	$addwvectorfacility+0x00(SB), R1
	MOVD    $·addVW_vec(SB), R2
        MOVD    R2, 0(R1)
	//MOVD	$·addVW_vec(SB), 0(R1)
	BR	·addVW_vec(SB)

GLOBL addwvectorfacility+0x00(SB), NOPTR, $8
DATA addwvectorfacility+0x00(SB)/8, $·addVW_check(SB)


// func addVW_vec(z, x []Word, y Word) (c Word)
TEXT ·addVW_vec(SB),NOSPLIT,$0
	MOVD	z_len+8(FP), R3
	MOVD	x+24(FP), R8
	MOVD	y+48(FP), R4	// c = y
	MOVD	z+0(FP), R2

	MOVD	$0, R0		// make sure it's zero
	MOVD	$0, R10		// i = 0
	MOVD	R8, R5
	MOVD	R2, R7

	// s/JL/JMP/ below to disable the unrolled loop
	SUB	$4, R3			// n -= 4
	BLT	v10			// if n < 0 goto v10
	SUB	$12, R3
	BLT	A10

	// n >= 0
	// regular loop body unrolled 16x

	VZERO	V0			// prepare V0 to be final carry register
	VZERO	V9			// to ensure upper half is zero
	VLVGG	$1, R4, V9
UU1:	VLM	0(R5), V1, V4		// 64-bytes into V1..V4
	ADD	$64, R5
	VPDI	$0x4,V1,V1,V1		// flip the doublewords to big-endian order
	VPDI	$0x4,V2,V2,V2		// flip the doublewords to big-endian order


	VACCCQ	V1, V9, V0, V25
	VACQ	V1, V9, V0, V17
	VZERO	V9
	VACCCQ	V2, V9, V25, V26
	VACQ	V2, V9, V25, V18


	VLM	0(R5), V5, V6		// 32-bytes into V5..V6
	ADD	$32, R5

	VPDI	$0x4,V3,V3,V3		// flip the doublewords to big-endian order
	VPDI	$0x4,V4,V4,V4		// flip the doublewords to big-endian order

	VACCCQ	V3, V9, V26, V27
	VACQ	V3, V9, V26, V19
	VACCCQ	V4, V9, V27, V28
	VACQ	V4, V9, V27, V20

	VLM	0(R5), V7, V8		// 32-bytes into V7..V8
	ADD	$32, R5

	VPDI	$0x4,V5,V5,V5		// flip the doublewords to big-endian order
	VPDI	$0x4,V6,V6,V6		// flip the doublewords to big-endian order

	VACCCQ	V5, V9, V28, V29
	VACQ	V5, V9, V28, V21
	VACCCQ	V6, V9, V29, V30
	VACQ	V6, V9, V29, V22

	VPDI	$0x4,V7,V7,V7		// flip the doublewords to big-endian order
	VPDI	$0x4,V8,V8,V8		// flip the doublewords to big-endian order

	VACCCQ	V7, V9, V30, V31
	VACQ	V7, V9, V30, V23
	VACCCQ	V8, V9, V31, V0	//V0 has carry-over
	VACQ	V8, V9, V31, V24

	VPDI	$0x4,V17,V17,V17	// flip the doublewords to big-endian order
	VPDI	$0x4,V18,V18,V18	// flip the doublewords to big-endian order
	VPDI	$0x4,V19,V19,V19	// flip the doublewords to big-endian order
	VPDI	$0x4,V20,V20,V20	// flip the doublewords to big-endian order
	VPDI	$0x4,V21,V21,V21	// flip the doublewords to big-endian order
	VPDI	$0x4,V22,V22,V22	// flip the doublewords to big-endian order
	VPDI	$0x4,V23,V23,V23	// flip the doublewords to big-endian order
	VPDI	$0x4,V24,V24,V24	// flip the doublewords to big-endian order
	VSTM	V17, V24, 0(R7)   	// 128-bytes into z
	ADD	$128, R7
	ADD	$128, R10		// i += 16
	SUB	$16,  R3		// n -= 16
	BGE	UU1		// if n >= 0 goto U1
	VLGVG	$1, V0, R4	// put cf into R4 in case we branch to v10

A10:	ADD	$12, R3		// n += 16


	// s/JL/JMP/ below to disable the unrolled loop

	BLT	v10		// if n < 0 goto v10


U4:	// n >= 0
	// regular loop body unrolled 4x
	MOVD 0(R8)(R10*1), R5
	MOVD 8(R8)(R10*1), R6
	MOVD 16(R8)(R10*1), R7
	MOVD 24(R8)(R10*1), R1
	ADDC R4, R5
	ADDE R0, R6
	ADDE R0, R7
	ADDE R0, R1
	ADDE R0, R0
	MOVD R0, R4		// save CF
	SUB  R0, R0
	MOVD R5, 0(R2)(R10*1)
	MOVD R6, 8(R2)(R10*1)
	MOVD R7, 16(R2)(R10*1)
	MOVD R1, 24(R2)(R10*1)

	ADD $32, R10		// i += 4 -> i +=32
	SUB $4, R3		// n -= 4
	BGE U4			// if n >= 0 goto U4

v10:	ADD $4, R3		// n += 4
	BLE E10			// if n <= 0 goto E4


L4:	// n > 0
	MOVD	0(R8)(R10*1), R5
	ADDC	R4, R5
	ADDE	R0, R0
	MOVD	R0, R4		// save CF
	SUB 	R0, R0
	MOVD	R5, 0(R2)(R10*1)

	ADD	$8, R10		// i++
	SUB	$1, R3		// n--
	BGT	L4		// if n > 0 goto L4

E10:	MOVD	R4, c+56(FP)	// return c

	RET


TEXT ·addVW_novec(SB),NOSPLIT,$0
//DI = R3, CX = R4, SI = r10, r8 = r8, r10 = r2 , r11 = r5, r12 = r6, r13 = r7, r14 = r1 (R0 set to 0)
	MOVD z_len+8(FP), R3
	MOVD x+24(FP), R8
	MOVD y+48(FP), R4	// c = y
	MOVD z+0(FP), R2
	MOVD $0, R0		// make sure it's 0
	MOVD $0, R10		// i = 0

	// s/JL/JMP/ below to disable the unrolled loop
	SUB $4, R3		// n -= 4
	BLT v4			// if n < 4 goto v4

U4:	// n >= 0
	// regular loop body unrolled 4x
	MOVD 0(R8)(R10*1), R5
	MOVD 8(R8)(R10*1), R6
	MOVD 16(R8)(R10*1), R7
	MOVD 24(R8)(R10*1), R1
	ADDC R4, R5
	ADDE R0, R6
	ADDE R0, R7
	ADDE R0, R1
	ADDE R0, R0
	MOVD R0, R4		// save CF
	SUB  R0, R0
	MOVD R5, 0(R2)(R10*1)
	MOVD R6, 8(R2)(R10*1)
	MOVD R7, 16(R2)(R10*1)
	MOVD R1, 24(R2)(R10*1)

	ADD $32, R10		// i += 4 -> i +=32
	SUB $4, R3		// n -= 4
	BGE U4			// if n >= 0 goto U4

v4:	ADD $4, R3		// n += 4
	BLE E4			// if n <= 0 goto E4

L4:	// n > 0
	MOVD 0(R8)(R10*1), R5
	ADDC R4, R5
	ADDE R0, R0
	MOVD R0, R4		// save CF
	SUB  R0, R0
	MOVD R5, 0(R2)(R10*1)

	ADD  $8, R10		// i++
	SUB  $1, R3		// n--
	BGT L4			// if n > 0 goto L4

E4:	MOVD R4, c+56(FP)	// return c

	RET

TEXT ·subVW(SB),NOSPLIT,$0
	MOVD	subwvectorfacility+0x00(SB),R1
	BR	(R1)
	
TEXT ·subVW_check(SB),NOSPLIT,$0
	MOVB	·hasVX(SB), R1
	CMPBEQ	R1, $1, vectorimpl      // vectorfacility = 1, vector supported
	MOVD	$subwvectorfacility+0x00(SB), R1
	MOVD    $·subVW_novec(SB), R2
        MOVD    R2, 0(R1)
	//MOVD	$·subVW_novec(SB), 0(R1)
	BR	·subVW_novec(SB)
vectorimpl:
	MOVD	$subwvectorfacility+0x00(SB), R1
	MOVD    $·subVW_vec(SB), R2
        MOVD    R2, 0(R1)
	//MOVD	$·subVW_vec(SB), 0(R1)
	BR	·subVW_vec(SB)

GLOBL subwvectorfacility+0x00(SB), NOPTR, $8
DATA subwvectorfacility+0x00(SB)/8, $·subVW_check(SB)

// func subVW(z, x []Word, y Word) (c Word)
TEXT ·subVW_vec(SB),NOSPLIT,$0
	MOVD	z_len+8(FP), R3
	MOVD	x+24(FP), R8
	MOVD	y+48(FP), R4	// c = y
	MOVD	z+0(FP), R2

	MOVD	$0, R0		// make sure it's zero
	MOVD	$0, R10		// i = 0
	MOVD	R8, R5
	MOVD	R2, R7

	// s/JL/JMP/ below to disable the unrolled loop
	SUB	$4, R3			// n -= 4
	BLT	v11			// if n < 0 goto v11
	SUB	$12, R3
	BLT	A11

	VZERO	V0
	MOVD	$1, R6			// prepare V0 to be final carry register
	VLVGG	$1, R6, V0		// borrow is initially "no borrow"
	VZERO	V9			// to ensure upper half is zero
	VLVGG	$1, R4, V9

	// n >= 0
	// regular loop body unrolled 16x


UU1:	VLM	0(R5), V1, V4		// 64-bytes into V1..V4
	ADD	$64, R5
	VPDI	$0x4,V1,V1,V1		// flip the doublewords to big-endian order
	VPDI	$0x4,V2,V2,V2		// flip the doublewords to big-endian order


	VSBCBIQ	V1, V9, V0, V25
	VSBIQ	V1, V9, V0, V17
	VZERO	V9
	VSBCBIQ	V2, V9, V25, V26
	VSBIQ	V2, V9, V25, V18

	VLM	0(R5), V5, V6		// 32-bytes into V5..V6
	ADD	$32, R5

	VPDI	$0x4,V3,V3,V3		// flip the doublewords to big-endian order
	VPDI	$0x4,V4,V4,V4		// flip the doublewords to big-endian order


	VSBCBIQ	V3, V9, V26, V27
	VSBIQ	V3, V9, V26, V19
	VSBCBIQ	V4, V9, V27, V28
	VSBIQ	V4, V9, V27, V20

	VLM	0(R5), V7, V8		// 32-bytes into V7..V8
	ADD	$32, R5

	VPDI	$0x4,V5,V5,V5		// flip the doublewords to big-endian order
	VPDI	$0x4,V6,V6,V6		// flip the doublewords to big-endian order

	VSBCBIQ	V5, V9, V28, V29
	VSBIQ	V5, V9, V28, V21
	VSBCBIQ	V6, V9, V29, V30
	VSBIQ	V6, V9, V29, V22

	VPDI	$0x4,V7,V7,V7		// flip the doublewords to big-endian order
	VPDI	$0x4,V8,V8,V8		// flip the doublewords to big-endian order

	VSBCBIQ	V7, V9, V30, V31
	VSBIQ	V7, V9, V30, V23
	VSBCBIQ	V8, V9, V31, V0	// V0 has carry-over
	VSBIQ	V8, V9, V31, V24

	VPDI	$0x4,V17,V17,V17	// flip the doublewords to big-endian order
	VPDI	$0x4,V18,V18,V18	// flip the doublewords to big-endian order
	VPDI	$0x4,V19,V19,V19	// flip the doublewords to big-endian order
	VPDI	$0x4,V20,V20,V20	// flip the doublewords to big-endian order
	VPDI	$0x4,V21,V21,V21	// flip the doublewords to big-endian order
	VPDI	$0x4,V22,V22,V22	// flip the doublewords to big-endian order
	VPDI	$0x4,V23,V23,V23	// flip the doublewords to big-endian order
	VPDI	$0x4,V24,V24,V24	// flip the doublewords to big-endian order
	VSTM	V17, V24, 0(R7)   	// 128-bytes into z
	ADD	$128, R7
	ADD	$128, R10		// i += 16
	SUB	$16,  R3		// n -= 16
	BGE	UU1			// if n >= 0 goto U1
	VLGVG	$1, V0, R4		// put cf into R4 in case we branch to v10
	SUB	$1, R4			// save cf
	NEG	R4, R4
A11:	ADD	$12, R3			// n += 16

	BLT	v11			// if n < 0 goto v11

	// n >= 0
	// regular loop body unrolled 4x

U4:	// n >= 0
	// regular loop body unrolled 4x
	MOVD 0(R8)(R10*1), R5
	MOVD 8(R8)(R10*1), R6
	MOVD 16(R8)(R10*1), R7
	MOVD 24(R8)(R10*1), R1
	SUBC R4, R5 //SLGR  -> SUBC
	SUBE R0, R6 //SLBGR -> SUBE
	SUBE R0, R7
	SUBE R0, R1
	SUBE R4, R4		// save CF
	NEG  R4, R4
	MOVD R5, 0(R2)(R10*1)
	MOVD R6, 8(R2)(R10*1)
	MOVD R7, 16(R2)(R10*1)
	MOVD R1, 24(R2)(R10*1)

	ADD $32, R10		// i += 4 -> i +=32
	SUB $4, R3		// n -= 4
	BGE U4			// if n >= 0 goto U4

v11:	ADD $4, R3		// n += 4
	BLE E11			// if n <= 0 goto E4

L4:	// n > 0

	MOVD	0(R8)(R10*1), R5
	SUBC	R4, R5
	SUBE	R4, R4		// save CF
	NEG	R4, R4
	MOVD	R5, 0(R2)(R10*1)

	ADD	$8, R10		// i++
	SUB	$1, R3		// n--
	BGT	L4		// if n > 0 goto L4

E11:	MOVD	R4, c+56(FP)	// return c

	RET

//DI = R3, CX = R4, SI = r10, r8 = r8, r10 = r2 , r11 = r5, r12 = r6, r13 = r7, r14 = r1 (R0 set to 0)
// func subVW(z, x []Word, y Word) (c Word)
// (same as addVW except for SUBC/SUBE instead of ADDC/ADDE and label names)
TEXT ·subVW_novec(SB),NOSPLIT,$0
	MOVD z_len+8(FP), R3
	MOVD x+24(FP), R8
	MOVD y+48(FP), R4	// c = y
	MOVD z+0(FP), R2
	MOVD $0, R0		// make sure it's 0
	MOVD $0, R10		// i = 0

	// s/JL/JMP/ below to disable the unrolled loop
	SUB $4, R3		// n -= 4
	BLT v4			// if n < 4 goto v4

U4:	// n >= 0
	// regular loop body unrolled 4x
	MOVD 0(R8)(R10*1), R5
	MOVD 8(R8)(R10*1), R6
	MOVD 16(R8)(R10*1), R7
	MOVD 24(R8)(R10*1), R1
	SUBC R4, R5 //SLGR  -> SUBC
	SUBE R0, R6 //SLBGR -> SUBE
	SUBE R0, R7
	SUBE R0, R1
	SUBE R4, R4		// save CF
	NEG  R4, R4
	MOVD R5, 0(R2)(R10*1)
	MOVD R6, 8(R2)(R10*1)
	MOVD R7, 16(R2)(R10*1)
	MOVD R1, 24(R2)(R10*1)

	ADD $32, R10		// i += 4 -> i +=32
	SUB $4, R3		// n -= 4
	BGE U4			// if n >= 0 goto U4

v4:	ADD $4, R3		// n += 4
	BLE E4			// if n <= 0 goto E4

L4:	// n > 0
	MOVD 0(R8)(R10*1), R5
	SUBC R4, R5
	SUBE R4, R4		// save CF
	NEG  R4, R4
	MOVD R5, 0(R2)(R10*1)

	ADD  $8, R10		// i++
	SUB  $1, R3		// n--
	BGT L4			// if n > 0 goto L4

E4:	MOVD R4, c+56(FP)	// return c

	RET

// func shlVU(z, x []Word, s uint) (c Word)
TEXT ·shlVU(SB),NOSPLIT,$0
	MOVD	z_len+8(FP), R5
	MOVD	$0, R0
	SUB	$1, R5             // n--
	BLT	X8b                // n < 0        (n <= 0)

	// n > 0
	MOVD	s+48(FP), R4
	CMPBEQ	R0, R4, Z80	   //handle 0 case beq
	MOVD	$64, R6
	CMPBEQ	R6, R4, Z864	   //handle 64 case beq
	MOVD	z+0(FP), R2
	MOVD	x+24(FP), R8
	SLD	$3, R5             // n = n*8
	SUB	R4, R6, R7
	MOVD	(R8)(R5*1), R10    // w1 = x[i-1]
	SRD	R7, R10, R3
	MOVD	R3, c+56(FP)

	MOVD	$0, R1             // i = 0
	BR	E8

	// i < n-1
L8:	MOVD	R10, R3             // w = w1
	MOVD	-8(R8)(R5*1), R10   // w1 = x[i+1]

	SLD	R4,  R3             // w<<s | w1>>ŝ
	SRD	R7, R10, R6
	OR 	R6, R3
	MOVD	R3, (R2)(R5*1)      // z[i] = w<<s | w1>>ŝ
	SUB	$8, R5              // i--

E8:	CMPBGT	R5, R0, L8	    // i < n-1

	// i >= n-1
X8a:	SLD	R4, R10             // w1<<s
	MOVD	R10, (R2)           // z[0] = w1<<s
	RET

X8b:	MOVD	R0, c+56(FP)
	RET

Z80:	MOVD	z+0(FP), R2
	MOVD	x+24(FP), R8
	SLD	$3, R5             // n = n*8

	MOVD	(R8), R10
	MOVD	$0, R3
	MOVD	R3, c+56(FP)

	MOVD	$0, R1             // i = 0
	BR	E8Z

	// i < n-1
L8Z:	MOVD	R10, R3
	MOVD	8(R8)(R1*1), R10

	MOVD	R3, (R2)(R1*1)
	ADD 	$8, R1

E8Z:	CMPBLT	R1, R5, L8Z

	// i >= n-1
	MOVD	R10, (R2)(R5*1)
	RET

Z864:	MOVD	z+0(FP), R2
	MOVD	x+24(FP), R8
	SLD	$3, R5             // n = n*8
	MOVD	(R8)(R5*1), R3     // w1 = x[n-1]
	MOVD	R3, c+56(FP)       // z[i] = x[n-1]

	BR	E864

	// i < n-1
L864:	MOVD	-8(R8)(R5*1), R3

	MOVD	R3, (R2)(R5*1)     // z[i] = x[n-1]
	SUB	$8, R5             // i--

E864:	CMPBGT	R5, R0, L864       // i < n-1

	MOVD	R0, (R2)           // z[n-1] = 0
	RET


// CX = R4, r8 = r8, r10 = r2 , r11 = r5, DX = r3, AX = r10 , BX = R1 , 64-count = r7 (R0 set to 0) temp = R6
// func shrVU(z, x []Word, s uint) (c Word)
TEXT ·shrVU(SB),NOSPLIT,$0
	MOVD	z_len+8(FP), R5
	MOVD	$0, R0
	SUB	$1, R5             // n--
	BLT	X9b                // n < 0        (n <= 0)

	// n > 0
	MOVD	s+48(FP), R4
	CMPBEQ	R0, R4, ZB0	//handle 0 case beq
	MOVD	$64, R6
	CMPBEQ 	R6, R4, ZB64	//handle 64 case beq
	MOVD	z+0(FP), R2
	MOVD	x+24(FP), R8
	SLD	$3, R5		// n = n*8
	SUB	R4, R6, R7
	MOVD	(R8), R10	// w1 = x[0]
	SLD	R7, R10, R3
	MOVD	R3, c+56(FP)

	MOVD	$0, R1		// i = 0
	BR 	E9

	// i < n-1
L9:	MOVD	R10, R3		// w = w1
	MOVD	8(R8)(R1*1), R10	// w1 = x[i+1]

	SRD	R4,  R3		// w>>s | w1<<s
	SLD	R7, R10, R6
	OR	R6, R3
	MOVD	R3, (R2)(R1*1)	// z[i] = w>>s | w1<<s
	ADD	$8, R1		// i++

E9:	CMPBLT	R1, R5, L9	// i < n-1

	// i >= n-1
X9a:	SRD	R4, R10		// w1>>s
	MOVD	R10, (R2)(R5*1)	// z[n-1] = w1>>s
	RET

X9b:	MOVD	R0, c+56(FP)
	RET

ZB0:	MOVD	z+0(FP), R2
	MOVD	x+24(FP), R8
	SLD	$3, R5		// n = n*8

	MOVD	(R8), R10	// w1 = x[0]
	MOVD	$0, R3		// R10 << 64
	MOVD	R3, c+56(FP)

	MOVD	$0, R1		// i = 0
	BR	E9Z

	// i < n-1
L9Z:	MOVD	R10, R3		// w = w1
	MOVD	8(R8)(R1*1), R10	// w1 = x[i+1]

	MOVD	R3, (R2)(R1*1)	// z[i] = w>>s | w1<<s
	ADD	$8, R1		// i++

E9Z:	CMPBLT	R1, R5, L9Z	// i < n-1

	// i >= n-1
	MOVD	R10, (R2)(R5*1)	// z[n-1] = w1>>s
	RET

ZB64:	MOVD	z+0(FP), R2
	MOVD	x+24(FP), R8
	SLD	$3, R5		// n = n*8
	MOVD	(R8), R3	// w1 = x[0]
	MOVD	R3, c+56(FP)

	MOVD	$0, R1		// i = 0
	BR	E964

	// i < n-1
L964:	MOVD	8(R8)(R1*1), R3	// w1 = x[i+1]

	MOVD	R3, (R2)(R1*1)	// z[i] = w>>s | w1<<s
	ADD	$8, R1		// i++

E964:	CMPBLT	R1, R5, L964	// i < n-1

	// i >= n-1
	MOVD	$0, R10            // w1>>s
	MOVD	R10, (R2)(R5*1)    // z[n-1] = w1>>s
	RET

// CX = R4, r8 = r8, r9=r9, r10 = r2 , r11 = r5, DX = r3, AX = r6 , BX = R1 , (R0 set to 0) + use R11 + use R7 for i
// func mulAddVWW(z, x []Word, y, r Word) (c Word)
TEXT ·mulAddVWW(SB),NOSPLIT,$0
	MOVD	z+0(FP), R2
	MOVD	x+24(FP), R8
	MOVD	y+48(FP), R9
	MOVD	r+56(FP), R4	// c = r
	MOVD	z_len+8(FP), R5
	MOVD	$0, R1		// i = 0
	MOVD	$0, R7		// i*8 = 0
	MOVD	$0, R0		// make sure it's zero
	BR	E5

L5:	MOVD	(R8)(R1*1), R6
	MULHDU	R9, R6
	ADDC	R4, R11 	//add to low order bits
	ADDE	R0, R6
	MOVD	R11, (R2)(R1*1)
	MOVD	R6, R4
	ADD	$8, R1		// i*8 + 8
	ADD	$1, R7		// i++

E5:	CMPBLT	R7, R5, L5	// i < n

	MOVD	R4, c+64(FP)
	RET

// func addMulVVW(z, x []Word, y Word) (c Word)
// CX = R4, r8 = r8, r9=r9, r10 = r2 , r11 = r5, AX = r11, DX = R6, r12=r12, BX = R1 , (R0 set to 0) + use R11 + use R7 for i
TEXT ·addMulVVW(SB),NOSPLIT,$0
	MOVD	z+0(FP), R2
	MOVD	x+24(FP), R8
	MOVD	y+48(FP), R9
	MOVD	z_len+8(FP), R5

	MOVD	$0, R1		// i*8 = 0
	MOVD	$0, R7		// i = 0
	MOVD	$0, R0		// make sure it's zero
	MOVD	$0, R4		// c = 0

	MOVD	R5, R12
	AND	$-2, R12
	CMPBGE	R5, $2, A6
	BR	E6

A6:	MOVD	(R8)(R1*1), R6
	MULHDU	R9, R6
	MOVD	(R2)(R1*1), R10
	ADDC	R10, R11	//add to low order bits
	ADDE	R0, R6
	ADDC	R4, R11
	ADDE	R0, R6
	MOVD	R6, R4
	MOVD	R11, (R2)(R1*1)

	MOVD	(8)(R8)(R1*1), R6
	MULHDU	R9, R6
	MOVD	(8)(R2)(R1*1), R10
	ADDC	R10, R11	//add to low order bits
	ADDE	R0, R6
	ADDC	R4, R11
	ADDE	R0, R6
	MOVD	R6, R4
	MOVD	R11, (8)(R2)(R1*1)

	ADD	$16, R1		// i*8 + 8
	ADD	$2, R7		// i++

	CMPBLT	R7, R12, A6
	BR	E6

L6:	MOVD	(R8)(R1*1), R6
	MULHDU	R9, R6
	MOVD	(R2)(R1*1), R10
	ADDC	R10, R11	//add to low order bits
	ADDE	R0, R6
	ADDC	R4, R11
	ADDE	R0, R6
	MOVD	R6, R4
	MOVD	R11, (R2)(R1*1)

	ADD	$8, R1		// i*8 + 8
	ADD	$1, R7		// i++

E6:	CMPBLT	R7, R5, L6	// i < n

	MOVD	R4, c+56(FP)
	RET

// func divWVW(z []Word, xn Word, x []Word, y Word) (r Word)
// CX = R4, r8 = r8, r9=r9, r10 = r2 , r11 = r5, AX = r11, DX = R6, r12=r12, BX = R1(*8) , (R0 set to 0) + use R11 + use R7 for i
TEXT ·divWVW(SB),NOSPLIT,$0
	MOVD	z+0(FP), R2
	MOVD	xn+24(FP), R10	// r = xn
	MOVD	x+32(FP), R8
	MOVD	y+56(FP), R9
	MOVD	z_len+8(FP), R7	// i = z
	SLD	$3, R7, R1		// i*8
	MOVD	$0, R0		// make sure it's zero
	BR	E7

L7:	MOVD	(R8)(R1*1), R11
	WORD	$0xB98700A9	//DLGR R10,R9
	MOVD	R11, (R2)(R1*1)

E7:	SUB	$1, R7		// i--
	SUB	$8, R1
	BGE	L7		// i >= 0

	MOVD	R10, r+64(FP)
	RET
