// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !math_big_pure_go

#include "textflag.h"

// This file provides fast assembly versions for the elementary
// arithmetic operations on vectors implemented in arith.go.

// DI = R3, CX = R4, SI = r10, r8 = r8, r9=r9, r10 = r2, r11 = r5, r12 = r6, r13 = r7, r14 = r1 (R0 set to 0) + use R11
// func addVV(z, x, y []Word) (c Word)

TEXT ·addVV(SB), NOSPLIT, $0
	MOVD addvectorfacility+0x00(SB), R1
	BR   (R1)

TEXT ·addVV_check(SB), NOSPLIT, $0
	MOVB   ·hasVX(SB), R1
	CMPBEQ R1, $1, vectorimpl              // vectorfacility = 1, vector supported
	MOVD   $addvectorfacility+0x00(SB), R1
	MOVD   $·addVV_novec(SB), R2
	MOVD   R2, 0(R1)

	// MOVD	$·addVV_novec(SB), 0(R1)
	BR ·addVV_novec(SB)

vectorimpl:
	MOVD $addvectorfacility+0x00(SB), R1
	MOVD $·addVV_vec(SB), R2
	MOVD R2, 0(R1)

	// MOVD	$·addVV_vec(SB), 0(R1)
	BR ·addVV_vec(SB)

GLOBL addvectorfacility+0x00(SB), NOPTR, $8
DATA addvectorfacility+0x00(SB)/8, $·addVV_check(SB)

TEXT ·addVV_vec(SB), NOSPLIT, $0
	MOVD z_len+8(FP), R3
	MOVD x+24(FP), R8
	MOVD y+48(FP), R9
	MOVD z+0(FP), R2

	MOVD $0, R4  // c = 0
	MOVD $0, R0  // make sure it's zero
	MOVD $0, R10 // i = 0

	// s/JL/JMP/ below to disable the unrolled loop
	SUB $4, R3
	BLT v1
	SUB $12, R3 // n -= 16
	BLT A1      // if n < 0 goto A1

	MOVD R8, R5
	MOVD R9, R6
	MOVD R2, R7

	// n >= 0
	// regular loop body unrolled 16x
	VZERO V0 // c = 0

UU1:
	VLM  0(R5), V1, V4    // 64-bytes into V1..V8
	ADD  $64, R5
	VPDI $0x4, V1, V1, V1 // flip the doublewords to big-endian order
	VPDI $0x4, V2, V2, V2 // flip the doublewords to big-endian order

	VLM  0(R6), V9, V12      // 64-bytes into V9..V16
	ADD  $64, R6
	VPDI $0x4, V9, V9, V9    // flip the doublewords to big-endian order
	VPDI $0x4, V10, V10, V10 // flip the doublewords to big-endian order

	VACCCQ V1, V9, V0, V25
	VACQ   V1, V9, V0, V17
	VACCCQ V2, V10, V25, V26
	VACQ   V2, V10, V25, V18

	VLM 0(R5), V5, V6   // 32-bytes into V1..V8
	VLM 0(R6), V13, V14 // 32-bytes into V9..V16
	ADD $32, R5
	ADD $32, R6

	VPDI $0x4, V3, V3, V3    // flip the doublewords to big-endian order
	VPDI $0x4, V4, V4, V4    // flip the doublewords to big-endian order
	VPDI $0x4, V11, V11, V11 // flip the doublewords to big-endian order
	VPDI $0x4, V12, V12, V12 // flip the doublewords to big-endian order

	VACCCQ V3, V11, V26, V27
	VACQ   V3, V11, V26, V19
	VACCCQ V4, V12, V27, V28
	VACQ   V4, V12, V27, V20

	VLM 0(R5), V7, V8   // 32-bytes into V1..V8
	VLM 0(R6), V15, V16 // 32-bytes into V9..V16
	ADD $32, R5
	ADD $32, R6

	VPDI $0x4, V5, V5, V5    // flip the doublewords to big-endian order
	VPDI $0x4, V6, V6, V6    // flip the doublewords to big-endian order
	VPDI $0x4, V13, V13, V13 // flip the doublewords to big-endian order
	VPDI $0x4, V14, V14, V14 // flip the doublewords to big-endian order

	VACCCQ V5, V13, V28, V29
	VACQ   V5, V13, V28, V21
	VACCCQ V6, V14, V29, V30
	VACQ   V6, V14, V29, V22

	VPDI $0x4, V7, V7, V7    // flip the doublewords to big-endian order
	VPDI $0x4, V8, V8, V8    // flip the doublewords to big-endian order
	VPDI $0x4, V15, V15, V15 // flip the doublewords to big-endian order
	VPDI $0x4, V16, V16, V16 // flip the doublewords to big-endian order

	VACCCQ V7, V15, V30, V31
	VACQ   V7, V15, V30, V23
	VACCCQ V8, V16, V31, V0  // V0 has carry-over
	VACQ   V8, V16, V31, V24

	VPDI  $0x4, V17, V17, V17 // flip the doublewords to big-endian order
	VPDI  $0x4, V18, V18, V18 // flip the doublewords to big-endian order
	VPDI  $0x4, V19, V19, V19 // flip the doublewords to big-endian order
	VPDI  $0x4, V20, V20, V20 // flip the doublewords to big-endian order
	VPDI  $0x4, V21, V21, V21 // flip the doublewords to big-endian order
	VPDI  $0x4, V22, V22, V22 // flip the doublewords to big-endian order
	VPDI  $0x4, V23, V23, V23 // flip the doublewords to big-endian order
	VPDI  $0x4, V24, V24, V24 // flip the doublewords to big-endian order
	VSTM  V17, V24, 0(R7)     // 128-bytes into z
	ADD   $128, R7
	ADD   $128, R10           // i += 16
	SUB   $16, R3             // n -= 16
	BGE   UU1                 // if n >= 0 goto U1
	VLGVG $1, V0, R4          // put cf into R4
	NEG   R4, R4              // save cf

A1:
	ADD $12, R3 // n += 16

	// s/JL/JMP/ below to disable the unrolled loop
	BLT v1 // if n < 0 goto v1

U1:  // n >= 0
	// regular loop body unrolled 4x
	MOVD 0(R8)(R10*1), R5
	MOVD 8(R8)(R10*1), R6
	MOVD 16(R8)(R10*1), R7
	MOVD 24(R8)(R10*1), R1
	ADDC R4, R4             // restore CF
	MOVD 0(R9)(R10*1), R11
	ADDE R11, R5
	MOVD 8(R9)(R10*1), R11
	ADDE R11, R6
	MOVD 16(R9)(R10*1), R11
	ADDE R11, R7
	MOVD 24(R9)(R10*1), R11
	ADDE R11, R1
	MOVD R0, R4
	ADDE R4, R4             // save CF
	NEG  R4, R4
	MOVD R5, 0(R2)(R10*1)
	MOVD R6, 8(R2)(R10*1)
	MOVD R7, 16(R2)(R10*1)
	MOVD R1, 24(R2)(R10*1)

	ADD $32, R10 // i += 4
	SUB $4, R3   // n -= 4
	BGE U1       // if n >= 0 goto U1

v1:
	ADD $4, R3 // n += 4
	BLE E1     // if n <= 0 goto E1

L1:  // n > 0
	ADDC R4, R4            // restore CF
	MOVD 0(R8)(R10*1), R5
	MOVD 0(R9)(R10*1), R11
	ADDE R11, R5
	MOVD R5, 0(R2)(R10*1)
	MOVD R0, R4
	ADDE R4, R4            // save CF
	NEG  R4, R4

	ADD $8, R10 // i++
	SUB $1, R3  // n--
	BGT L1      // if n > 0 goto L1

E1:
	NEG  R4, R4
	MOVD R4, c+72(FP) // return c
	RET

TEXT ·addVV_novec(SB), NOSPLIT, $0
novec:
	MOVD z_len+8(FP), R3
	MOVD x+24(FP), R8
	MOVD y+48(FP), R9
	MOVD z+0(FP), R2

	MOVD $0, R4  // c = 0
	MOVD $0, R0  // make sure it's zero
	MOVD $0, R10 // i = 0

	// s/JL/JMP/ below to disable the unrolled loop
	SUB $4, R3 // n -= 4
	BLT v1n    // if n < 0 goto v1n

U1n:  // n >= 0
	// regular loop body unrolled 4x
	MOVD 0(R8)(R10*1), R5
	MOVD 8(R8)(R10*1), R6
	MOVD 16(R8)(R10*1), R7
	MOVD 24(R8)(R10*1), R1
	ADDC R4, R4             // restore CF
	MOVD 0(R9)(R10*1), R11
	ADDE R11, R5
	MOVD 8(R9)(R10*1), R11
	ADDE R11, R6
	MOVD 16(R9)(R10*1), R11
	ADDE R11, R7
	MOVD 24(R9)(R10*1), R11
	ADDE R11, R1
	MOVD R0, R4
	ADDE R4, R4             // save CF
	NEG  R4, R4
	MOVD R5, 0(R2)(R10*1)
	MOVD R6, 8(R2)(R10*1)
	MOVD R7, 16(R2)(R10*1)
	MOVD R1, 24(R2)(R10*1)

	ADD $32, R10 // i += 4
	SUB $4, R3   // n -= 4
	BGE U1n      // if n >= 0 goto U1n

v1n:
	ADD $4, R3 // n += 4
	BLE E1n    // if n <= 0 goto E1n

L1n:  // n > 0
	ADDC R4, R4            // restore CF
	MOVD 0(R8)(R10*1), R5
	MOVD 0(R9)(R10*1), R11
	ADDE R11, R5
	MOVD R5, 0(R2)(R10*1)
	MOVD R0, R4
	ADDE R4, R4            // save CF
	NEG  R4, R4

	ADD $8, R10 // i++
	SUB $1, R3  // n--
	BGT L1n     // if n > 0 goto L1n

E1n:
	NEG  R4, R4
	MOVD R4, c+72(FP) // return c
	RET

TEXT ·subVV(SB), NOSPLIT, $0
	MOVD subvectorfacility+0x00(SB), R1
	BR   (R1)

TEXT ·subVV_check(SB), NOSPLIT, $0
	MOVB   ·hasVX(SB), R1
	CMPBEQ R1, $1, vectorimpl              // vectorfacility = 1, vector supported
	MOVD   $subvectorfacility+0x00(SB), R1
	MOVD   $·subVV_novec(SB), R2
	MOVD   R2, 0(R1)

	// MOVD	$·subVV_novec(SB), 0(R1)
	BR ·subVV_novec(SB)

vectorimpl:
	MOVD $subvectorfacility+0x00(SB), R1
	MOVD $·subVV_vec(SB), R2
	MOVD R2, 0(R1)

	// MOVD	$·subVV_vec(SB), 0(R1)
	BR ·subVV_vec(SB)

GLOBL subvectorfacility+0x00(SB), NOPTR, $8
DATA subvectorfacility+0x00(SB)/8, $·subVV_check(SB)

// DI = R3, CX = R4, SI = r10, r8 = r8, r9=r9, r10 = r2, r11 = r5, r12 = r6, r13 = r7, r14 = r1 (R0 set to 0) + use R11
// func subVV(z, x, y []Word) (c Word)
// (same as addVV except for SUBC/SUBE instead of ADDC/ADDE and label names)
TEXT ·subVV_vec(SB), NOSPLIT, $0
	MOVD z_len+8(FP), R3
	MOVD x+24(FP), R8
	MOVD y+48(FP), R9
	MOVD z+0(FP), R2
	MOVD $0, R4          // c = 0
	MOVD $0, R0          // make sure it's zero
	MOVD $0, R10         // i = 0

	// s/JL/JMP/ below to disable the unrolled loop
	SUB $4, R3  // n -= 4
	BLT v1      // if n < 0 goto v1
	SUB $12, R3 // n -= 16
	BLT A1      // if n < 0 goto A1

	MOVD R8, R5
	MOVD R9, R6
	MOVD R2, R7

	// n >= 0
	// regular loop body unrolled 16x
	VZERO V0         // cf = 0
	MOVD  $1, R4     // for 390 subtraction cf starts as 1 (no borrow)
	VLVGG $1, R4, V0 // put carry into V0

UU1:
	VLM  0(R5), V1, V4    // 64-bytes into V1..V8
	ADD  $64, R5
	VPDI $0x4, V1, V1, V1 // flip the doublewords to big-endian order
	VPDI $0x4, V2, V2, V2 // flip the doublewords to big-endian order

	VLM  0(R6), V9, V12      // 64-bytes into V9..V16
	ADD  $64, R6
	VPDI $0x4, V9, V9, V9    // flip the doublewords to big-endian order
	VPDI $0x4, V10, V10, V10 // flip the doublewords to big-endian order

	VSBCBIQ V1, V9, V0, V25
	VSBIQ   V1, V9, V0, V17
	VSBCBIQ V2, V10, V25, V26
	VSBIQ   V2, V10, V25, V18

	VLM 0(R5), V5, V6   // 32-bytes into V1..V8
	VLM 0(R6), V13, V14 // 32-bytes into V9..V16
	ADD $32, R5
	ADD $32, R6

	VPDI $0x4, V3, V3, V3    // flip the doublewords to big-endian order
	VPDI $0x4, V4, V4, V4    // flip the doublewords to big-endian order
	VPDI $0x4, V11, V11, V11 // flip the doublewords to big-endian order
	VPDI $0x4, V12, V12, V12 // flip the doublewords to big-endian order

	VSBCBIQ V3, V11, V26, V27
	VSBIQ   V3, V11, V26, V19
	VSBCBIQ V4, V12, V27, V28
	VSBIQ   V4, V12, V27, V20

	VLM 0(R5), V7, V8   // 32-bytes into V1..V8
	VLM 0(R6), V15, V16 // 32-bytes into V9..V16
	ADD $32, R5
	ADD $32, R6

	VPDI $0x4, V5, V5, V5    // flip the doublewords to big-endian order
	VPDI $0x4, V6, V6, V6    // flip the doublewords to big-endian order
	VPDI $0x4, V13, V13, V13 // flip the doublewords to big-endian order
	VPDI $0x4, V14, V14, V14 // flip the doublewords to big-endian order

	VSBCBIQ V5, V13, V28, V29
	VSBIQ   V5, V13, V28, V21
	VSBCBIQ V6, V14, V29, V30
	VSBIQ   V6, V14, V29, V22

	VPDI $0x4, V7, V7, V7    // flip the doublewords to big-endian order
	VPDI $0x4, V8, V8, V8    // flip the doublewords to big-endian order
	VPDI $0x4, V15, V15, V15 // flip the doublewords to big-endian order
	VPDI $0x4, V16, V16, V16 // flip the doublewords to big-endian order

	VSBCBIQ V7, V15, V30, V31
	VSBIQ   V7, V15, V30, V23
	VSBCBIQ V8, V16, V31, V0  // V0 has carry-over
	VSBIQ   V8, V16, V31, V24

	VPDI  $0x4, V17, V17, V17 // flip the doublewords to big-endian order
	VPDI  $0x4, V18, V18, V18 // flip the doublewords to big-endian order
	VPDI  $0x4, V19, V19, V19 // flip the doublewords to big-endian order
	VPDI  $0x4, V20, V20, V20 // flip the doublewords to big-endian order
	VPDI  $0x4, V21, V21, V21 // flip the doublewords to big-endian order
	VPDI  $0x4, V22, V22, V22 // flip the doublewords to big-endian order
	VPDI  $0x4, V23, V23, V23 // flip the doublewords to big-endian order
	VPDI  $0x4, V24, V24, V24 // flip the doublewords to big-endian order
	VSTM  V17, V24, 0(R7)     // 128-bytes into z
	ADD   $128, R7
	ADD   $128, R10           // i += 16
	SUB   $16, R3             // n -= 16
	BGE   UU1                 // if n >= 0 goto U1
	VLGVG $1, V0, R4          // put cf into R4
	SUB   $1, R4              // save cf

A1:
	ADD $12, R3 // n += 16
	BLT v1      // if n < 0 goto v1

U1:  // n >= 0
	// regular loop body unrolled 4x
	MOVD 0(R8)(R10*1), R5
	MOVD 8(R8)(R10*1), R6
	MOVD 16(R8)(R10*1), R7
	MOVD 24(R8)(R10*1), R1
	MOVD R0, R11
	SUBC R4, R11            // restore CF
	MOVD 0(R9)(R10*1), R11
	SUBE R11, R5
	MOVD 8(R9)(R10*1), R11
	SUBE R11, R6
	MOVD 16(R9)(R10*1), R11
	SUBE R11, R7
	MOVD 24(R9)(R10*1), R11
	SUBE R11, R1
	MOVD R0, R4
	SUBE R4, R4             // save CF
	MOVD R5, 0(R2)(R10*1)
	MOVD R6, 8(R2)(R10*1)
	MOVD R7, 16(R2)(R10*1)
	MOVD R1, 24(R2)(R10*1)

	ADD $32, R10 // i += 4
	SUB $4, R3   // n -= 4
	BGE U1       // if n >= 0 goto U1n

v1:
	ADD $4, R3 // n += 4
	BLE E1     // if n <= 0 goto E1

L1:  // n > 0
	MOVD R0, R11
	SUBC R4, R11           // restore CF
	MOVD 0(R8)(R10*1), R5
	MOVD 0(R9)(R10*1), R11
	SUBE R11, R5
	MOVD R5, 0(R2)(R10*1)
	MOVD R0, R4
	SUBE R4, R4            // save CF

	ADD $8, R10 // i++
	SUB $1, R3  // n--
	BGT L1      // if n > 0 goto L1n

E1:
	NEG  R4, R4
	MOVD R4, c+72(FP) // return c
	RET

// DI = R3, CX = R4, SI = r10, r8 = r8, r9=r9, r10 = r2, r11 = r5, r12 = r6, r13 = r7, r14 = r1 (R0 set to 0) + use R11
// func subVV(z, x, y []Word) (c Word)
// (same as addVV except for SUBC/SUBE instead of ADDC/ADDE and label names)
TEXT ·subVV_novec(SB), NOSPLIT, $0
	MOVD z_len+8(FP), R3
	MOVD x+24(FP), R8
	MOVD y+48(FP), R9
	MOVD z+0(FP), R2

	MOVD $0, R4  // c = 0
	MOVD $0, R0  // make sure it's zero
	MOVD $0, R10 // i = 0

	// s/JL/JMP/ below to disable the unrolled loop
	SUB $4, R3 // n -= 4
	BLT v1     // if n < 0 goto v1

U1:  // n >= 0
	// regular loop body unrolled 4x
	MOVD 0(R8)(R10*1), R5
	MOVD 8(R8)(R10*1), R6
	MOVD 16(R8)(R10*1), R7
	MOVD 24(R8)(R10*1), R1
	MOVD R0, R11
	SUBC R4, R11            // restore CF
	MOVD 0(R9)(R10*1), R11
	SUBE R11, R5
	MOVD 8(R9)(R10*1), R11
	SUBE R11, R6
	MOVD 16(R9)(R10*1), R11
	SUBE R11, R7
	MOVD 24(R9)(R10*1), R11
	SUBE R11, R1
	MOVD R0, R4
	SUBE R4, R4             // save CF
	MOVD R5, 0(R2)(R10*1)
	MOVD R6, 8(R2)(R10*1)
	MOVD R7, 16(R2)(R10*1)
	MOVD R1, 24(R2)(R10*1)

	ADD $32, R10 // i += 4
	SUB $4, R3   // n -= 4
	BGE U1       // if n >= 0 goto U1

v1:
	ADD $4, R3 // n += 4
	BLE E1     // if n <= 0 goto E1

L1:  // n > 0
	MOVD R0, R11
	SUBC R4, R11           // restore CF
	MOVD 0(R8)(R10*1), R5
	MOVD 0(R9)(R10*1), R11
	SUBE R11, R5
	MOVD R5, 0(R2)(R10*1)
	MOVD R0, R4
	SUBE R4, R4            // save CF

	ADD $8, R10 // i++
	SUB $1, R3  // n--
	BGT L1      // if n > 0 goto L1

E1:
	NEG  R4, R4
	MOVD R4, c+72(FP) // return c
	RET

// func lshVU(z, x []Word, s uint) (c Word)
TEXT ·lshVU(SB), NOSPLIT, $0
	BR ·lshVU_g(SB)

// func rshVU(z, x []Word, s uint) (c Word)
TEXT ·rshVU(SB), NOSPLIT, $0
	BR ·rshVU_g(SB)

// CX = R4, r8 = r8, r9=r9, r10 = r2, r11 = r5, DX = r3, AX = r6, BX = R1, (R0 set to 0) + use R11 + use R7 for i
// func mulAddVWW(z, x []Word, m, a Word) (c Word)
TEXT ·mulAddVWW(SB), NOSPLIT, $0
	MOVD z+0(FP), R2
	MOVD x+24(FP), R8
	MOVD m+48(FP), R9
	MOVD a+56(FP), R4    // c = a
	MOVD z_len+8(FP), R5
	MOVD $0, R1          // i = 0
	MOVD $0, R7          // i*8 = 0
	MOVD $0, R0          // make sure it's zero
	BR   E5

L5:
	MOVD   (R8)(R1*1), R6
	MULHDU R9, R6
	ADDC   R4, R11         // add to low order bits
	ADDE   R0, R6
	MOVD   R11, (R2)(R1*1)
	MOVD   R6, R4
	ADD    $8, R1          // i*8 + 8
	ADD    $1, R7          // i++

E5:
	CMPBLT R7, R5, L5 // i < n

	MOVD R4, c+64(FP)
	RET

// func addMulVVWW(z, x, y []Word, m, a Word) (c Word)
// CX = R4, r8 = r8, r9=r9, r10 = r2, r11 = r5, AX = r11, DX = R6, r12=r12, BX = R1, (R0 set to 0) + use R11 + use R7 for i
TEXT ·addMulVVWW(SB), NOSPLIT, $0
	MOVD z+0(FP), R3
	MOVD x+24(FP), R2
	MOVD y+48(FP), R8
	MOVD m+72(FP), R9
	MOVD z_len+8(FP), R5

	MOVD $0, R1 // i*8 = 0
	MOVD $0, R7 // i = 0
	MOVD $0, R0 // make sure it's zero
	MOVD a+80(FP), R4 // c = 0

	MOVD   R5, R12
	AND    $-2, R12
	CMPBGE R5, $2, A6
	BR     E6

A6:
	MOVD   (R8)(R1*1), R6
	MULHDU R9, R6
	MOVD   (R2)(R1*1), R10
	ADDC   R10, R11        // add to low order bits
	ADDE   R0, R6
	ADDC   R4, R11
	ADDE   R0, R6
	MOVD   R6, R4
	MOVD   R11, (R3)(R1*1)

	MOVD   (8)(R8)(R1*1), R6
	MULHDU R9, R6
	MOVD   (8)(R2)(R1*1), R10
	ADDC   R10, R11           // add to low order bits
	ADDE   R0, R6
	ADDC   R4, R11
	ADDE   R0, R6
	MOVD   R6, R4
	MOVD   R11, (8)(R3)(R1*1)

	ADD $16, R1 // i*8 + 8
	ADD $2, R7  // i++

	CMPBLT R7, R12, A6
	BR     E6

L6:
	MOVD   (R8)(R1*1), R6
	MULHDU R9, R6
	MOVD   (R2)(R1*1), R10
	ADDC   R10, R11        // add to low order bits
	ADDE   R0, R6
	ADDC   R4, R11
	ADDE   R0, R6
	MOVD   R6, R4
	MOVD   R11, (R3)(R1*1)

	ADD $8, R1 // i*8 + 8
	ADD $1, R7 // i++

E6:
	CMPBLT R7, R5, L6 // i < n

	MOVD R4, c+88(FP)
	RET

