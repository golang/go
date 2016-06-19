// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !math_big_pure_go,s390x

#include "textflag.h"

// This file provides fast assembly versions for the elementary
// arithmetic operations on vectors implemented in arith.go.

TEXT ·mulWW(SB),NOSPLIT,$0
	MOVD x+0(FP), R3
	MOVD y+8(FP), R4
	MULHDU R3, R4
	MOVD R10, z1+16(FP)
	MOVD R11, z0+24(FP)
	RET

// func divWW(x1, x0, y Word) (q, r Word)
TEXT ·divWW(SB),NOSPLIT,$0
	MOVD  x1+0(FP), R10
	MOVD  x0+8(FP), R11
	MOVD  y+16(FP), R5
	WORD  $0xb98700a5 // dlgr r10,r5
	MOVD  R11, q+24(FP)
	MOVD  R10, r+32(FP)
	RET

// DI = R3, CX = R4, SI = r10, r8 = r8, r9=r9, r10 = r2 , r11 = r5, r12 = r6, r13 = r7, r14 = r1 (R0 set to 0) + use R11
// func addVV(z, x, y []Word) (c Word)
TEXT ·addVV(SB),NOSPLIT,$0
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
	ADDC R4, R4		// restore CF
	MOVD 0(R9)(R10*1), R11
	ADDE R11, R5
	MOVD 8(R9)(R10*1), R11
	ADDE R11, R6
	MOVD 16(R9)(R10*1), R11
	ADDE R11, R7
	MOVD 24(R9)(R10*1), R11
	ADDE R11, R1
	MOVD R0, R4
	ADDE R4, R4		// save CF
	NEG  R4, R4
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
	ADDC R4, R4		// restore CF
	MOVD 0(R8)(R10*1), R5
	MOVD 0(R9)(R10*1), R11
	ADDE R11, R5
	MOVD R5, 0(R2)(R10*1)
	MOVD R0, R4
	ADDE R4, R4		// save CF
	NEG  R4, R4

	ADD  $8, R10		// i++
	SUB  $1, R3		// n--
	BGT L1			// if n > 0 goto L1

E1:	NEG  R4, R4
	MOVD R4, c+72(FP)	// return c
	RET

// DI = R3, CX = R4, SI = r10, r8 = r8, r9=r9, r10 = r2 , r11 = r5, r12 = r6, r13 = r7, r14 = r1 (R0 set to 0) + use R11
// func subVV(z, x, y []Word) (c Word)
// (same as addVV except for SUBC/SUBE instead of ADDC/ADDE and label names)
TEXT ·subVV(SB),NOSPLIT,$0
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


// func addVW(z, x []Word, y Word) (c Word)
TEXT ·addVW(SB),NOSPLIT,$0
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

//DI = R3, CX = R4, SI = r10, r8 = r8, r10 = r2 , r11 = r5, r12 = r6, r13 = r7, r14 = r1 (R0 set to 0)
// func subVW(z, x []Word, y Word) (c Word)
// (same as addVW except for SUBC/SUBE instead of ADDC/ADDE and label names)
TEXT ·subVW(SB),NOSPLIT,$0
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
	MOVD z_len+8(FP), R5
	SUB  $1, R5             // n--
	BLT  X8b                // n < 0        (n <= 0)

	// n > 0
	MOVD s+48(FP), R4
	CMPBEQ	R0, R4, Z80	       //handle 0 case beq
	MOVD $64, R6
	CMPBEQ  R6, R4, Z864	       //handle 64 case beq
	MOVD z+0(FP), R2
	MOVD x+24(FP), R8
	SLD  $3, R5             // n = n*8
	SUB  R4, R6, R7
	MOVD (R8)(R5*1), R10    // w1 = x[i-1]
	SRD  R7, R10, R3
	MOVD R3, c+56(FP)

	MOVD $0, R1             // i = 0
	BR   E8

	// i < n-1
L8:	MOVD R10, R3             // w = w1
	MOVD -8(R8)(R5*1), R10   // w1 = x[i+1]

	SLD  R4,  R3             // w<<s | w1>>ŝ
	SRD  R7, R10, R6
	OR   R6, R3
	MOVD R3, (R2)(R5*1)      // z[i] = w<<s | w1>>ŝ
	SUB  $8, R5              // i--

E8:	CMPBGT R5, R0, L8        // i < n-1

	// i >= n-1
X8a:	SLD  R4, R10             // w1<<s
	MOVD R10, (R2)           // z[0] = w1<<s
	RET

X8b:	MOVD R0, c+56(FP)
	RET

Z80:	MOVD z+0(FP), R2
	MOVD x+24(FP), R8
	SLD  $3, R5             // n = n*8

	MOVD (R8), R10
	MOVD $0, R3
	MOVD R3, c+56(FP)

	MOVD $0, R1             // i = 0
	BR   E8Z

	// i < n-1
L8Z:	MOVD R10, R3
	MOVD 8(R8)(R1*1), R10

	MOVD R3, (R2)(R1*1)
	ADD  $8, R1

E8Z:	CMPBLT R1, R5, L8Z

	// i >= n-1
	MOVD R10, (R2)(R5*1)
	RET

Z864:	MOVD z+0(FP), R2
	MOVD x+24(FP), R8
	SLD  $3, R5             // n = n*8
	MOVD (R8)(R5*1), R3     // w1 = x[n-1]
	MOVD R3, c+56(FP)       // z[i] = x[n-1]

	BR   E864

	// i < n-1
L864:	MOVD -8(R8)(R5*1), R3

	MOVD R3, (R2)(R5*1)     // z[i] = x[n-1]
	SUB  $8, R5             // i--

E864:	CMPBGT R5, R0, L864     // i < n-1

	MOVD R0, (R2)           // z[n-1] = 0
	RET


// CX = R4, r8 = r8, r10 = r2 , r11 = r5, DX = r3, AX = r10 , BX = R1 , 64-count = r7 (R0 set to 0) temp = R6
// func shrVU(z, x []Word, s uint) (c Word)
TEXT ·shrVU(SB),NOSPLIT,$0
	MOVD z_len+8(FP), R5
	SUB  $1, R5             // n--
	BLT  X9b                // n < 0        (n <= 0)

	// n > 0
	MOVD s+48(FP), R4
	CMPBEQ	R0, R4, ZB0	       //handle 0 case beq
	MOVD $64, R6
	CMPBEQ  R6, R4, ZB64	       //handle 64 case beq
	MOVD z+0(FP), R2
	MOVD x+24(FP), R8
	SLD  $3, R5             // n = n*8
	SUB  R4, R6, R7
	MOVD (R8), R10          // w1 = x[0]
	SLD  R7, R10, R3
	MOVD R3, c+56(FP)

	MOVD $0, R1            // i = 0
	BR   E9

	// i < n-1
L9:	MOVD R10, R3            // w = w1
	MOVD 8(R8)(R1*1), R10   // w1 = x[i+1]

	SRD  R4,  R3            // w>>s | w1<<s
	SLD  R7, R10, R6
	OR   R6, R3
	MOVD R3, (R2)(R1*1)     // z[i] = w>>s | w1<<s
	ADD  $8, R1             // i++

E9:	CMPBLT R1, R5, L9       // i < n-1

	// i >= n-1
X9a:	SRD  R4, R10            // w1>>s
	MOVD R10, (R2)(R5*1)    // z[n-1] = w1>>s
	RET

X9b:	MOVD R0, c+56(FP)
	RET

ZB0:	MOVD z+0(FP), R2
	MOVD x+24(FP), R8
	SLD  $3, R5             // n = n*8

	MOVD (R8), R10          // w1 = x[0]
	MOVD $0, R3             // R10 << 64
	MOVD R3, c+56(FP)

	MOVD $0, R1             // i = 0
	BR   E9Z

	// i < n-1
L9Z:	MOVD R10, R3            // w = w1
	MOVD 8(R8)(R1*1), R10   // w1 = x[i+1]

	MOVD R3, (R2)(R1*1)     // z[i] = w>>s | w1<<s
	ADD  $8, R1             // i++

E9Z:	CMPBLT R1, R5, L9Z      // i < n-1

	// i >= n-1
	MOVD R10, (R2)(R5*1)    // z[n-1] = w1>>s
	RET

ZB64:	MOVD z+0(FP), R2
	MOVD x+24(FP), R8
	SLD  $3, R5             // n = n*8
	MOVD (R8), R3          // w1 = x[0]
	MOVD R3, c+56(FP)

	MOVD $0, R1            // i = 0
	BR   E964

	// i < n-1
L964:	MOVD 8(R8)(R1*1), R3   // w1 = x[i+1]

	MOVD R3, (R2)(R1*1)     // z[i] = w>>s | w1<<s
	ADD  $8, R1             // i++

E964:	CMPBLT R1, R5, L964     // i < n-1

	// i >= n-1
	MOVD  $0, R10            // w1>>s
	MOVD R10, (R2)(R5*1)    // z[n-1] = w1>>s
	RET

// CX = R4, r8 = r8, r9=r9, r10 = r2 , r11 = r5, DX = r3, AX = r6 , BX = R1 , (R0 set to 0) + use R11 + use R7 for i
// func mulAddVWW(z, x []Word, y, r Word) (c Word)
TEXT ·mulAddVWW(SB),NOSPLIT,$0
	MOVD z+0(FP), R2
	MOVD x+24(FP), R8
	MOVD y+48(FP), R9
	MOVD r+56(FP), R4	// c = r
	MOVD z_len+8(FP), R5
	MOVD $0, R1		// i = 0
	MOVD $0, R7		// i*8 = 0
	MOVD $0, R0		// make sure it's zero
	BR E5

L5:	MOVD (R8)(R1*1), R6
	MULHDU R9, R6
	ADDC R4, R11 		//add to low order bits
	ADDE R0, R6
	MOVD R11, (R2)(R1*1)
	MOVD R6, R4
	ADD  $8, R1		// i*8 + 8
	ADD  $1, R7		// i++

E5:	CMPBLT R7, R5, L5	// i < n

	MOVD R4, c+64(FP)
	RET

// func addMulVVW(z, x []Word, y Word) (c Word)
// CX = R4, r8 = r8, r9=r9, r10 = r2 , r11 = r5, AX = r11, DX = R6, r12=r12, BX = R1 , (R0 set to 0) + use R11 + use R7 for i
TEXT ·addMulVVW(SB),NOSPLIT,$0
	MOVD z+0(FP), R2
	MOVD x+24(FP), R8
	MOVD y+48(FP), R9
	MOVD z_len+8(FP), R5

	MOVD $0, R1		// i*8 = 0
	MOVD $0, R7		// i = 0
	MOVD $0, R0		// make sure it's zero
	MOVD $0, R4		// c = 0

	MOVD R5, R12
	AND  $-2, R12
	CMPBGE R5, $2, A6
	BR   E6

A6:	MOVD (R8)(R1*1), R6
	MULHDU R9, R6
	MOVD (R2)(R1*1), R10
	ADDC R10, R11	//add to low order bits
	ADDE R0, R6
	ADDC R4, R11
	ADDE R0, R6
	MOVD R6, R4
	MOVD R11, (R2)(R1*1)

	MOVD (8)(R8)(R1*1), R6
	MULHDU R9, R6
	MOVD (8)(R2)(R1*1), R10
	ADDC R10, R11	//add to low order bits
	ADDE R0, R6
	ADDC R4, R11
	ADDE R0, R6
	MOVD R6, R4
	MOVD R11, (8)(R2)(R1*1)

	ADD  $16, R1		// i*8 + 8
	ADD  $2, R7		// i++

	CMPBLT R7, R12, A6
	BR E6

L6:	MOVD (R8)(R1*1), R6
	MULHDU R9, R6
	MOVD (R2)(R1*1), R10
	ADDC R10, R11	//add to low order bits
	ADDE R0, R6
	ADDC R4, R11
	ADDE R0, R6
	MOVD R6, R4
	MOVD R11, (R2)(R1*1)

	ADD  $8, R1		// i*8 + 8
	ADD  $1, R7		// i++

E6:	CMPBLT R7, R5, L6	// i < n

	MOVD R4, c+56(FP)
	RET

// func divWVW(z []Word, xn Word, x []Word, y Word) (r Word)
// CX = R4, r8 = r8, r9=r9, r10 = r2 , r11 = r5, AX = r11, DX = R6, r12=r12, BX = R1(*8) , (R0 set to 0) + use R11 + use R7 for i
TEXT ·divWVW(SB),NOSPLIT,$0
	MOVD z+0(FP), R2
	MOVD xn+24(FP), R10	// r = xn
	MOVD x+32(FP), R8
	MOVD y+56(FP), R9
	MOVD z_len+8(FP), R7	// i = z
	SLD  $3, R7, R1		// i*8
	MOVD $0, R0		// make sure it's zero
	BR E7

L7:	MOVD (R8)(R1*1), R11
	WORD $0xB98700A9  //DLGR R10,R9
	MOVD R11, (R2)(R1*1)

E7:	SUB  $1, R7		// i--
	SUB  $8, R1
	BGE L7			// i >= 0

	MOVD R10, r+64(FP)
	RET

// func bitLen(x Word) (n int)
TEXT ·bitLen(SB),NOSPLIT,$0
	MOVD x+0(FP), R2
	WORD $0xb9830022 // FLOGR R2,R2
	MOVD $64, R3
	SUB  R2, R3
	MOVD R3, n+8(FP)
	RET
