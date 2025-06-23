// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

#include "textflag.h"

// func cryptBlocksGCM(fn code, key, dst, src, buf []byte, cnt *[16]byte)
TEXT ·cryptBlocksGCM(SB),NOSPLIT,$0-112
       MOVD    src_len+64(FP), R0
       MOVD    buf_base+80(FP), R1
       MOVD    cnt+104(FP), R12
       LMG     (R12), R2, R3

       // Check that the src size is less than or equal to the buffer size.
       MOVD    buf_len+88(FP), R4
       CMP     R0, R4
       BGT     crash

       // Check that the src size is a multiple of 16-bytes.
       MOVD    R0, R4
       AND     $0xf, R4
       BLT     crash // non-zero

       // Check that the src size is less than or equal to the dst size.
       MOVD    dst_len+40(FP), R4
       CMP     R0, R4
       BGT     crash

       MOVD    R2, R4
       MOVD    R2, R6
       MOVD    R2, R8
       MOVD    R3, R5
       MOVD    R3, R7
       MOVD    R3, R9
       ADDW    $1, R5
       ADDW    $2, R7
       ADDW    $3, R9
incr:
       CMP     R0, $64
       BLT     tail
       STMG    R2, R9, (R1)
       ADDW    $4, R3
       ADDW    $4, R5
       ADDW    $4, R7
       ADDW    $4, R9
       MOVD    $64(R1), R1
       SUB     $64, R0
       BR      incr
tail:
       CMP     R0, $0
       BEQ     crypt
       STMG    R2, R3, (R1)
       ADDW    $1, R3
       MOVD    $16(R1), R1
       SUB     $16, R0
       BR      tail
crypt:
       STMG    R2, R3, (R12)       // update next counter value
       MOVD    fn+0(FP), R0        // function code (encryption)
       MOVD    key_base+8(FP), R1  // key
       MOVD    buf_base+80(FP), R2 // counter values
       MOVD    dst_base+32(FP), R4 // dst
       MOVD    src_base+56(FP), R6 // src
       MOVD    src_len+64(FP), R7  // len
loop:
       KMCTR   R4, R2, R6          // cipher message with counter (KMCTR)
       BVS     loop                // branch back if interrupted
       RET
crash:
       MOVD    $0, (R0)
       RET


// func ghashAsm(key *gcmHashKey, hash *[16]byte, data []byte)
TEXT ·ghashAsm(SB),NOSPLIT,$32-40
	MOVD    $65, R0 // GHASH function code
	MOVD	key+0(FP), R2
	LMG	(R2), R6, R7
	MOVD	hash+8(FP), R8
	LMG	(R8), R4, R5
	MOVD	$params-32(SP), R1
	STMG	R4, R7, (R1)
	LMG	data+16(FP), R2, R3 // R2=base, R3=len
loop:
	KIMD	R0, R2      // compute intermediate message digest (KIMD)
	BVS     loop        // branch back if interrupted
	MVC     $16, (R1), (R8)
	MOVD	$0, R0
	RET

// func kmaGCM(fn int, key, dst, src, aad []byte, tag *[16]byte, cnt *[gcmBlockSize]byte)
TEXT ·kmaGCM(SB),NOSPLIT,$112-120
	MOVD	fn+0(FP), R0
	MOVD	$params-112(SP), R1

	// load ptr/len pairs
	LMG	dst+32(FP), R2, R3 // R2=base R3=len
	LMG	src+56(FP), R4, R5 // R4=base R5=len
	LMG	aad+80(FP), R6, R7 // R6=base R7=len

	// setup parameters
	MOVD	cnt+112(FP), R8
	XC	$12, (R1), (R1)     // reserved
	MVC	$4, 12(R8), 12(R1)  // set chain value
	MVC	$16, (R8), 64(R1)   // set initial counter value
	XC	$32, 16(R1), 16(R1) // set hash subkey and tag
	SLD	$3, R7, R12
	MOVD	R12, 48(R1)         // set total AAD length
	SLD	$3, R5, R12
	MOVD	R12, 56(R1)         // set total plaintext/ciphertext length

	LMG	key+8(FP), R8, R9   // R8=base R9=len
	MVC	$16, (R8), 80(R1)   // set key
	CMPBEQ	R9, $16, kma
	MVC	$8, 16(R8), 96(R1)
	CMPBEQ	R9, $24, kma
	MVC	$8, 24(R8), 104(R1)

kma:
	KMA	R2, R6, R4       // Cipher Message with Authentication
	BVS	kma

	MOVD	tag+104(FP), R2
	MVC	$16, 16(R1), 0(R2) // copy tag to output
	MOVD	cnt+112(FP), R8
	MVC	$4, 12(R1), 12(R8) // update counter value

	RET
