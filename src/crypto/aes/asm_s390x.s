// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func hasAsm() bool
TEXT ·hasAsm(SB),NOSPLIT,$16-1
	XOR	R0, R0          // set function code to 0 (query)
	LA	mask-16(SP), R1 // 16-byte stack variable for mask
	MOVD	$(0x38<<40), R3 // mask for bits 18-20 (big endian)

	// check for KM AES functions
	WORD	$0xB92E0024 // cipher message (KM)
	MOVD	mask-16(SP), R2
	AND	R3, R2
	CMPBNE	R2, R3, notfound

	// check for KMC AES functions
	WORD	$0xB92F0024 // cipher message with chaining (KMC)
	MOVD	mask-16(SP), R2
	AND	R3, R2
	CMPBNE	R2, R3, notfound

	MOVB	$1, ret+0(FP)
	RET
notfound:
	MOVB	$0, ret+0(FP)
	RET

// func cryptBlocks(function code, key, dst, src *byte, length int)
TEXT ·cryptBlocks(SB),NOSPLIT,$0-40
	MOVD	key+8(FP), R1
	MOVD	dst+16(FP), R2
	MOVD	src+24(FP), R4
	MOVD	length+32(FP), R5
	MOVD	function+0(FP), R0
loop:
	WORD	$0xB92E0024 // cipher message (KM)
	BVS	loop        // branch back if interrupted
	XOR	R0, R0
	RET

// func cryptBlocksChain(function code, iv, key, dst, src *byte, length int)
TEXT ·cryptBlocksChain(SB),NOSPLIT,$48-48
	LA	params-48(SP), R1
	MOVD	iv+8(FP), R8
	MOVD	key+16(FP), R9
	MVC	$16, 0(R8), 0(R1)  // move iv into params
	MVC	$32, 0(R9), 16(R1) // move key into params
	MOVD	dst+24(FP), R2
	MOVD	src+32(FP), R4
	MOVD	length+40(FP), R5
	MOVD	function+0(FP), R0
loop:
	WORD	$0xB92F0024       // cipher message with chaining (KMC)
	BVS	loop              // branch back if interrupted
	XOR	R0, R0
	MVC	$16, 0(R1), 0(R8) // update iv
	RET

// func xorBytes(dst, a, b []byte) int
TEXT ·xorBytes(SB),NOSPLIT,$0-80
	MOVD	dst_base+0(FP), R1
	MOVD	a_base+24(FP), R2
	MOVD	b_base+48(FP), R3
	MOVD	a_len+32(FP), R4
	MOVD	b_len+56(FP), R5
	CMPBLE	R4, R5, skip
	MOVD	R5, R4
skip:
	MOVD	R4, ret+72(FP)
	MOVD	$0, R5
	CMPBLT	R4, $8, tail
loop:
	MOVD	0(R2)(R5*1), R7
	MOVD	0(R3)(R5*1), R8
	XOR	R7, R8
	MOVD	R8, 0(R1)(R5*1)
	LAY	8(R5), R5
	SUB	$8, R4
	CMPBGE	R4, $8, loop
tail:
	CMPBEQ	R4, $0, done
	MOVB	0(R2)(R5*1), R7
	MOVB	0(R3)(R5*1), R8
	XOR	R7, R8
	MOVB	R8, 0(R1)(R5*1)
	LAY	1(R5), R5
	SUB	$1, R4
	BR	tail
done:
	RET
