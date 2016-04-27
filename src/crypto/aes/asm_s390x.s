// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func hasAsm() bool
TEXT ·hasAsm(SB),NOSPLIT,$16-1
	XOR	R0, R0          // set function code to 0 (query)
	LA	mask-16(SP), R1 // 16-byte stack variable for mask
	WORD	$0xB92E0024     // cipher message (KM)

	// check if bits 18-20 (big endian) are set
	MOVD	mask-16(SP), R2
	MOVD	$(0x38<<40), R3
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
