// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

#include "textflag.h"

// func cryptBlocks(c code, key, dst, src *byte, length int)
TEXT ·cryptBlocks(SB),NOSPLIT,$0-40
	MOVD	key+8(FP), R1
	MOVD	dst+16(FP), R2
	MOVD	src+24(FP), R4
	MOVD	length+32(FP), R5
	MOVD	c+0(FP), R0
loop:
	KM	R2, R4      // cipher message (KM)
	BVS	loop        // branch back if interrupted
	XOR	R0, R0
	RET

// func cryptBlocksChain(c code, iv, key, dst, src *byte, length int)
TEXT ·cryptBlocksChain(SB),NOSPLIT,$48-48
	LA	params-48(SP), R1
	MOVD	iv+8(FP), R8
	MOVD	key+16(FP), R9
	MVC	$16, 0(R8), 0(R1)  // move iv into params
	MVC	$32, 0(R9), 16(R1) // move key into params
	MOVD	dst+24(FP), R2
	MOVD	src+32(FP), R4
	MOVD	length+40(FP), R5
	MOVD	c+0(FP), R0
loop:
	KMC	R2, R4            // cipher message with chaining (KMC)
	BVS	loop              // branch back if interrupted
	XOR	R0, R0
	MVC	$16, 0(R1), 0(R8) // update iv
	RET

