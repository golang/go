// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// See memmove Go doc for important implementation constraints.

// Register map
//
// dstin  R0
// src    R1
// count  R2
// dst    R3 (same as R0, but gets modified in unaligned cases)
// srcend R4
// dstend R5
// data   R6-R17
// tmp1   R14

// Copies are split into 3 main cases: small copies of up to 32 bytes, medium
// copies of up to 128 bytes, and large copies. The overhead of the overlap
// check is negligible since it is only required for large copies.
//
// Large copies use a software pipelined loop processing 64 bytes per iteration.
// The destination pointer is 16-byte aligned to minimize unaligned accesses.
// The loop tail is handled by always copying 64 bytes from the end.

// func memmove(to, from unsafe.Pointer, n uintptr)
TEXT runtime·memmove(SB), NOSPLIT|NOFRAME, $0-24
	MOVD	to+0(FP), R0
	MOVD	from+8(FP), R1
	MOVD	n+16(FP), R2
	CBZ	R2, copy0

	// Small copies: 1..16 bytes
	CMP	$16, R2
	BLE	copy16

	// Large copies
	CMP	$128, R2
	BHI	copy_long
	CMP	$32, R2
	BHI	copy32_128

	// Small copies: 17..32 bytes.
	LDP	(R1), (R6, R7)
	ADD	R1, R2, R4          // R4 points just past the last source byte
	LDP	-16(R4), (R12, R13)
	STP	(R6, R7), (R0)
	ADD	R0, R2, R5          // R5 points just past the last destination byte
	STP	(R12, R13), -16(R5)
	RET

// Small copies: 1..16 bytes.
copy16:
	ADD	R1, R2, R4 // R4 points just past the last source byte
	ADD	R0, R2, R5 // R5 points just past the last destination byte
	CMP	$8, R2
	BLT	copy7
	MOVD	(R1), R6
	MOVD	-8(R4), R7
	MOVD	R6, (R0)
	MOVD	R7, -8(R5)
	RET

copy7:
	TBZ	$2, R2, copy3
	MOVWU	(R1), R6
	MOVWU	-4(R4), R7
	MOVW	R6, (R0)
	MOVW	R7, -4(R5)
	RET

copy3:
	TBZ	$1, R2, copy1
	MOVHU	(R1), R6
	MOVHU	-2(R4), R7
	MOVH	R6, (R0)
	MOVH	R7, -2(R5)
	RET

copy1:
	MOVBU	(R1), R6
	MOVB	R6, (R0)

copy0:
	RET

	// Medium copies: 33..128 bytes.
copy32_128:
	ADD	R1, R2, R4          // R4 points just past the last source byte
	ADD	R0, R2, R5          // R5 points just past the last destination byte
	LDP	(R1), (R6, R7)
	LDP	16(R1), (R8, R9)
	LDP	-32(R4), (R10, R11)
	LDP	-16(R4), (R12, R13)
	CMP	$64, R2
	BHI	copy128
	STP	(R6, R7), (R0)
	STP	(R8, R9), 16(R0)
	STP	(R10, R11), -32(R5)
	STP	(R12, R13), -16(R5)
	RET

	// Copy 65..128 bytes.
copy128:
	LDP	32(R1), (R14, R15)
	LDP	48(R1), (R16, R17)
	CMP	$96, R2
	BLS	copy96
	LDP	-64(R4), (R2, R3)
	LDP	-48(R4), (R1, R4)
	STP	(R2, R3), -64(R5)
	STP	(R1, R4), -48(R5)

copy96:
	STP	(R6, R7), (R0)
	STP	(R8, R9), 16(R0)
	STP	(R14, R15), 32(R0)
	STP	(R16, R17), 48(R0)
	STP	(R10, R11), -32(R5)
	STP	(R12, R13), -16(R5)
	RET

	// Copy more than 128 bytes.
copy_long:
	ADD	R1, R2, R4 // R4 points just past the last source byte
	ADD	R0, R2, R5 // R5 points just past the last destination byte
	MOVD	ZR, R7
	MOVD	ZR, R8

	CMP	$1024, R2
	BLT	backward_check
	// feature detect to decide how to align
	MOVBU	runtime·arm64UseAlignedLoads(SB), R6
	CBNZ	R6, use_aligned_loads
	MOVD	R0, R7
	MOVD	R5, R8
	B	backward_check
use_aligned_loads:
	MOVD	R1, R7
	MOVD	R4, R8
	// R7 and R8 are used here for the realignment calculation. In
	// the use_aligned_loads case, R7 is the src pointer and R8 is
	// srcend pointer, which is used in the backward copy case.
	// When doing aligned stores, R7 is the dst pointer and R8 is
	// the dstend pointer.

backward_check:
	// Use backward copy if there is an overlap.
	SUB	R1, R0, R14
	CBZ	R14, copy0
	CMP	R2, R14
	BCC	copy_long_backward

	// Copy 16 bytes and then align src (R1) or dst (R0) to 16-byte alignment.
	LDP	(R1), (R12, R13)     // Load  A
	AND	$15, R7, R14         // Calculate the realignment offset
	SUB	R14, R1, R1
	SUB	R14, R0, R3          // move dst back same amount as src
	ADD	R14, R2, R2
	LDP	16(R1), (R6, R7)     // Load   B
	STP	(R12, R13), (R0)     // Store A
	LDP	32(R1), (R8, R9)     // Load    C
	LDP	48(R1), (R10, R11)   // Load     D
	LDP.W	64(R1), (R12, R13)   // Load      E
	// 80 bytes have been loaded; if less than 80+64 bytes remain, copy from the end
	SUBS	$144, R2, R2
	BLS	copy64_from_end

loop64:
	STP	(R6, R7), 16(R3)     // Store  B
	LDP	16(R1), (R6, R7)     // Load   B (next iteration)
	STP	(R8, R9), 32(R3)     // Store   C
	LDP	32(R1), (R8, R9)     // Load    C
	STP	(R10, R11), 48(R3)   // Store    D
	LDP	48(R1), (R10, R11)   // Load     D
	STP.W	(R12, R13), 64(R3)   // Store     E
	LDP.W	64(R1), (R12, R13)   // Load      E
	SUBS	$64, R2, R2
	BHI	loop64

	// Write the last iteration and copy 64 bytes from the end.
copy64_from_end:
	LDP	-64(R4), (R14, R15)  // Load       F
	STP	(R6, R7), 16(R3)     // Store  B
	LDP	-48(R4), (R6, R7)    // Load        G
	STP	(R8, R9), 32(R3)     // Store   C
	LDP	-32(R4), (R8, R9)    // Load         H
	STP	(R10, R11), 48(R3)   // Store    D
	LDP	-16(R4), (R10, R11)  // Load          I
	STP	(R12, R13), 64(R3)   // Store     E
	STP	(R14, R15), -64(R5)  // Store      F
	STP	(R6, R7), -48(R5)    // Store       G
	STP	(R8, R9), -32(R5)    // Store        H
	STP	(R10, R11), -16(R5)  // Store         I
	RET

	// Large backward copy for overlapping copies.
	// Copy 16 bytes and then align srcend (R4) or dstend (R5) to 16-byte alignment.
copy_long_backward:
	LDP	-16(R4), (R12, R13)
	AND	$15, R8, R14
	SUB	R14, R4, R4
	SUB	R14, R2, R2
	LDP	-16(R4), (R6, R7)
	STP	(R12, R13), -16(R5)
	LDP	-32(R4), (R8, R9)
	LDP	-48(R4), (R10, R11)
	LDP.W	-64(R4), (R12, R13)
	SUB	R14, R5, R5
	SUBS	$128, R2, R2
	BLS	copy64_from_start

loop64_backward:
	STP	(R6, R7), -16(R5)
	LDP	-16(R4), (R6, R7)
	STP	(R8, R9), -32(R5)
	LDP	-32(R4), (R8, R9)
	STP	(R10, R11), -48(R5)
	LDP	-48(R4), (R10, R11)
	STP.W	(R12, R13), -64(R5)
	LDP.W	-64(R4), (R12, R13)
	SUBS	$64, R2, R2
	BHI	loop64_backward

	// Write the last iteration and copy 64 bytes from the start.
copy64_from_start:
	LDP	48(R1), (R2, R3)
	STP	(R6, R7), -16(R5)
	LDP	32(R1), (R6, R7)
	STP	(R8, R9), -32(R5)
	LDP	16(R1), (R8, R9)
	STP	(R10, R11), -48(R5)
	LDP	(R1), (R10, R11)
	STP	(R12, R13), -64(R5)
	STP	(R2, R3), 48(R0)
	STP	(R6, R7), 32(R0)
	STP	(R8, R9), 16(R0)
	STP	(R10, R11), (R0)
	RET
