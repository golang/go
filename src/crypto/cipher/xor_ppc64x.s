// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ppc64 ppc64le

#include "textflag.h"

// func xorBytesVSX(dst, a, b *byte, n int)
TEXT ·xorBytesVSX(SB), NOSPLIT, $0
	MOVD	dst+0(FP), R3	// R3 = dst
	MOVD	a+8(FP), R4	// R4 = a
	MOVD	b+16(FP), R5	// R5 = b
	MOVD	n+24(FP), R6	// R6 = n

	CMPU	R6, $32, CR7	// Check if n ≥ 32 bytes
	MOVD	R0, R8		// R8 = index
	CMPU	R6, $8, CR6	// Check if 8 ≤ n < 32 bytes
	BLT	CR6, small	// Smaller than 8
	BLT	CR7, xor16	// Case for 16 ≤ n < 32 bytes

	// Case for n ≥ 32 bytes
preloop32:
	SRD	$5, R6, R7	// Setup loop counter
	MOVD	R7, CTR
	MOVD	$16, R10
	ANDCC	$31, R6, R9	// Check for tailing bytes for later
loop32:
	LXVD2X		(R4)(R8), VS32		// VS32 = a[i,...,i+15]
	LXVD2X		(R4)(R10), VS34
	LXVD2X		(R5)(R8), VS33		// VS33 = b[i,...,i+15]
	LXVD2X		(R5)(R10), VS35
	XXLXOR		VS32, VS33, VS32	// VS34 = a[] ^ b[]
	XXLXOR		VS34, VS35, VS34
	STXVD2X		VS32, (R3)(R8)		// Store to dst
	STXVD2X		VS34, (R3)(R10)
	ADD		$32, R8			// Update index
	ADD		$32, R10
	BC		16, 0, loop32		// bdnz loop16

	BEQ		CR0, done

	MOVD		R9, R6
	CMP		R6, $8
	BLT		small
xor16:
	CMP		R6, $16
	BLT		xor8
	LXVD2X		(R4)(R8), VS32
	LXVD2X		(R5)(R8), VS33
	XXLXOR		VS32, VS33, VS32
	STXVD2X		VS32, (R3)(R8)
	ADD		$16, R8
	ADD		$-16, R6
	CMP		R6, $8
	BLT		small
xor8:
	// Case for 8 ≤ n < 16 bytes
	MOVD    (R4)(R8), R14   // R14 = a[i,...,i+7]
	MOVD    (R5)(R8), R15   // R15 = b[i,...,i+7]
	XOR     R14, R15, R16   // R16 = a[] ^ b[]
	SUB     $8, R6          // n = n - 8
	MOVD    R16, (R3)(R8)   // Store to dst
	ADD     $8, R8

	// Check if we're finished
	CMP     R6, R0
	BGT     small
	RET

	// Case for n < 8 bytes and tailing bytes from the
	// previous cases.
small:
	CMP	R6, R0
	BEQ	done
	MOVD	R6, CTR		// Setup loop counter

loop:
	MOVBZ	(R4)(R8), R14	// R14 = a[i]
	MOVBZ	(R5)(R8), R15	// R15 = b[i]
	XOR	R14, R15, R16	// R16 = a[i] ^ b[i]
	MOVB	R16, (R3)(R8)	// Store to dst
	ADD	$1, R8
	BC	16, 0, loop	// bdnz loop

done:
	RET
