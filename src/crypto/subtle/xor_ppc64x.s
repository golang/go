// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (ppc64 || ppc64le) && !purego

#include "textflag.h"

// func xorBytes(dst, a, b *byte, n int)
TEXT ·xorBytes(SB), NOSPLIT, $0
	MOVD	dst+0(FP), R3	// R3 = dst
	MOVD	a+8(FP), R4	// R4 = a
	MOVD	b+16(FP), R5	// R5 = b
	MOVD	n+24(FP), R6	// R6 = n

	CMPU	R6, $64, CR7	// Check if n ≥ 64 bytes
	MOVD	R0, R8		// R8 = index
	CMPU	R6, $8, CR6	// Check if 8 ≤ n < 64 bytes
	BLE	CR6, small	// <= 8
	BLT	CR7, xor32	// Case for 32 ≤ n < 64 bytes

	// Case for n ≥ 64 bytes
preloop64:
	SRD	$6, R6, R7	// Set up loop counter
	MOVD	R7, CTR
	MOVD	$16, R10
	MOVD	$32, R14
	MOVD	$48, R15
	ANDCC	$63, R6, R9	// Check for tailing bytes for later
	PCALIGN $16
	// Case for >= 64 bytes
	// Process 64 bytes per iteration
	// Load 4 vectors of a and b
	// XOR the corresponding vectors
	// from a and b and store the result
loop64:
	LXVD2X	(R4)(R8), VS32
	LXVD2X	(R4)(R10), VS34
	LXVD2X	(R4)(R14), VS36
	LXVD2X	(R4)(R15), VS38
	LXVD2X	(R5)(R8), VS33
	LXVD2X	(R5)(R10), VS35
	LXVD2X	(R5)(R14), VS37
	LXVD2X	(R5)(R15), VS39
	XXLXOR	VS32, VS33, VS32
	XXLXOR	VS34, VS35, VS34
	XXLXOR	VS36, VS37, VS36
	XXLXOR	VS38, VS39, VS38
	STXVD2X	VS32, (R3)(R8)
	STXVD2X	VS34, (R3)(R10)
	STXVD2X	VS36, (R3)(R14)
	STXVD2X	VS38, (R3)(R15)
	ADD	$64, R8
	ADD	$64, R10
	ADD	$64, R14
	ADD	$64, R15
	BDNZ	loop64
	BC	12,2,LR		// BEQLR
	MOVD	R9, R6
	CMP	R6, $8
	BLE	small
	// Case for 8 <= n < 64 bytes
	// Process 32 bytes if available
xor32:
	CMP	R6, $32
	BLT	xor16
	ADD	$16, R8, R9
	LXVD2X	(R4)(R8), VS32
	LXVD2X	(R4)(R9), VS33
	LXVD2X	(R5)(R8), VS34
	LXVD2X	(R5)(R9), VS35
	XXLXOR	VS32, VS34, VS32
	XXLXOR	VS33, VS35, VS33
	STXVD2X	VS32, (R3)(R8)
	STXVD2X	VS33, (R3)(R9)
	ADD	$32, R8
	ADD	$-32, R6
	CMP	R6, $8
	BLE	small
	// Case for 8 <= n < 32 bytes
	// Process 16 bytes if available
xor16:
	CMP	R6, $16
	BLT	xor8
	LXVD2X	(R4)(R8), VS32
	LXVD2X	(R5)(R8), VS33
	XXLXOR	VS32, VS33, VS32
	STXVD2X	VS32, (R3)(R8)
	ADD	$16, R8
	ADD	$-16, R6
small:
	CMP	R6, R0
	BC	12,2,LR		// BEQLR
xor8:
#ifdef GOPPC64_power10
	SLD	$56,R6,R17
	ADD	R4,R8,R18
	ADD	R5,R8,R19
	ADD	R3,R8,R20
	LXVL	R18,R17,V0
	LXVL	R19,R17,V1
	VXOR	V0,V1,V1
	STXVL	V1,R20,R17
	RET
#else
	CMP	R6, $8
	BLT	xor4
	// Case for 8 ≤ n < 16 bytes
	MOVD	(R4)(R8), R14   // R14 = a[i,...,i+7]
	MOVD	(R5)(R8), R15   // R15 = b[i,...,i+7]
	XOR	R14, R15, R16   // R16 = a[] ^ b[]
	SUB	$8, R6          // n = n - 8
	MOVD	R16, (R3)(R8)   // Store to dst
	ADD	$8, R8
xor4:
	CMP	R6, $4
	BLT	xor2
	MOVWZ	(R4)(R8), R14
	MOVWZ	(R5)(R8), R15
	XOR	R14, R15, R16
	MOVW	R16, (R3)(R8)
	ADD	$4,R8
	ADD	$-4,R6
xor2:
	CMP	R6, $2
	BLT	xor1
	MOVHZ	(R4)(R8), R14
	MOVHZ	(R5)(R8), R15
	XOR	R14, R15, R16
	MOVH	R16, (R3)(R8)
	ADD	$2,R8
	ADD	$-2,R6
xor1:
	CMP	R6, R0
	BC	12,2,LR		// BEQLR
	MOVBZ	(R4)(R8), R14	// R14 = a[i]
	MOVBZ	(R5)(R8), R15	// R15 = b[i]
	XOR	R14, R15, R16	// R16 = a[i] ^ b[i]
	MOVB	R16, (R3)(R8)	// Store to dst
#endif
done:
	RET
