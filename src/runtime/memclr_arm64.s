// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// See memclrNoHeapPointers Go doc for important implementation constraints.

// func memclrNoHeapPointers(ptr unsafe.Pointer, n uintptr)
// Also called from assembly in sys_windows_arm64.s without g (but using Go stack convention).
TEXT runtime·memclrNoHeapPointers<ABIInternal>(SB),NOSPLIT,$0-16
#ifndef GOEXPERIMENT_regabiargs
	MOVD	ptr+0(FP), R0
	MOVD	n+8(FP), R1
#endif

	CMP	$16, R1
	// If n is equal to 16 bytes, use zero_exact_16 to zero
	BEQ	zero_exact_16

	// If n is greater than 16 bytes, use zero_by_16 to zero
	BHI	zero_by_16

	// n is less than 16 bytes
	ADD	R1, R0, R7
	TBZ	$3, R1, less_than_8
	MOVD	ZR, (R0)
	MOVD	ZR, -8(R7)
	RET

less_than_8:
	TBZ	$2, R1, less_than_4
	MOVW	ZR, (R0)
	MOVW	ZR, -4(R7)
	RET

less_than_4:
	CBZ	R1, ending
	MOVB	ZR, (R0)
	TBZ	$1, R1, ending
	MOVH	ZR, -2(R7)

ending:
	RET

zero_exact_16:
	// n is exactly 16 bytes
	STP	(ZR, ZR), (R0)
	RET

zero_by_16:
	// n greater than 16 bytes, check if the start address is aligned
	NEG	R0, R4
	ANDS	$15, R4, R4
	// Try zeroing using zva if the start address is aligned with 16
	BEQ	try_zva

	// Non-aligned store
	STP	(ZR, ZR), (R0)
	// Make the destination aligned
	SUB	R4, R1, R1
	ADD	R4, R0, R0
	B	try_zva

tail_maybe_long:
	CMP	$64, R1
	BHS	no_zva

tail63:
	ANDS	$48, R1, R3
	BEQ	last16
	CMPW	$32, R3
	BEQ	last48
	BLT	last32
	STP.P	(ZR, ZR), 16(R0)
last48:
	STP.P	(ZR, ZR), 16(R0)
last32:
	STP.P	(ZR, ZR), 16(R0)
	// The last store length is at most 16, so it is safe to use
	// stp to write last 16 bytes
last16:
	ANDS	$15, R1, R1
	CBZ	R1, last_end
	ADD	R1, R0, R0
	STP	(ZR, ZR), -16(R0)
last_end:
	RET

no_zva:
	SUB	$16, R0, R0
	SUB	$64, R1, R1

loop_64:
	STP	(ZR, ZR), 16(R0)
	STP	(ZR, ZR), 32(R0)
	STP	(ZR, ZR), 48(R0)
	STP.W	(ZR, ZR), 64(R0)
	SUBS	$64, R1, R1
	BGE	loop_64
	ANDS	$63, R1, ZR
	ADD	$16, R0, R0
	BNE	tail63
	RET

try_zva:
	// Try using the ZVA feature to zero entire cache lines
	// It is not meaningful to use ZVA if the block size is less than 64,
	// so make sure that n is greater than or equal to 64
	CMP	$63, R1
	BLE	tail63

	CMP	$128, R1
	// Ensure n is at least 128 bytes, so that there is enough to copy after
	// alignment.
	BLT	no_zva
	// Check if ZVA is allowed from user code, and if so get the block size
	MOVW	block_size<>(SB), R5
	TBNZ	$31, R5, no_zva
	CBNZ	R5, zero_by_line
	// DCZID_EL0 bit assignments
	// [63:5] Reserved
	// [4]    DZP, if bit set DC ZVA instruction is prohibited, else permitted
	// [3:0]  log2 of the block size in words, eg. if it returns 0x4 then block size is 16 words
	MRS	DCZID_EL0, R3
	TBZ	$4, R3, init
	// ZVA not available
	MOVW	$~0, R5
	MOVW	R5, block_size<>(SB)
	B	no_zva

init:
	MOVW	$4, R9
	ANDW	$15, R3, R5
	LSLW	R5, R9, R5
	MOVW	R5, block_size<>(SB)

	ANDS	$63, R5, R9
	// Block size is less than 64.
	BNE	no_zva

zero_by_line:
	CMP	R5, R1
	// Not enough memory to reach alignment
	BLO	no_zva
	SUB	$1, R5, R6
	NEG	R0, R4
	ANDS	R6, R4, R4
	// Already aligned
	BEQ	aligned

	// check there is enough to copy after alignment
	SUB	R4, R1, R3

	// Check that the remaining length to ZVA after alignment
	// is greater than 64.
	CMP	$64, R3
	CCMP	GE, R3, R5, $10  // condition code GE, NZCV=0b1010
	BLT	no_zva

	// We now have at least 64 bytes to zero, update n
	MOVD	R3, R1

loop_zva_prolog:
	STP	(ZR, ZR), (R0)
	STP	(ZR, ZR), 16(R0)
	STP	(ZR, ZR), 32(R0)
	SUBS	$64, R4, R4
	STP	(ZR, ZR), 48(R0)
	ADD	$64, R0, R0
	BGE	loop_zva_prolog

	ADD	R4, R0, R0

aligned:
	SUB	R5, R1, R1

loop_zva:
	WORD	$0xd50b7420 // DC ZVA, R0
	ADD	R5, R0, R0
	SUBS	R5, R1, R1
	BHS	loop_zva
	ANDS	R6, R1, R1
	BNE	tail_maybe_long
	RET

GLOBL block_size<>(SB), NOPTR, $8
