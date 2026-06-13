// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func IndexByte(b []byte, c byte) int
// input:
//   R0: b ptr
//   R1: b len
//   R2: b cap (unused)
//   R3: c byte to search
// return
//   R0: result
TEXT ·IndexByte<ABIInternal>(SB),NOSPLIT,$0-40
	MOVD	R3, R2
	B	·IndexByteString<ABIInternal>(SB)

// func IndexByteString(s string, c byte) int
// input:
//   R0: s ptr
//   R1: s len
//   R2: c byte to search
// return
//   R0: result
TEXT ·IndexByteString<ABIInternal>(SB),NOSPLIT,$0-32
	// MTE-safe implementation: never reads before start or beyond end.
	// Uses SWAR for 8-15 byte strings, overlapping NEON loads for
	// 16-31 byte strings, and NEON prefix + aligned loop for 32+.

	CBZ	R1, fail
	MOVD	R0, R11          // Save original pointer
	ADD	R0, R1, R4       // R4 = end pointer

	// Small string: scalar byte-by-byte
	CMP	$8, R1
	BLO	tail_loop

	// Broadcast search byte (used by all SIMD/SWAR paths)
	VMOV	R2, V0.B16

	// Small-medium string: SWAR with two overlapping 8-byte loads
	CMP	$16, R1
	BLO	small_8_15

	// Setup SIMD syndrome constant (for N >= 16 NEON paths)
	// Magic constant 0x40100401 allows us to identify which lane matches.
	// 0x40100401 = ((1<<0) + (4<<8) + (16<<16) + (64<<24))
	MOVD	$0x40100401, R5
	VMOV	R5, V5.S4

	// Medium string: two overlapping 16-byte NEON loads
	CMP	$32, R1
	BLO	medium_16_31

	// --- N >= 32: aligned SIMD main loop ---
	// Handle unaligned prefix with NEON (MTE-safe: N >= 32)
	ANDS	$0x1f, R0, R9
	BEQ	aligned_start

	// Load first 32 bytes (unaligned, safe since N >= 32)
	VLD1	(R0), [V1.B16, V2.B16]
	VCMEQ	V0.B16, V1.B16, V3.B16
	VCMEQ	V0.B16, V2.B16, V4.B16
	VORR	V4.B16, V3.B16, V6.B16
	VADDP	V6.D2, V6.D2, V6.D2
	VMOV	V6.D[0], R6
	CBNZ	R6, found32_at_r0

	// Advance to next 32-byte aligned boundary
	BIC	$0x1f, R0, R0
	ADD	$32, R0

aligned_start:
	// Check if we have at least 32 bytes remaining BEFORE loading
	SUB	R0, R4, R5       // R5 = remaining bytes
	CMP	$32, R5
	BLO	tail16_check

loop32:
	// Load 32 aligned bytes - safe because we verified 32+ bytes remain
	VLD1.P	32(R0), [V1.B16, V2.B16]

	// Compare both vectors against search byte
	VCMEQ	V0.B16, V1.B16, V3.B16
	VCMEQ	V0.B16, V2.B16, V4.B16

	// Quick check: any matches in either vector?
	VORR	V4.B16, V3.B16, V6.B16
	VADDP	V6.D2, V6.D2, V6.D2
	VMOV	V6.D[0], R6
	CBNZ	R6, found32

	// Check bounds BEFORE next iteration
	SUB	R0, R4, R5
	CMP	$32, R5
	BHS	loop32

tail16_check:
	// Try 16-byte chunks if available
	CMP	$16, R5
	BLO	tail_scalar

loop16:
	// Load 16 aligned bytes
	VLD1.P	16(R0), [V1.B16]

	VCMEQ	V0.B16, V1.B16, V3.B16
	VAND	V5.B16, V3.B16, V3.B16
	VADDP	V3.B16, V3.B16, V3.B16  // 128->64
	VADDP	V3.B16, V3.B16, V3.B16  // 64->32
	VMOV	V3.S[0], R6
	CBNZ	R6, found16

	// Check bounds BEFORE next iteration
	SUB	R0, R4, R5
	CMP	$16, R5
	BHS	loop16

tail_scalar:
	// Handle remaining bytes one at a time (always safe)
	CMP	R0, R4
	BEQ	fail

tail_loop:
	MOVBU	(R0), R5
	CMP	R2, R5
	BEQ	found_scalar
	ADD	$1, R0
	CMP	R0, R4
	BNE	tail_loop
	B	fail

// --- Small string: 8 <= N < 16 using SWAR ---
// Two overlapping 8-byte scalar loads with zero-byte detection.
// has_zero(v) = ((v - 0x0101..01) & ~v & 0x8080..80) detects
// zero bytes after XOR with the replicated search byte.
small_8_15:
	VMOV	V0.D[0], R5       // R5 = search byte replicated 8 times
	MOVD	$0x0101010101010101, R7
	MOVD	$0x8080808080808080, R8

	// Check first 8 bytes
	MOVD	(R0), R6
	EOR	R5, R6, R6         // Matching bytes become 0x00
	SUB	R7, R6, R3         // (v - 0x0101..01)
	BIC	R6, R3, R3         // & ~v
	AND	R8, R3, R3         // & 0x8080..80
	CBNZ	R3, swar_found_first

	// Check last 8 bytes (overlapping, MTE-safe since N >= 8)
	MOVD	-8(R4), R6
	EOR	R5, R6, R6
	SUB	R7, R6, R3
	BIC	R6, R3, R3
	AND	R8, R3, R3
	CBNZ	R3, swar_found_last
	B	fail

swar_found_first:
	RBIT	R3, R3
	CLZ	R3, R3
	LSR	$3, R3, R0         // Bit position / 8 = byte offset
	RET

swar_found_last:
	RBIT	R3, R3
	CLZ	R3, R3
	LSR	$3, R3, R3         // Byte offset in second load
	SUB	$8, R1, R6         // Offset of second load from start
	ADD	R6, R3, R0
	RET

// --- Medium string: 16 <= N < 32 ---
// Two overlapping 16-byte NEON loads.
medium_16_31:
	// First 16 bytes (unaligned, MTE-safe since N >= 16)
	VLD1	(R0), [V1.B16]
	VCMEQ	V0.B16, V1.B16, V3.B16
	VAND	V5.B16, V3.B16, V3.B16
	VADDP	V3.B16, V3.B16, V3.B16
	VADDP	V3.B16, V3.B16, V3.B16
	VMOV	V3.S[0], R6
	CBNZ	R6, found16_at_r0

	// Last 16 bytes (overlapping, MTE-safe since N >= 16)
	SUB	$16, R4, R3
	VLD1	(R3), [V1.B16]
	VCMEQ	V0.B16, V1.B16, V3.B16
	VAND	V5.B16, V3.B16, V3.B16
	VADDP	V3.B16, V3.B16, V3.B16
	VADDP	V3.B16, V3.B16, V3.B16
	VMOV	V3.S[0], R6
	CBNZ	R6, found_medium_last
	B	fail

found_medium_last:
	RBIT	R6, R6
	CLZ	R6, R6
	ADD	R6>>1, R3, R0     // R3 = load address
	SUB	R11, R0, R0
	RET

// --- Found handlers ---
found32:
	SUB	$32, R0, R0       // Compensate post-increment
found32_at_r0:
	// Calculate full syndrome for precise position
	VAND	V5.B16, V3.B16, V3.B16
	VAND	V5.B16, V4.B16, V4.B16
	VADDP	V4.B16, V3.B16, V6.B16  // 256->128
	VADDP	V6.B16, V6.B16, V6.B16  // 128->64
	VMOV	V6.D[0], R6
	RBIT	R6, R6
	CLZ	R6, R6
	ADD	R6>>1, R0, R0    // R6 is twice the byte offset
	SUB	R11, R0, R0
	RET

found16:
	SUB	$16, R0, R0       // Compensate post-increment
found16_at_r0:
	RBIT	R6, R6
	CLZ	R6, R6
	ADD	R6>>1, R0, R0    // R6 is twice the byte offset
	SUB	R11, R0, R0
	RET

found_scalar:
	SUB	R11, R0, R0
	RET

fail:
	MOVD	$-1, R0
	RET
