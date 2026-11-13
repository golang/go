// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func LastIndexByte(b []byte, c byte) int
// input:
//   R0: b ptr
//   R1: b len
//   R2: b cap (unused)
//   R3: c byte to search
// return
//   R0: result
TEXT ·LastIndexByte<ABIInternal>(SB),NOSPLIT,$0-40
	MOVD	R3, R2
	B	·LastIndexByteString<ABIInternal>(SB)

// func LastIndexByteString(s string, c byte) int
// input:
//   R0: s ptr
//   R1: s len
//   R2: c byte to search
// return
//   R0: result
TEXT ·LastIndexByteString<ABIInternal>(SB),NOSPLIT,$0-32
	// Core algorithm uses the same chunked NEON comparison as
	// IndexByteString (see indexbyte_arm64.s): we load 16-byte chunks,
	// compare against the target byte, and narrow the per-byte
	// comparison mask down to one nibble per byte in a general purpose
	// register. Unlike IndexByteString, we scan from the end of the
	// string towards the start, and within a chunk we pick the
	// highest-indexed match (the last matching nibble) instead of the
	// lowest.

#define	PTR	R0
#define	LEN	R1
#define	TARGET	R2
#define	WIN	R3
#define	MASK	R4
#define	VTARGET	V0.B16

	// Length 0, nothing to do.
	CBZ	LEN, fail

	// Make vector containing the byte we're searching for in each lane.
	VMOV	TARGET, VTARGET		// [c c c ... c c c]

	// Small, need to be extra careful about out of bounds.
	CMP	$16, LEN
	BLT	small

	// WIN = address of the rightmost 16-byte window.
	ADD	PTR, LEN, WIN
	SUB	$16, WIN

loop:
	VLD1	(WIN), [V2.B16]		// load data
	VCMEQ	VTARGET, V2.B16, V2.B16	// compare each byte against the target byte
	VSHRN	$4, V2.H8, V2.B8	// compact to one nibble per byte
	VMOV	V2.D[0], MASK		// move to general purpose register
	CBNZ	MASK, found
	SUB	$16, WIN		// move the window to the left
	CMP	PTR, WIN
	BHI	loop

	// Check the leftmost 16-byte window, starting exactly at PTR. This
	// window may overlap with the windows we've already checked, but
	// that's ok.
	MOVD	PTR, WIN
	VLD1	(WIN), [V2.B16]
	VCMEQ	VTARGET, V2.B16, V2.B16
	VSHRN	$4, V2.H8, V2.B8
	VMOV	V2.D[0], MASK
	CBZ	MASK, fail

// MASK contains a nonzero nibble bitmask of matches within the 16-byte
// window loaded from WIN; find the highest (rightmost) one.
found:
	CLZ	MASK, R9
	MOVD	$63, R10
	SUB	R9, R10, R9		// R9 = index of the highest set bit
	LSR	$2, R9, R9		// R9 = nibble index of the rightmost match in the window
	SUB	PTR, WIN, R10		// R10 = offset of the window start from the string start
	ADD	R9, R10, R0
	RET

fail:
	MOVD	$-1, R0
	RET

	// 1-15 bytes
	PCALIGN	$16
small:
	AND	$0xf, PTR, R8		// R8 = offset of start in 16-byte region
	ADD	R8, LEN, R9		// R9 = offset of end (from start's 16-byte region boundary)
	MOVD	$0, R10			// R10 = low bits of match data to throw away
	CMP	$16, R9
	BGT	smallNoAdjust		// straddles two 16-byte regions - safe to load directly

	// data is all within a single 16-byte region
	BIC	$0xf, PTR, PTR		// round down to start of 16-byte region
	LSL	$2, R8, R10		// throw away the match bits below original start

smallNoAdjust:
	VLD1	(PTR), [V2.B16]
	VCMEQ	VTARGET, V2.B16, V2.B16
	VSHRN	$4, V2.H8, V2.B8	// compact to one nibble per byte
	VMOV	V2.D[0], MASK		// move to general purpose register
	LSR	R10, MASK, MASK		// discard matches before string start

	// Discard matches at or beyond the end of the string.
	MOVD	$1, R9
	LSL	$2, LEN, R11
	LSL	R11, R9, R9
	SUB	$1, R9, R9
	AND	R9, MASK, MASK
	CBZ	MASK, fail

	CLZ	MASK, R9
	MOVD	$63, R11
	SUB	R9, R11, R9
	LSR	$2, R9, R0
	RET
