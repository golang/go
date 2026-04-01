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
	// Core algorithm:
	// We load chunks of data, 16 bytes at a time.
	// We compare them with the target byte using
	// a vector comparison. The vector comparison
	// generates byte mask outputs. We convert the byte
	// mask to a nibble mask and move it to an int register.
	// The lowest bit index / 4 is the matching lane.
	//
	// An example (using 8 byte vectors for clarity -
	// the real code uses 16 byte vectors):
	//
	// target:     [92 92 92 92 92 92 92 92] repeat of input byte
	// data:       [3a 92 3c 47 21 03 92 b9] loaded from input string
	// equalMask:  [00 ff 00 00 00 00 ff 00] comparison (VCMEQ)
	// nibbleMask: [f0 00 00 0f .. .. .. ..] narrow to nibbles (VSHRN $4)
	// register:   0x0f0000f0                move to int register
	// idx:        1                         lowest set bit / 4
	//
	// If there is any match, the int register is nonzero and
	// we can read the match index off from it. Otherwise, there
	// is no match in those 16 bytes.
	//
	// For the small-input cases (<16 bytes), we have to be careful
	// not to read outside 16-byte aligned chunks because of MTE.
	// We use some additional masking to extract only matches that
	// are valid for the input data.

#define	PTR	R0
#define	LEN	R1
#define	TARGET	R2
#define	BASE	R3
#define MASK	R4
#define	VTARGET	V0.B16

	// Length 0, nothing to do.
	CBZ	LEN, fail

	// Make vector containing the byte we're searching for in each lane.
	VMOV	TARGET, VTARGET		// [c c c ... c c c]

	// Small, need to be extra careful about out of bounds.
	CMP	$16, LEN
	BLT	small

	// Save original location (for computing result).
	MOVD	PTR, BASE

	// Check low bits of length.
	AND	$0xf, LEN, R9
	CBZ	R9, multipleOf16

	// Length is not a multiple of 16.
	// Check 16 bytes, but then advance less to make length aligned.
	VLD1	(PTR), [V2.B16]		// load data
	VCMEQ	VTARGET, V2.B16, V2.B16	// compare each byte against the target byte
	VSHRN	$4, V2.H8, V2.B8	// compact to one nibble per byte
	VMOV	V2.D[0], MASK		// move to general purpose register
	CBNZ	MASK, foundStart
	ADD	R9, PTR
	SUB	R9, LEN

	// Length is a nonzero multiple of 16.
multipleOf16:
	TBZ	$4, LEN, multipleOf32
	VLD1.P	(PTR), [V2.B16]
	SUB	$16, LEN
	VCMEQ	VTARGET, V2.B16, V2.B16
	VSHRN	$4, V2.H8, V2.B8
	VMOV	V2.D[0], MASK
	CBNZ	MASK, found16
	CBZ	LEN, fail

	// Length is a nonzero multiple of 32.
multipleOf32:
	TBZ	$5, LEN, multipleOf64
	VLD1.P	(PTR), [V2.B16, V3.B16]	// load data, PTR += 32
	SUB	$32, LEN
	VCMEQ	VTARGET, V2.B16, V2.B16
	VCMEQ	VTARGET, V3.B16, V3.B16
	VSHRN	$4, V2.H8, V2.B8
	VSHRN	$4, V3.H8, V3.B8
	VMOV	V2.D[0], MASK
	CBNZ	MASK, found32
	VMOV	V3.D[0], MASK
	CBNZ	MASK, found16
	CBZ	LEN, fail

	// Length is a nonzero multiple of 64.
multipleOf64:
	VLD1.P	(PTR), [V2.B16, V3.B16, V4.B16, V5.B16]	// load data, PTR += 64
	SUB	$64, LEN
	VCMEQ	VTARGET, V2.B16, V2.B16
	VCMEQ	VTARGET, V3.B16, V3.B16
	VCMEQ	VTARGET, V4.B16, V4.B16
	VCMEQ	VTARGET, V5.B16, V5.B16
	VORR	V2.B16, V3.B16, V10.B16
	VORR	V4.B16, V5.B16, V11.B16
	VORR	V10.B16, V11.B16, V10.B16
	VADDP	V10.D2, V10.D2, V10.D2
	VMOV	V10.D[0], R10
	CBNZ	R10, found64		// at least one lane matched
	CBNZ	LEN, multipleOf64

fail:
	MOVD	$-1, R0
	RET

foundStart:
	RBIT	MASK, MASK
	CLZ	MASK, R9		// count trailing zeros
	LSR	$2, R9, R0		// divide by 4
	RET

found16:
	SUB	$16, PTR		// undo .P

	// On entry to found0, MASK contains a nonzero nibble bitmask
	// of matches starting at PTR.
found0:
	SUB	BASE, PTR		// convert pointer to offset
	RBIT	MASK, MASK
	CLZ	MASK, R9		// count trailing zeros
	ADD	R9>>2, PTR, R0		// add nibble index to offset
	RET

found32:
	SUB	$32, PTR		// undo .P
	B	found0

found64:
	SUB	$64, PTR		// undo .P
	VSHRN	$4, V2.H8, V2.B8
	VMOV	V2.D[0], MASK
	CBNZ	MASK, found0
	ADD	$16, PTR		// redo 1/4 of .P
	VSHRN	$4, V3.H8, V3.B8
	VMOV	V3.D[0], MASK
	CBNZ	MASK, found0
	ADD	$16, PTR		// redo 1/4 of .P
	VSHRN	$4, V4.H8, V4.B8
	VMOV	V4.D[0], MASK
	CBNZ	MASK, found0
	ADD	$16, PTR		// redo 1/4 of .P
	VSHRN	$4, V5.H8, V5.B8
	VMOV	V5.D[0], MASK
	B	found0

	// 1-15 bytes
	PCALIGN	$16
small:
	AND	$0xf, PTR, R8		// R8 = offset of start in 16-byte region
	ADD	R8, LEN, R9		// R9 = offset of end (from start's 16-byte region boundary)
	MOVD	$0, R10			// R10 = low bits of match data to throw away
	CMP	$16, R9
	BGT	noAdjust		// straddles two 16-byte regions - safe to load directly

	// data is all within a single 16-byte region
	BIC	$0xf, PTR, PTR		// round down to start of 16-byte region
	LSL	$2, R8, R10		// throw away the match bits below original start

noAdjust:
	VLD1	(PTR), [V2.B16]
	VCMEQ	VTARGET, V2.B16, V2.B16
	VSHRN	$4, V2.H8, V2.B8	// compact to one nibble per byte
	VMOV	V2.D[0], MASK		// move to general purpose register
	LSR	R10, MASK, MASK		// discard matches before string start
	RBIT	MASK, MASK
	CLZ	MASK, R9
	LSR	$2, R9, R9
	CMP	LEN, R9
	CSINV	LT, R9, ZR, R0		// if first match past end of string, set return value to -1
	RET
