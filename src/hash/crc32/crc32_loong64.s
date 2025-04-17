// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// castagnoliUpdate updates the non-inverted crc with the given data.

// func castagnoliUpdate(crc uint32, p []byte) uint32
TEXT ·castagnoliUpdate(SB),NOSPLIT,$0-36
	MOVWU	crc+0(FP), R4		// a0 = CRC value
	MOVV	p+8(FP), R5		// a1 = data pointer
	MOVV	p_len+16(FP), R6	// a2 = len(p)

	SGT	$8, R6, R12
	BNE	R12, less_than_8
	AND	$7, R5, R12
	BEQ	R12, aligned

	// Process the first few bytes to 8-byte align the input.
	// t0 = 8 - t0. We need to process this many bytes to align.
	SUB	$1, R12
	XOR	$7, R12

	AND	$1, R12, R13
	BEQ	R13, align_2
	MOVB	(R5), R13
	CRCCWBW	R4, R13, R4
	ADDV	$1, R5
	ADDV	$-1, R6

align_2:
	AND	$2, R12, R13
	BEQ	R13, align_4
	MOVH	(R5), R13
	CRCCWHW	R4, R13, R4
	ADDV	$2, R5
	ADDV	$-2, R6

align_4:
	AND	$4, R12, R13
	BEQ	R13, aligned
	MOVW	(R5), R13
	CRCCWWW	R4, R13, R4
	ADDV	$4, R5
	ADDV	$-4, R6

aligned:
	// The input is now 8-byte aligned and we can process 8-byte chunks.
	SGT	$8, R6, R12
	BNE	R12, less_than_8
	MOVV	(R5), R13
	CRCCWVW	R4, R13, R4
	ADDV	$8, R5
	ADDV	$-8, R6
	JMP	aligned

less_than_8:
	// We may have some bytes left over; process 4 bytes, then 2, then 1.
	AND	$4, R6, R12
	BEQ	R12, less_than_4
	MOVW	(R5), R13
	CRCCWWW	R4, R13, R4
	ADDV	$4, R5
	ADDV	$-4, R6

less_than_4:
	AND	$2, R6, R12
	BEQ	R12, less_than_2
	MOVH	(R5), R13
	CRCCWHW	R4, R13, R4
	ADDV	$2, R5
	ADDV	$-2, R6

less_than_2:
	BEQ	R6, done
	MOVB	(R5), R13
	CRCCWBW	R4, R13, R4

done:
	MOVW	R4, ret+32(FP)
	RET

// ieeeUpdate updates the non-inverted crc with the given data.

// func ieeeUpdate(crc uint32, p []byte) uint32
TEXT ·ieeeUpdate(SB),NOSPLIT,$0-36
	MOVWU	crc+0(FP), R4		// a0 = CRC value
	MOVV	p+8(FP), R5		// a1 = data pointer
	MOVV	p_len+16(FP), R6	// a2 = len(p)

	SGT	$8, R6, R12
	BNE	R12, less_than_8
	AND	$7, R5, R12
	BEQ	R12, aligned

	// Process the first few bytes to 8-byte align the input.
	// t0 = 8 - t0. We need to process this many bytes to align.
	SUB	$1, R12
	XOR	$7, R12

	AND	$1, R12, R13
	BEQ	R13, align_2
	MOVB	(R5), R13
	CRCWBW	R4, R13, R4
	ADDV	$1, R5
	ADDV	$-1, R6

align_2:
	AND	$2, R12, R13
	BEQ	R13, align_4
	MOVH	(R5), R13
	CRCWHW	R4, R13, R4
	ADDV	$2, R5
	ADDV	$-2, R6

align_4:
	AND	$4, R12, R13
	BEQ	R13, aligned
	MOVW	(R5), R13
	CRCWWW	R4, R13, R4
	ADDV	$4, R5
	ADDV	$-4, R6

aligned:
	// The input is now 8-byte aligned and we can process 8-byte chunks.
	SGT	$8, R6, R12
	BNE	R12, less_than_8
	MOVV	(R5), R13
	CRCWVW	R4, R13, R4
	ADDV	$8, R5
	ADDV	$-8, R6
	JMP	aligned

less_than_8:
	// We may have some bytes left over; process 4 bytes, then 2, then 1.
	AND	$4, R6, R12
	BEQ	R12, less_than_4
	MOVW	(R5), R13
	CRCWWW	R4, R13, R4
	ADDV	$4, R5
	ADDV	$-4, R6

less_than_4:
	AND	$2, R6, R12
	BEQ	R12, less_than_2
	MOVH	(R5), R13
	CRCWHW	R4, R13, R4
	ADDV	$2, R5
	ADDV	$-2, R6

less_than_2:
	BEQ	R6, done
	MOVB	(R5), R13
	CRCWBW	R4, R13, R4

done:
	MOVW	R4, ret+32(FP)
	RET

