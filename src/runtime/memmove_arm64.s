// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func memmove(to, from unsafe.Pointer, n uintptr)
TEXT runtime·memmove(SB), NOSPLIT|NOFRAME, $0-24
	MOVD	to+0(FP), R3
	MOVD	from+8(FP), R4
	MOVD	n+16(FP), R5
	CBNZ	R5, check
	RET

check:
	CMP	$16, R5
	BLE	copy16

	AND	$~31, R5, R7	// R7 is N&~31
	SUB	R7, R5, R6	// R6 is N&31

	CMP	R3, R4
	BLT	backward

	// Copying forward proceeds by copying R7/8 words then copying R6 bytes.
	// R3 and R4 are advanced as we copy.

        // (There may be implementations of armv8 where copying by bytes until
        // at least one of source or dest is word aligned is a worthwhile
        // optimization, but the on the one tested so far (xgene) it did not
        // make a significance difference.)

	CBZ	R7, noforwardlarge	// Do we need to do any doubleword-by-doubleword copying?

	ADD	R3, R7, R9	// R9 points just past where we copy by word

forwardlargeloop:
	LDP.P	32(R4), (R8, R10)
	STP.P	(R8, R10), 32(R3)
	LDP	-16(R4), (R11, R12)
	STP	(R11, R12), -16(R3)
	SUB 	$32, R7, R7
	CBNZ	R7, forwardlargeloop

noforwardlarge:
	CBNZ	R6, forwardtail		// Do we need to do any byte-by-byte copying?
	RET

forwardtail:
	ADD	R3, R6, R9	// R9 points just past the destination memory

forwardtailloop:
	MOVBU.P 1(R4), R8
	MOVBU.P	R8, 1(R3)
	CMP	R3, R9
	BNE	forwardtailloop
	RET

	// Small copies: 1..16 bytes.
copy16:
	ADD	R4, R5, R8	// R8 points just past the last source byte
	ADD	R3, R5, R9	// R9 points just past the last destination byte
	CMP	$8, R5
	BLT	copy7
	MOVD	(R4), R6
	MOVD	-8(R8), R7
	MOVD	R6, (R3)
	MOVD	R7, -8(R9)
	RET

copy7:
	TBZ	$2, R5, copy3
	MOVWU	(R4), R6
	MOVWU	-4(R8), R7
	MOVW	R6, (R3)
	MOVW	R7, -4(R9)
	RET

copy3:
	TBZ	$1, R5, copy1
	MOVHU	(R4), R6
	MOVHU	-2(R8), R7
	MOVH	R6, (R3)
	MOVH	R7, -2(R9)
	RET

copy1:
	MOVBU	(R4), R6
	MOVB	R6, (R3)
	RET

backward:
	// Copying backwards proceeds by copying R6 bytes then copying R7/8 words.
	// R3 and R4 are advanced to the end of the destination/source buffers
	// respectively and moved back as we copy.

	ADD	R4, R5, R4	// R4 points just past the last source byte
	ADD	R3, R5, R3	// R3 points just past the last destination byte

	CBZ	R6, nobackwardtail	// Do we need to do any byte-by-byte copying?

	SUB	R6, R3, R9	// R9 points at the lowest destination byte that should be copied by byte.
backwardtailloop:
	MOVBU.W	-1(R4), R8
	MOVBU.W	R8, -1(R3)
	CMP	R9, R3
	BNE	backwardtailloop

nobackwardtail:
	CBNZ     R7, backwardlarge	// Do we need to do any doubleword-by-doubleword copying?
	RET

backwardlarge:
        SUB	R7, R3, R9      // R9 points at the lowest destination byte

backwardlargeloop:
	LDP	-16(R4), (R8, R10)
	STP	(R8, R10), -16(R3)
	LDP.W	-32(R4), (R11, R12)
	STP.W	(R11, R12), -32(R3)
	CMP	R9, R3
	BNE	backwardlargeloop
	RET
