// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// void runtime·memmove(void*, void*, uintptr)
TEXT runtime·memmove(SB), NOSPLIT, $-8-24
	MOVD	to+0(FP), R3
	MOVD	from+8(FP), R4
	MOVD	n+16(FP), R5
	CMP	$0, R5
	BNE	check
	RET

check:
	AND	$~7, R5, R7	// R7 is N&~7
	// TODO(mwhudson): this is written this way to avoid tickling
	// warnings from addpool when written as AND $7, R5, R6 (see
	// https://golang.org/issue/12708)
	SUB	R7, R5, R6	// R6 is N&7

	CMP	R3, R4
	BLT	backward

	// Copying forward proceeds by copying R7/8 words then copying R6 bytes.
	// R3 and R4 are advanced as we copy.

        // (There may be implementations of armv8 where copying by bytes until
        // at least one of source or dest is word aligned is a worthwhile
        // optimization, but the on the one tested so far (xgene) it did not
        // make a significance difference.)

	CMP	$0, R7		// Do we need to do any word-by-word copying?
	BEQ	noforwardlarge

	ADD	R3, R7, R9	// R9 points just past where we copy by word

forwardlargeloop:
	MOVD.P	8(R4), R8	// R8 is just a scratch register
	MOVD.P	R8, 8(R3)
	CMP	R3, R9
	BNE	forwardlargeloop

noforwardlarge:
	CMP	$0, R6		// Do we need to do any byte-by-byte copying?
	BNE	forwardtail
	RET

forwardtail:
	ADD	R3, R6, R9	// R9 points just past the destination memory

forwardtailloop:
	MOVBU.P 1(R4), R8
	MOVBU.P	R8, 1(R3)
	CMP	R3, R9
	BNE	forwardtailloop
	RET

backward:
	// Copying backwards proceeds by copying R6 bytes then copying R7/8 words.
	// R3 and R4 are advanced to the end of the destination/source buffers
	// respectively and moved back as we copy.

	ADD	R4, R5, R4	// R4 points just past the last source byte
	ADD	R3, R5, R3	// R3 points just past the last destination byte

	CMP	$0, R6		// Do we need to do any byte-by-byte copying?
	BEQ	nobackwardtail

	SUB	R6, R3, R9	// R9 points at the lowest destination byte that should be copied by byte.
backwardtailloop:
	MOVBU.W	-1(R4), R8
	MOVBU.W	R8, -1(R3)
	CMP	R9, R3
	BNE	backwardtailloop

nobackwardtail:
	CMP     $0, R7		// Do we need to do any word-by-word copying?
	BNE	backwardlarge
	RET

backwardlarge:
        SUB	R7, R3, R9      // R9 points at the lowest destination byte

backwardlargeloop:
	MOVD.W	-8(R4), R8
	MOVD.W	R8, -8(R3)
	CMP	R9, R3
	BNE	backwardlargeloop
	RET
