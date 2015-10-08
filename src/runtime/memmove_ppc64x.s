// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ppc64 ppc64le

#include "textflag.h"

// void runtime·memmove(void*, void*, uintptr)
TEXT runtime·memmove(SB), NOSPLIT|NOFRAME, $0-24
	MOVD	to+0(FP), R3
	MOVD	from+8(FP), R4
	MOVD	n+16(FP), R5
	CMP	R5, $0
	BNE	check
	RET

check:
	ANDCC	$7, R5, R7	// R7 is the number of bytes to copy and CR0[EQ] is set if there are none.
	SRAD	$3, R5, R6	// R6 is the number of words to copy
	CMP	R6, $0, CR1	// CR1[EQ] is set if there are no words to copy.

	CMP	R3, R4, CR2
	BC	12, 9, backward	// I think you should be able to write this as "BGT CR2, backward"

	// Copying forward proceeds by copying R6 words then copying R7 bytes.
	// R3 and R4 are advanced as we copy. Becuase PPC64 lacks post-increment
	// load/store, R3 and R4 point before the bytes that are to be copied.

	BC	12, 6, noforwardlarge	// "BEQ CR1, noforwardlarge"

	MOVD	R6, CTR

	SUB	$8, R3
	SUB	$8, R4

forwardlargeloop:
	MOVDU	8(R4), R8
	MOVDU	R8, 8(R3)
	BC	16, 0, forwardlargeloop // "BDNZ"

	ADD	$8, R3
	ADD	$8, R4

noforwardlarge:
	BNE	forwardtail	// Tests the bit set by ANDCC above
	RET

forwardtail:
	SUB	$1, R3
	SUB	$1, R4
	MOVD	R7, CTR

forwardtailloop:
	MOVBZU	1(R4), R8
	MOVBZU	R8, 1(R3)
	BC	16, 0, forwardtailloop
	RET

backward:
	// Copying backwards proceeds by copying R7 bytes then copying R6 words.
	// R3 and R4 are advanced to the end of the destination/source buffers
	// respectively and moved back as we copy.

	ADD	R5, R4, R4
	ADD	R3, R5, R3

	BEQ	nobackwardtail

	MOVD	R7, CTR

backwardtailloop:
	MOVBZU	-1(R4), R8
	MOVBZU	R8, -1(R3)
	BC	16, 0, backwardtailloop

nobackwardtail:
	BC	4, 6, backwardlarge		// "BNE CR1"
	RET

backwardlarge:
	MOVD	R6, CTR

backwardlargeloop:
	MOVDU	-8(R4), R8
	MOVDU	R8, -8(R3)
	BC	16, 0, backwardlargeloop	// "BDNZ"
	RET
