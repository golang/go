// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// void runtime·memclr(void*, uintptr)
TEXT runtime·memclr(SB),NOSPLIT,$0-16
	MOVD	ptr+0(FP), R3
	MOVD	n+8(FP), R4
	// TODO(mwhudson): this is written this way to avoid tickling
	// warnings from addpool when written as AND $7, R4, R6 (see
	// https://golang.org/issue/12708)
	AND	$~7, R4, R5	// R5 is N&~7
	SUB	R5, R4, R6	// R6 is N&7

	CMP	$0, R5
	BEQ	nowords

	ADD	R3, R5, R5

wordloop: // TODO: Optimize for unaligned ptr.
	MOVD.P	$0, 8(R3)
	CMP	R3, R5
	BNE	wordloop
nowords:
        CMP	$0, R6
        BEQ	done

	ADD	R3, R6, R6

byteloop:
	MOVBU.P	$0, 1(R3)
	CMP	R3, R6
	BNE	byteloop
done:
	RET
