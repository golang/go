// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ppc64 ppc64le

#include "textflag.h"

// void runtime·memclrNoHeapPointers(void*, uintptr)
TEXT runtime·memclrNoHeapPointers(SB), NOSPLIT|NOFRAME, $0-16
	MOVD ptr+0(FP), R3
	MOVD n+8(FP), R4

	// Determine if there are doublewords to clear
check:
	ANDCC $7, R4, R5  // R5: leftover bytes to clear
	SRAD  $3, R4, R6  // R6: double words to clear
	CMP   R6, $0, CR1 // CR1[EQ] set if no double words

	BC     12, 6, nozerolarge // only single bytes
	MOVD   R6, CTR            // R6 = number of double words
	SRADCC $2, R6, R7         // 32 byte chunks?
	BNE    zero32setup

	// Clear double words

zero8:
	MOVD R0, 0(R3)    // double word
	ADD  $8, R3
	BC   16, 0, zero8 // dec ctr, br zero8 if ctr not 0
	BR   nozerolarge  // handle remainder

	// Prepare to clear 32 bytes at a time.

zero32setup:
	DCBTST (R3)    // prepare data cache
	MOVD   R7, CTR // number of 32 byte chunks

zero32:
	MOVD    R0, 0(R3)       // clear 4 double words
	MOVD    R0, 8(R3)
	MOVD    R0, 16(R3)
	MOVD    R0, 24(R3)
	ADD     $32, R3
	BC      16, 0, zero32   // dec ctr, br zero32 if ctr not 0
	RLDCLCC $61, R4, $3, R6 // remaining doublewords
	BEQ     nozerolarge
	MOVD    R6, CTR         // set up the CTR for doublewords
	BR      zero8

nozerolarge:
	CMP R5, $0   // any remaining bytes
	BC  4, 1, LR // ble lr

zerotail:
	MOVD R5, CTR // set up to clear tail bytes

zerotailloop:
	MOVB R0, 0(R3)           // clear single bytes
	ADD  $1, R3
	BC   16, 0, zerotailloop // dec ctr, br zerotailloop if ctr not 0
	RET
