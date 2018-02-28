// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build s390x,!gccgo,!appengine

#include "textflag.h"

// func hasHWSupport() bool
TEXT ·hasHWSupport(SB),NOSPLIT,$16-1
	XOR	R0, R0          // set function code to 0 (query)
	LA	mask-16(SP), R1 // 16-byte stack variable for mask
	MOVD	$(0x38<<40), R3 // mask for bits 18-20 (big endian)

	// check for KM AES functions
	WORD	$0xB92E0024 // cipher message (KM)
	MOVD	mask-16(SP), R2
	AND	R3, R2
	CMPBNE	R2, R3, notfound

	// check for KMC AES functions
	WORD	$0xB92F0024 // cipher message with chaining (KMC)
	MOVD	mask-16(SP), R2
	AND	R3, R2
	CMPBNE	R2, R3, notfound

	// check for KMCTR AES functions
	WORD	$0xB92D4024 // cipher message with counter (KMCTR)
	MOVD	mask-16(SP), R2
	AND	R3, R2
	CMPBNE	R2, R3, notfound

	// check for KIMD GHASH function
	WORD	$0xB93E0024    // compute intermediate message digest (KIMD)
	MOVD	mask-8(SP), R2 // bits 64-127
	MOVD	$(1<<62), R5
	AND	R5, R2
	CMPBNE	R2, R5, notfound

	MOVB	$1, ret+0(FP)
	RET
notfound:
	MOVB	$0, ret+0(FP)
	RET
