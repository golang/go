// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func hasKM() bool
TEXT ·hasKM(SB),NOSPLIT,$16-1
 	XOR	R0, R0          // set function code to 0 (query)
	LA	mask-16(SP), R1 // 16-byte stack variable for mask
	MOVD	$(0x38<<40), R3 // mask for bits 18-20 (big endian)

	// check for KM AES functions
	WORD	$0xB92E0024 // cipher message (KM)
	MOVD	mask-16(SP), R2
	AND	R3, R2
	CMPBNE	R2, R3, notfound

	MOVB	$1, ret+0(FP)
	RET
notfound:
	MOVB	$0, ret+0(FP)
	RET

// func hasKMC() bool
TEXT ·hasKMC(SB),NOSPLIT,$16-1
 	XOR	R0, R0          // set function code to 0 (query)
	LA	mask-16(SP), R1 // 16-byte stack variable for mask
	MOVD	$(0x38<<40), R3 // mask for bits 18-20 (big endian)

	// check for KMC AES functions
	WORD	$0xB92F0024 // cipher message with chaining (KMC)
	MOVD	mask-16(SP), R2
	AND	R3, R2
	CMPBNE	R2, R3, notfound

	MOVB	$1, ret+0(FP)
	RET
notfound:
	MOVB	$0, ret+0(FP)
	RET

// func hasKMCTR() bool
TEXT ·hasKMCTR(SB),NOSPLIT,$16-1
	XOR	R0, R0          // set function code to 0 (query)
	LA	mask-16(SP), R1 // 16-byte stack variable for mask
	MOVD	$(0x38<<40), R3 // mask for bits 18-20 (big endian)

	// check for KMCTR AES functions
	WORD	$0xB92D4024 // cipher message with counter (KMCTR)
	MOVD	mask-16(SP), R2
	AND	R3, R2
	CMPBNE	R2, R3, notfound

	MOVB	$1, ret+0(FP)
	RET
notfound:
	MOVB	$0, ret+0(FP)
	RET

// func hasKMA() bool
TEXT ·hasKMA(SB),NOSPLIT,$24-1
	MOVD	$tmp-24(SP), R1
	MOVD	$2, R0       // store 24-bytes
	XC	$24, (R1), (R1)
	WORD	$0xb2b01000  // STFLE (R1)
	MOVWZ	16(R1), R2
	ANDW	$(1<<13), R2 // test bit 146 (message-security-assist 8)
	BEQ	no

	MOVD	$0, R0       // KMA-Query
	XC	$16, (R1), (R1)
	WORD	$0xb9296024  // kma %r6,%r2,%r4
	MOVWZ	(R1), R2
	WORD	$0xa7213800  // TMLL R2, $0x3800
	BVS	yes
no:
	MOVB	$0, ret+0(FP)
	RET
yes:
	MOVB	$1, ret+0(FP)
	RET

// func hasKIMD() bool
TEXT ·hasKIMD(SB),NOSPLIT,$16-1
	XOR	R0, R0          // set function code to 0 (query)
	LA	mask-16(SP), R1 // 16-byte stack variable for mask
	MOVD	$(0x38<<40), R3 // mask for bits 18-20 (big endian)

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
