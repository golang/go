// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ppc64 ppc64le

#include "textflag.h"

// See memclrNoHeapPointers Go doc for important implementation constraints.

// func memclrNoHeapPointers(ptr unsafe.Pointer, n uintptr)
TEXT runtime·memclrNoHeapPointers(SB), NOSPLIT|NOFRAME, $0-16
	MOVD ptr+0(FP), R3
	MOVD n+8(FP), R4

	// Determine if there are doublewords to clear
check:
	ANDCC $7, R4, R5  // R5: leftover bytes to clear
	SRD   $3, R4, R6  // R6: double words to clear
	CMP   R6, $0, CR1 // CR1[EQ] set if no double words

	BC    12, 6, nozerolarge // only single bytes
	CMP   R4, $512
	BLT   under512           // special case for < 512
	ANDCC $127, R3, R8       // check for 128 alignment of address
	BEQ   zero512setup

	ANDCC $7, R3, R15
	BEQ   zero512xsetup // at least 8 byte aligned

	// zero bytes up to 8 byte alignment

	ANDCC $1, R3, R15 // check for byte alignment
	BEQ   byte2
	MOVB  R0, 0(R3)   // zero 1 byte
	ADD   $1, R3      // bump ptr by 1
	ADD   $-1, R4

byte2:
	ANDCC $2, R3, R15 // check for 2 byte alignment
	BEQ   byte4
	MOVH  R0, 0(R3)   // zero 2 bytes
	ADD   $2, R3      // bump ptr by 2
	ADD   $-2, R4

byte4:
	ANDCC $4, R3, R15   // check for 4 byte alignment
	BEQ   zero512xsetup
	MOVW  R0, 0(R3)     // zero 4 bytes
	ADD   $4, R3        // bump ptr by 4
	ADD   $-4, R4
	BR    zero512xsetup // ptr should now be 8 byte aligned

under512:
	MOVD  R6, CTR     // R6 = number of double words
	SRDCC $2, R6, R7  // 32 byte chunks?
	BNE   zero32setup

	// Clear double words

zero8:
	MOVD R0, 0(R3)    // double word
	ADD  $8, R3
	ADD  $-8, R4
	BC   16, 0, zero8 // dec ctr, br zero8 if ctr not 0
	BR   nozerolarge  // handle leftovers

	// Prepare to clear 32 bytes at a time.

zero32setup:
	DCBTST (R3)             // prepare data cache
	XXLXOR VS32, VS32, VS32 // clear VS32 (V0)
	MOVD   R7, CTR          // number of 32 byte chunks
	MOVD   $16, R8

zero32:
	STXVD2X VS32, (R3+R0)   // store 16 bytes
	STXVD2X VS32, (R3+R8)
	ADD     $32, R3
	ADD     $-32, R4
	BC      16, 0, zero32   // dec ctr, br zero32 if ctr not 0
	RLDCLCC $61, R4, $3, R6 // remaining doublewords
	BEQ     nozerolarge
	MOVD    R6, CTR         // set up the CTR for doublewords
	BR      zero8

nozerolarge:
	ANDCC $7, R4, R5 // any remaining bytes
	BC    4, 1, LR   // ble lr

zerotail:
	MOVD R5, CTR // set up to clear tail bytes

zerotailloop:
	MOVB R0, 0(R3)           // clear single bytes
	ADD  $1, R3
	BC   16, 0, zerotailloop // dec ctr, br zerotailloop if ctr not 0
	RET

zero512xsetup:  // 512 chunk with extra needed
	ANDCC $8, R3, R11    // 8 byte alignment?
	BEQ   zero512setup16
	MOVD  R0, 0(R3)      // clear 8 bytes
	ADD   $8, R3         // update ptr to next 8
	ADD   $-8, R4        // dec count by 8

zero512setup16:
	ANDCC $127, R3, R14 // < 128 byte alignment
	BEQ   zero512setup  // handle 128 byte alignment
	MOVD  $128, R15
	SUB   R14, R15, R14 // find increment to 128 alignment
	SRD   $4, R14, R15  // number of 16 byte chunks

zero512presetup:
	MOVD   R15, CTR         // loop counter of 16 bytes
	XXLXOR VS32, VS32, VS32 // clear VS32 (V0)

zero512preloop:  // clear up to 128 alignment
	STXVD2X VS32, (R3+R0)         // clear 16 bytes
	ADD     $16, R3               // update ptr
	ADD     $-16, R4              // dec count
	BC      16, 0, zero512preloop

zero512setup:  // setup for dcbz loop
	CMP  R4, $512   // check if at least 512
	BLT  remain
	SRD  $9, R4, R8 // loop count for 512 chunks
	MOVD R8, CTR    // set up counter
	MOVD $128, R9   // index regs for 128 bytes
	MOVD $256, R10
	MOVD $384, R11

zero512:
	DCBZ (R3+R0)        // clear first chunk
	DCBZ (R3+R9)        // clear second chunk
	DCBZ (R3+R10)       // clear third chunk
	DCBZ (R3+R11)       // clear fourth chunk
	ADD  $512, R3
	ADD  $-512, R4
	BC   16, 0, zero512

remain:
	CMP  R4, $128  // check if 128 byte chunks left
	BLT  smaller
	DCBZ (R3+R0)   // clear 128
	ADD  $128, R3
	ADD  $-128, R4
	BR   remain

smaller:
	ANDCC $127, R4, R7 // find leftovers
	BEQ   done
	CMP   R7, $64      // more than 64, do 32 at a time
	BLT   zero8setup   // less than 64, do 8 at a time
	SRD   $5, R7, R7   // set up counter for 32
	BR    zero32setup

zero8setup:
	SRDCC $3, R7, R7  // less than 8 bytes
	BEQ   nozerolarge
	MOVD  R7, CTR
	BR    zero8

done:
	RET
