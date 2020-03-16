// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ppc64 ppc64le

#include "textflag.h"

// See memmove Go doc for important implementation constraints.

// func memmove(to, from unsafe.Pointer, n uintptr)
TEXT runtime·memmove(SB), NOSPLIT|NOFRAME, $0-24
	MOVD	to+0(FP), R3
	MOVD	from+8(FP), R4
	MOVD	n+16(FP), R5

	// Determine if there are doublewords to
	// copy so a more efficient move can be done
check:
	ANDCC	$7, R5, R7	// R7: bytes to copy
	SRD	$3, R5, R6	// R6: double words to copy
	CMP	R6, $0, CR1	// CR1[EQ] set if no double words to copy

	// Determine overlap by subtracting dest - src and comparing against the
	// length.  The catches the cases where src and dest are in different types
	// of storage such as stack and static to avoid doing backward move when not
	// necessary.

	SUB	R4, R3, R8	// dest - src
	CMPU	R8, R5, CR2	// < len?
	BC	12, 8, backward // BLT CR2 backward

	// Copying forward if no overlap.

	BC	12, 6, noforwardlarge	// "BEQ CR1, noforwardlarge"
	SRDCC	$2,R6,R8		// 32 byte chunks?
	BNE	forward32setup		//
	MOVD	R6,CTR			// R6 = number of double words

	// Move double words

forward8:
	MOVD    0(R4), R8		// double word
	ADD     $8,R4
	MOVD    R8, 0(R3)		//
	ADD     $8,R3
	BC      16, 0, forward8
	BR	noforwardlarge		// handle remainder

	// Prepare for moves of 32 bytes at a time.

forward32setup:
	DCBTST	(R3)			// prepare data cache
	DCBT	(R4)
	MOVD	R8, CTR			// double work count
	MOVD	$16, R8

forward32:
	LXVD2X	(R4+R0), VS32		// load 16 bytes
	LXVD2X	(R4+R8), VS33
	ADD	$32, R4
	STXVD2X	VS32, (R3+R0)		// store 16 bytes
	STXVD2X	VS33, (R3+R8)
	ADD	$32,R3			// bump up for next set
	BC	16, 0, forward32	// continue
	RLDCLCC	$61,R5,$3,R6		// remaining doublewords
	BEQ	noforwardlarge
	MOVD	R6,CTR			// set up the CTR
	BR	forward8

noforwardlarge:
	CMP	R7,$0			// any remaining bytes
	BC	4, 1, LR		// ble lr

forwardtail:
	MOVD	R7, CTR			// move tail bytes

forwardtailloop:
	MOVBZ	0(R4), R8		// move single bytes
	ADD	$1,R4
	MOVBZ	R8, 0(R3)
	ADD	$1,R3
	BC	16, 0, forwardtailloop
	RET

backward:
	// Copying backwards proceeds by copying R7 bytes then copying R6 double words.
	// R3 and R4 are advanced to the end of the destination/source buffers
	// respectively and moved back as we copy.

	ADD	R5, R4, R4		// end of source
	ADD	R3, R5, R3		// end of dest

	BEQ	nobackwardtail		// earlier condition

	MOVD	R7, CTR			// bytes to move

backwardtailloop:
	MOVBZ 	-1(R4), R8		// point to last byte
	SUB	$1,R4
	MOVBZ 	R8, -1(R3)
	SUB	$1,R3
	BC	16, 0, backwardtailloop // bndz

nobackwardtail:
	BC	4, 5, LR		// ble CR1 lr

backwardlarge:
	MOVD	R6, CTR
	SUB	R3, R4, R9		// Use vsx if moving
	CMP	R9, $32			// at least 32 byte chunks
	BLT	backwardlargeloop	// and distance >= 32
	SRDCC	$2,R6,R8		// 32 byte chunks
	BNE	backward32setup

backwardlargeloop:
	MOVD 	-8(R4), R8
	SUB	$8,R4
	MOVD 	R8, -8(R3)
	SUB	$8,R3
	BC	16, 0, backwardlargeloop // bndz
	RET

backward32setup:
	MOVD	R8, CTR			// set up loop ctr
	MOVD	$16, R8			// 32 bytes at at time

backward32loop:
	SUB	$32, R4
	SUB	$32, R3
	LXVD2X	(R4+R0), VS32           // load 16 bytes
	LXVD2X	(R4+R8), VS33
	STXVD2X	VS32, (R3+R0)           // store 16 bytes
	STXVD2X	VS33, (R3+R8)
	BC      16, 0, backward32loop   // bndz
	BC	4, 5, LR		// ble CR1 lr
	MOVD	R6, CTR
	BR	backwardlargeloop
