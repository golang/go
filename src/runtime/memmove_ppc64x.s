// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ppc64 ppc64le

#include "textflag.h"

// See memmove Go doc for important implementation constraints.

// func memmove(to, from unsafe.Pointer, n uintptr)

// target address
#define TGT R3
// source address
#define SRC R4
// length to move
#define LEN R5
// number of doublewords
#define DWORDS R6
// number of bytes < 8
#define BYTES R7
// const 16 used as index
#define IDX16 R8
// temp used for copies, etc.
#define TMP R9
// number of 32 byte chunks
#define QWORDS R10

TEXT runtime·memmove(SB), NOSPLIT|NOFRAME, $0-24
	MOVD	to+0(FP), TGT
	MOVD	from+8(FP), SRC
	MOVD	n+16(FP), LEN

	// Determine if there are doublewords to
	// copy so a more efficient move can be done
check:
	ANDCC	$7, LEN, BYTES	// R7: bytes to copy
	SRD	$3, LEN, DWORDS	// R6: double words to copy
	MOVFL	CR0, CR3	// save CR from ANDCC
	CMP	DWORDS, $0, CR1	// CR1[EQ] set if no double words to copy

	// Determine overlap by subtracting dest - src and comparing against the
	// length.  This catches the cases where src and dest are in different types
	// of storage such as stack and static to avoid doing backward move when not
	// necessary.

	SUB	SRC, TGT, TMP	// dest - src
	CMPU	TMP, LEN, CR2	// < len?
	BC	12, 8, backward // BLT CR2 backward

	// Copying forward if no overlap.

	BC	12, 6, checkbytes	// BEQ CR1, checkbytes
	SRDCC	$2, DWORDS, QWORDS	// 32 byte chunks?
	BEQ	lt32gt8			// < 32 bytes

	// Prepare for moves of 32 bytes at a time.

forward32setup:
	DCBTST	(TGT)			// prepare data cache
	DCBT	(SRC)
	MOVD	QWORDS, CTR		// Number of 32 byte chunks
	MOVD	$16, IDX16		// 16 for index

forward32:
	LXVD2X	(R0)(SRC), VS32		// load 16 bytes
	LXVD2X	(IDX16)(SRC), VS33	// load 16 bytes
	ADD	$32, SRC
	STXVD2X	VS32, (R0)(TGT)		// store 16 bytes
	STXVD2X	VS33, (IDX16)(TGT)
	ADD	$32,TGT			// bump up for next set
	BC	16, 0, forward32	// continue
	ANDCC	$3, DWORDS		// remaining doublewords
	BEQ	checkbytes		// only bytes remain

lt32gt8:
        // At this point >= 8 and < 32
	// Move 16 bytes if possible
	CMP     DWORDS, $2
	BLT     lt16
	LXVD2X	(R0)(SRC), VS32
	ADD	$-2, DWORDS
	STXVD2X	VS32, (R0)(TGT)
	ADD     $16, SRC
	ADD     $16, TGT

lt16:	// Move 8 bytes if possible
	CMP     DWORDS, $1
	BLT     checkbytes
	MOVD    0(SRC), TMP
	ADD	$8, SRC
	MOVD    TMP, 0(TGT)
	ADD     $8, TGT
checkbytes:
	BC	12, 14, LR		// BEQ lr
lt8:	// Move word if possible
	CMP BYTES, $4
	BLT lt4
	MOVWZ 0(SRC), TMP
	ADD $-4, BYTES
	MOVW TMP, 0(TGT)
	ADD $4, SRC
	ADD $4, TGT
lt4:	// Move halfword if possible
	CMP BYTES, $2
	BLT lt2
	MOVHZ 0(SRC), TMP
	ADD $-2, BYTES
	MOVH TMP, 0(TGT)
	ADD $2, SRC
	ADD $2, TGT
lt2:	// Move last byte if 1 left
	CMP BYTES, $1
	BC 12, 0, LR	// ble lr
	MOVBZ 0(SRC), TMP
	MOVBZ TMP, 0(TGT)
	RET

backward:
	// Copying backwards proceeds by copying R7 bytes then copying R6 double words.
	// R3 and R4 are advanced to the end of the destination/source buffers
	// respectively and moved back as we copy.

	ADD	LEN, SRC, SRC		// end of source
	ADD	TGT, LEN, TGT		// end of dest

	BEQ	nobackwardtail		// earlier condition

	MOVD	BYTES, CTR			// bytes to move

backwardtailloop:
	MOVBZ 	-1(SRC), TMP		// point to last byte
	SUB	$1,SRC
	MOVBZ 	TMP, -1(TGT)
	SUB	$1,TGT
	BC	16, 0, backwardtailloop // bndz

nobackwardtail:
	BC	4, 5, LR		// ble CR1 lr

backwardlarge:
	MOVD	DWORDS, CTR
	SUB	TGT, SRC, TMP		// Use vsx if moving
	CMP	TMP, $32		// at least 32 byte chunks
	BLT	backwardlargeloop	// and distance >= 32
	SRDCC	$2,DWORDS,QWORDS	// 32 byte chunks
	BNE	backward32setup

backwardlargeloop:
	MOVD 	-8(SRC), TMP
	SUB	$8,SRC
	MOVD 	TMP, -8(TGT)
	SUB	$8,TGT
	BC	16, 0, backwardlargeloop // bndz
	RET

backward32setup:
	MOVD	QWORDS, CTR			// set up loop ctr
	MOVD	$16, IDX16			// 32 bytes at at time

backward32loop:
	SUB	$32, TGT
	SUB	$32, SRC
	LXVD2X	(R0)(TGT), VS32           // load 16 bytes
	LXVD2X	(IDX16)(TGT), VS33
	STXVD2X	VS32, (R0)(SRC)           // store 16 bytes
	STXVD2X	VS33, (IDX16)(SRC)
	BC      16, 0, backward32loop   // bndz
	BC	4, 5, LR		// ble CR1 lr
	MOVD	DWORDS, CTR
	BR	backwardlargeloop
