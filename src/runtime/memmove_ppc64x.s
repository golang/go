// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ppc64 || ppc64le

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
// number of 64 byte chunks
#define QWORDS R10
// index values
#define IDX32 R14
#define IDX48 R15
#define OCTWORDS R16

TEXT runtime·memmove<ABIInternal>(SB), NOSPLIT|NOFRAME, $0-24
	// R3 = TGT = to
	// R4 = SRC = from
	// R5 = LEN = n

	// Determine if there are doublewords to
	// copy so a more efficient move can be done
check:
#ifdef GOPPC64_power10
	CMP	LEN, $16
	BGT	mcopy
	SLD	$56, LEN, TMP
	LXVL	SRC, TMP, V0
	STXVL	V0, TGT, TMP
	RET
#endif
mcopy:
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
	BLT	CR2, backward

	// Copying forward if no overlap.

	BEQ	CR1, checkbytes
	SRDCC	$3, DWORDS, OCTWORDS	// 64 byte chunks?
	MOVD	$16, IDX16
	BEQ	lt64gt8			// < 64 bytes

	// Prepare for moves of 64 bytes at a time.

forward64setup:
	DCBTST	(TGT)			// prepare data cache
	DCBT	(SRC)
	MOVD	OCTWORDS, CTR		// Number of 64 byte chunks
	MOVD	$32, IDX32
	MOVD	$48, IDX48
	PCALIGN	$16

forward64:
	LXVD2X	(R0)(SRC), VS32		// load 64 bytes
	LXVD2X	(IDX16)(SRC), VS33
	LXVD2X	(IDX32)(SRC), VS34
	LXVD2X	(IDX48)(SRC), VS35
	ADD	$64, SRC
	STXVD2X	VS32, (R0)(TGT)		// store 64 bytes
	STXVD2X	VS33, (IDX16)(TGT)
	STXVD2X	VS34, (IDX32)(TGT)
	STXVD2X VS35, (IDX48)(TGT)
	ADD	$64,TGT			// bump up for next set
	BC	16, 0, forward64	// continue
	ANDCC	$7, DWORDS		// remaining doublewords
	BEQ	checkbytes		// only bytes remain

lt64gt8:
	CMP	DWORDS, $4
	BLT	lt32gt8
	LXVD2X	(R0)(SRC), VS32
	LXVD2X	(IDX16)(SRC), VS33
	ADD	$-4, DWORDS
	STXVD2X	VS32, (R0)(TGT)
	STXVD2X	VS33, (IDX16)(TGT)
	ADD	$32, SRC
	ADD	$32, TGT

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
#ifdef GOPPC64_power10
	ADD	$8, BYTES
	SLD	$56, BYTES, TMP
	LXVL	SRC, TMP, V0
	STXVL	V0, TGT, TMP
	RET
#endif

	MOVD    0(SRC), TMP
	ADD	$8, SRC
	MOVD    TMP, 0(TGT)
	ADD     $8, TGT
checkbytes:
	BEQ	CR3, LR
#ifdef GOPPC64_power10
	SLD	$56, BYTES, TMP
	LXVL	SRC, TMP, V0
	STXVL	V0, TGT, TMP
	RET
#endif
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
	BLT CR0, LR
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
	BDNZ	backwardtailloop

nobackwardtail:
	BLE	CR1, LR                 // return if DWORDS == 0
	SRDCC	$2,DWORDS,QWORDS	// Compute number of 32B blocks and compare to 0
	BNE	backward32setup		// If QWORDS != 0, start the 32B copy loop.

backward24:
	// DWORDS is a value between 1-3.
	CMP	DWORDS, $2

	MOVD	-8(SRC), TMP
	MOVD	TMP, -8(TGT)
	BLT	CR0, LR                 // return if DWORDS == 1

	MOVD	-16(SRC), TMP
	MOVD	TMP, -16(TGT)
	BEQ	CR0, LR                 // return if DWORDS == 2

	MOVD	-24(SRC), TMP
	MOVD	TMP, -24(TGT)
	RET

backward32setup:
	ANDCC   $3,DWORDS		// Compute remaining DWORDS and compare to 0
	MOVD	QWORDS, CTR		// set up loop ctr
	MOVD	$16, IDX16		// 32 bytes at a time
	PCALIGN	$16

backward32loop:
	SUB	$32, TGT
	SUB	$32, SRC
	LXVD2X	(R0)(SRC), VS32		// load 16x2 bytes
	LXVD2X	(IDX16)(SRC), VS33
	STXVD2X	VS32, (R0)(TGT)		// store 16x2 bytes
	STXVD2X	VS33, (IDX16)(TGT)
	BDNZ	backward32loop
	BEQ	CR0, LR                 // return if DWORDS == 0
	BR	backward24
