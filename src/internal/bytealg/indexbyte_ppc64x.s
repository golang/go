// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ppc64 || ppc64le

#include "go_asm.h"
#include "textflag.h"

TEXT ·IndexByte<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-40
	// R3 = byte array pointer
	// R4 = length
	MOVD	R6, R5		// R5 = byte
	BR	indexbytebody<>(SB)

TEXT ·IndexByteString<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-32
	// R3 = string
	// R4 = length
	// R5 = byte
	BR	indexbytebody<>(SB)

#ifndef GOPPC64_power9
#ifdef GOARCH_ppc64le
DATA indexbytevbperm<>+0(SB)/8, $0x3830282018100800
DATA indexbytevbperm<>+8(SB)/8, $0x7870686058504840
#else
DATA indexbytevbperm<>+0(SB)/8, $0x0008101820283038
DATA indexbytevbperm<>+8(SB)/8, $0x4048505860687078
#endif
GLOBL indexbytevbperm<>+0(SB), RODATA, $16
#endif

// Some operations are endian specific, choose the correct opcode base on GOARCH.
// Note, _VCZBEBB is only available on power9 and newer.
#ifdef GOARCH_ppc64le
#define _LDBEX	MOVDBR
#define _LWBEX	MOVWBR
#define _LHBEX	MOVHBR
#define _VCZBEBB VCTZLSBB
#else
#define _LDBEX	MOVD
#define _LWBEX	MOVW
#define _LHBEX	MOVH
#define _VCZBEBB VCLZLSBB
#endif

// R3 = addr of string
// R4 = len of string
// R5 = byte to find
// On exit:
// R3 = return value
TEXT indexbytebody<>(SB),NOSPLIT|NOFRAME,$0-0
	CMPU	R4,$32

#ifndef GOPPC64_power9
	// Load VBPERMQ constant to reduce compare into an ordered bit mask.
	MOVD	$indexbytevbperm<>+00(SB),R16
	LXVD2X	(R16),V0	// Set up swap string
#endif

	MTVRD	R5,V1
	VSPLTB	$7,V1,V1	// Replicate byte across V1

	BLT	cmp16		// Jump to the small string case if it's <32 bytes.

	CMP	R4,$64,CR1
	MOVD	$16,R11
	MOVD	R3,R8
	BLT	CR1,cmp32	// Special case for length 32 - 63
	MOVD	$32,R12
	MOVD	$48,R6

	RLDICR  $0,R4,$63-6,R9	// R9 = len &^ 63
	ADD	R3,R9,R9	// R9 = &s[len &^ 63]
	ANDCC	$63,R4		// (len &= 63) cmp 0.

	PCALIGN	$16
loop64:
	LXVD2X	(R0)(R8),V2	// Scan 64 bytes at a time, starting at &s[0]
	VCMPEQUBCC	V2,V1,V6
	BNE	CR6,foundat0	// Match found at R8, jump out

	LXVD2X	(R11)(R8),V2
	VCMPEQUBCC	V2,V1,V6
	BNE	CR6,foundat1	// Match found at R8+16 bytes, jump out

	LXVD2X	(R12)(R8),V2
	VCMPEQUBCC	V2,V1,V6
	BNE	CR6,foundat2	// Match found at R8+32 bytes, jump out

	LXVD2X	(R6)(R8),V2
	VCMPEQUBCC	V2,V1,V6
	BNE	CR6,foundat3	// Match found at R8+48 bytes, jump out

	ADD	$64,R8
	CMPU	R8,R9,CR1
	BNE	CR1,loop64	// R8 != &s[len &^ 63]?

	PCALIGN	$32
	BEQ	notfound	// Is tail length 0? CR0 is set before entering loop64.

	CMP	R4,$32		// Tail length >= 32, use cmp32 path.
	CMP	R4,$16,CR1
	BGE	cmp32

	ADD	R8,R4,R9
	ADD	$-16,R9
	BLE	CR1,cmp64_tail_gt0

cmp64_tail_gt16:	// Tail length 17 - 32
	LXVD2X	(R0)(R8),V2
	VCMPEQUBCC	V2,V1,V6
	BNE	CR6,foundat0

cmp64_tail_gt0:	// Tail length 1 - 16
	MOVD	R9,R8
	LXVD2X	(R0)(R9),V2
	VCMPEQUBCC	V2,V1,V6
	BNE	CR6,foundat0

	BR	notfound

cmp32:	// Length 32 - 63

	// Bytes 0 - 15
	LXVD2X	(R0)(R8),V2
	VCMPEQUBCC	V2,V1,V6
	BNE	CR6,foundat0

	// Bytes 16 - 31
	LXVD2X	(R8)(R11),V2
	VCMPEQUBCC	V2,V1,V6
	BNE	CR6,foundat1		// Match found at R8+16 bytes, jump out

	BEQ	notfound		// Is length <= 32? (CR0 holds this comparison on entry to cmp32)
	CMP	R4,$48

	ADD	R4,R8,R9		// Compute &s[len(s)-16]
	ADD	$32,R8,R8
	ADD	$-16,R9,R9
	ISEL	CR0GT,R8,R9,R8		// R8 = len(s) <= 48 ? R9 : R8

	// Bytes 33 - 47
	LXVD2X	(R0)(R8),V2
	VCMPEQUBCC	V2,V1,V6
	BNE	CR6,foundat0		// match found at R8+32 bytes, jump out

	BLE	notfound

	// Bytes 48 - 63
	MOVD	R9,R8			// R9 holds the final check.
	LXVD2X	(R0)(R9),V2
	VCMPEQUBCC	V2,V1,V6
	BNE	CR6,foundat0		// Match found at R8+48 bytes, jump out

	BR	notfound

// If ISA 3.0 instructions are unavailable, we need to account for the extra 16 added by CNTLZW.
#ifndef GOPPC64_power9
#define ADJUST_FOR_CNTLZW -16
#else
#define ADJUST_FOR_CNTLZW 0
#endif

// Now, find the index of the 16B vector the match was discovered in. If CNTLZW is used
// to determine the offset into the 16B vector, it will overcount by 16. Account for it here.
foundat3:
	SUB	R3,R8,R3
	ADD	$48+ADJUST_FOR_CNTLZW,R3
	BR	vfound
foundat2:
	SUB	R3,R8,R3
	ADD	$32+ADJUST_FOR_CNTLZW,R3
	BR	vfound
foundat1:
	SUB	R3,R8,R3
	ADD	$16+ADJUST_FOR_CNTLZW,R3
	BR	vfound
foundat0:
	SUB	R3,R8,R3
	ADD	$0+ADJUST_FOR_CNTLZW,R3
vfound:
	// Map equal values into a 16 bit value with earlier matches setting higher bits.
#ifndef GOPPC64_power9
	VBPERMQ	V6,V0,V6
	MFVRD	V6,R4
	CNTLZW	R4,R4
#else
#ifdef GOARCH_ppc64le
	// Put the value back into LE ordering by swapping doublewords.
	XXPERMDI	V6,V6,$2,V6
#endif
	_VCZBEBB	V6,R4
#endif
	ADD	R3,R4,R3
	RET

cmp16:	// Length 16 - 31
	CMPU	R4,$16
	ADD	R4,R3,R9
	BLT	cmp8

	ADD	$-16,R9,R9		// &s[len(s)-16]

	// Bytes 0 - 15
	LXVD2X	(R0)(R3),V2
	VCMPEQUBCC	V2,V1,V6
	MOVD	R3,R8
	BNE	CR6,foundat0		// Match found at R8+32 bytes, jump out

	BEQ	notfound

	// Bytes 16 - 30
	MOVD	R9,R8			// R9 holds the final check.
	LXVD2X	(R0)(R9),V2
	VCMPEQUBCC	V2,V1,V6
	BNE	CR6,foundat0		// Match found at R8+48 bytes, jump out

	BR	notfound


cmp8:	// Length 8 - 15
#ifdef GOPPC64_power10
	// Load all the bytes into a single VSR in BE order.
	SLD	$56,R4,R5
	LXVLL	R3,R5,V2
	// Compare and count the number which don't match.
	VCMPEQUB	V2,V1,V6
	VCLZLSBB	V6,R3
	// If count is the number of bytes, or more. No matches are found.
	CMPU	R3,R4
	MOVD	$-1,R5
	// Otherwise, the count is the index of the first match.
	ISEL	CR0LT,R3,R5,R3
	RET
#else
	RLDIMI	$8,R5,$48,R5	// Replicating the byte across the register.
	RLDIMI	$16,R5,$32,R5
	RLDIMI	$32,R5,$0,R5
	CMPU	R4,$8
	BLT	cmp4
	MOVD	$-8,R11
	ADD	$-8,R4,R4

	_LDBEX	(R0)(R3),R10
	_LDBEX	(R11)(R9),R11
	CMPB	R10,R5,R10
	CMPB	R11,R5,R11
	CMPU	R10,$0
	CMPU	R11,$0,CR1
	CNTLZD	R10,R10
	CNTLZD	R11,R11
	SRD	$3,R10,R3
	SRD	$3,R11,R11
	BNE	found

	ADD	R4,R11,R4
	MOVD	$-1,R3
	ISEL	CR1EQ,R3,R4,R3
	RET

cmp4:	// Length 4 - 7
	CMPU	R4,$4
	BLT	cmp2
	MOVD	$-4,R11
	ADD	$-4,R4,R4

	_LWBEX	(R0)(R3),R10
	_LWBEX	(R11)(R9),R11
	CMPB	R10,R5,R10
	CMPB	R11,R5,R11
	CNTLZW	R10,R10
	CNTLZW	R11,R11
	CMPU	R10,$32
	CMPU	R11,$32,CR1
	SRD	$3,R10,R3
	SRD	$3,R11,R11
	BNE	found

	ADD	R4,R11,R4
	MOVD	$-1,R3
	ISEL	CR1EQ,R3,R4,R3
	RET

cmp2:	// Length 2 - 3
	CMPU	R4,$2
	BLT	cmp1

	_LHBEX	(R0)(R3),R10
	CMPB	R10,R5,R10
	SLDCC	$48,R10,R10
	CNTLZD	R10,R10
	SRD	$3,R10,R3
	BNE	found

cmp1:	// Length 1
	MOVD	$-1,R3
	ANDCC	$1,R4,R31
	BEQ	found

	MOVBZ	-1(R9),R10
	CMPB	R10,R5,R10
	ANDCC	$1,R10
	ADD	$-1,R4
	ISEL	CR0EQ,R3,R4,R3

found:
	RET
#endif

notfound:
	MOVD $-1,R3
	RET

