// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// ARM version of md5block.go

#include "textflag.h"

// Register definitions
#define Rtable	R0	// Pointer to MD5 constants table
#define Rdata	R1	// Pointer to data to hash
#define Ra	R2	// MD5 accumulator
#define Rb	R3	// MD5 accumulator
#define Rc	R4	// MD5 accumulator
#define Rd	R5	// MD5 accumulator
#define Rc0	R6	// MD5 constant
#define Rc1	R7	// MD5 constant
#define Rc2	R8	// MD5 constant
// r9, r10 are forbidden
// r11 is OK provided you check the assembler that no synthetic instructions use it
#define Rc3	R11	// MD5 constant
#define Rt0	R12	// temporary
#define Rt1	R14	// temporary

// func block(dig *digest, p []byte)
// 0(FP) is *digest
// 4(FP) is p.array (struct Slice)
// 8(FP) is p.len
//12(FP) is p.cap
//
// Stack frame
#define p_end	end-4(SP)	// pointer to the end of data
#define p_data	data-8(SP)	// current data pointer
#define buf	buffer-(8+4*16)(SP)	//16 words temporary buffer
		// 3 words at 4..12(R13) for called routine parameters

TEXT	·block(SB), NOSPLIT, $84-16
	MOVW	p+4(FP), Rdata	// pointer to the data
	MOVW	p_len+8(FP), Rt0	// number of bytes
	ADD	Rdata, Rt0
	MOVW	Rt0, p_end	// pointer to end of data

loop:
	MOVW	Rdata, p_data	// Save Rdata
	AND.S	$3, Rdata, Rt0	// TST $3, Rdata not working see issue 5921
	BEQ	aligned			// aligned detected - skip copy

	// Copy the unaligned source data into the aligned temporary buffer
	// memove(to=4(R13), from=8(R13), n=12(R13)) - Corrupts all registers
	MOVW	$buf, Rtable	// to
	MOVW	$64, Rc0		// n
	MOVM.IB	[Rtable,Rdata,Rc0], (R13)
	BL	runtime·memmove(SB)

	// Point to the local aligned copy of the data
	MOVW	$buf, Rdata

aligned:
	// Point to the table of constants
	// A PC relative add would be cheaper than this
	MOVW	$·table(SB), Rtable

	// Load up initial MD5 accumulator
	MOVW	dig+0(FP), Rc0
	MOVM.IA (Rc0), [Ra,Rb,Rc,Rd]

// a += (((c^d)&b)^d) + X[index] + const
// a = a<<shift | a>>(32-shift) + b
#define ROUND1(Ra, Rb, Rc, Rd, index, shift, Rconst) \
	EOR	Rc, Rd, Rt0		; \
	AND	Rb, Rt0			; \
	EOR	Rd, Rt0			; \
	MOVW	(index<<2)(Rdata), Rt1	; \
	ADD	Rt1, Rt0			; \
	ADD	Rconst, Rt0			; \
	ADD	Rt0, Ra			; \
	ADD	Ra@>(32-shift), Rb, Ra	;

	MOVM.IA.W (Rtable), [Rc0,Rc1,Rc2,Rc3]
	ROUND1(Ra, Rb, Rc, Rd,  0,	7, Rc0)
	ROUND1(Rd, Ra, Rb, Rc,  1, 12, Rc1)
	ROUND1(Rc, Rd, Ra, Rb,  2, 17, Rc2)
	ROUND1(Rb, Rc, Rd, Ra,  3, 22, Rc3)

	MOVM.IA.W (Rtable), [Rc0,Rc1,Rc2,Rc3]
	ROUND1(Ra, Rb, Rc, Rd,  4,	7, Rc0)
	ROUND1(Rd, Ra, Rb, Rc,  5, 12, Rc1)
	ROUND1(Rc, Rd, Ra, Rb,  6, 17, Rc2)
	ROUND1(Rb, Rc, Rd, Ra,  7, 22, Rc3)

	MOVM.IA.W (Rtable), [Rc0,Rc1,Rc2,Rc3]
	ROUND1(Ra, Rb, Rc, Rd,  8,	7, Rc0)
	ROUND1(Rd, Ra, Rb, Rc,  9, 12, Rc1)
	ROUND1(Rc, Rd, Ra, Rb, 10, 17, Rc2)
	ROUND1(Rb, Rc, Rd, Ra, 11, 22, Rc3)

	MOVM.IA.W (Rtable), [Rc0,Rc1,Rc2,Rc3]
	ROUND1(Ra, Rb, Rc, Rd, 12,	7, Rc0)
	ROUND1(Rd, Ra, Rb, Rc, 13, 12, Rc1)
	ROUND1(Rc, Rd, Ra, Rb, 14, 17, Rc2)
	ROUND1(Rb, Rc, Rd, Ra, 15, 22, Rc3)

// a += (((b^c)&d)^c) + X[index] + const
// a = a<<shift | a>>(32-shift) + b
#define ROUND2(Ra, Rb, Rc, Rd, index, shift, Rconst) \
	EOR	Rb, Rc, Rt0		; \
	AND	Rd, Rt0			; \
	EOR	Rc, Rt0			; \
	MOVW	(index<<2)(Rdata), Rt1	; \
	ADD	Rt1, Rt0			; \
	ADD	Rconst, Rt0			; \
	ADD	Rt0, Ra			; \
	ADD	Ra@>(32-shift), Rb, Ra	;

	MOVM.IA.W (Rtable), [Rc0,Rc1,Rc2,Rc3]
	ROUND2(Ra, Rb, Rc, Rd,  1,	5, Rc0)
	ROUND2(Rd, Ra, Rb, Rc,  6,	9, Rc1)
	ROUND2(Rc, Rd, Ra, Rb, 11, 14, Rc2)
	ROUND2(Rb, Rc, Rd, Ra,  0, 20, Rc3)

	MOVM.IA.W (Rtable), [Rc0,Rc1,Rc2,Rc3]
	ROUND2(Ra, Rb, Rc, Rd,  5,	5, Rc0)
	ROUND2(Rd, Ra, Rb, Rc, 10,	9, Rc1)
	ROUND2(Rc, Rd, Ra, Rb, 15, 14, Rc2)
	ROUND2(Rb, Rc, Rd, Ra,  4, 20, Rc3)

	MOVM.IA.W (Rtable), [Rc0,Rc1,Rc2,Rc3]
	ROUND2(Ra, Rb, Rc, Rd,  9,	5, Rc0)
	ROUND2(Rd, Ra, Rb, Rc, 14,	9, Rc1)
	ROUND2(Rc, Rd, Ra, Rb,  3, 14, Rc2)
	ROUND2(Rb, Rc, Rd, Ra,  8, 20, Rc3)

	MOVM.IA.W (Rtable), [Rc0,Rc1,Rc2,Rc3]
	ROUND2(Ra, Rb, Rc, Rd, 13,	5, Rc0)
	ROUND2(Rd, Ra, Rb, Rc,  2,	9, Rc1)
	ROUND2(Rc, Rd, Ra, Rb,  7, 14, Rc2)
	ROUND2(Rb, Rc, Rd, Ra, 12, 20, Rc3)

// a += (b^c^d) + X[index] + const
// a = a<<shift | a>>(32-shift) + b
#define ROUND3(Ra, Rb, Rc, Rd, index, shift, Rconst) \
	EOR	Rb, Rc, Rt0		; \
	EOR	Rd, Rt0			; \
	MOVW	(index<<2)(Rdata), Rt1	; \
	ADD	Rt1, Rt0			; \
	ADD	Rconst, Rt0			; \
	ADD	Rt0, Ra			; \
	ADD	Ra@>(32-shift), Rb, Ra	;

	MOVM.IA.W (Rtable), [Rc0,Rc1,Rc2,Rc3]
	ROUND3(Ra, Rb, Rc, Rd,  5,	4, Rc0)
	ROUND3(Rd, Ra, Rb, Rc,  8, 11, Rc1)
	ROUND3(Rc, Rd, Ra, Rb, 11, 16, Rc2)
	ROUND3(Rb, Rc, Rd, Ra, 14, 23, Rc3)

	MOVM.IA.W (Rtable), [Rc0,Rc1,Rc2,Rc3]
	ROUND3(Ra, Rb, Rc, Rd,  1,	4, Rc0)
	ROUND3(Rd, Ra, Rb, Rc,  4, 11, Rc1)
	ROUND3(Rc, Rd, Ra, Rb,  7, 16, Rc2)
	ROUND3(Rb, Rc, Rd, Ra, 10, 23, Rc3)

	MOVM.IA.W (Rtable), [Rc0,Rc1,Rc2,Rc3]
	ROUND3(Ra, Rb, Rc, Rd, 13,	4, Rc0)
	ROUND3(Rd, Ra, Rb, Rc,  0, 11, Rc1)
	ROUND3(Rc, Rd, Ra, Rb,  3, 16, Rc2)
	ROUND3(Rb, Rc, Rd, Ra,  6, 23, Rc3)

	MOVM.IA.W (Rtable), [Rc0,Rc1,Rc2,Rc3]
	ROUND3(Ra, Rb, Rc, Rd,  9,	4, Rc0)
	ROUND3(Rd, Ra, Rb, Rc, 12, 11, Rc1)
	ROUND3(Rc, Rd, Ra, Rb, 15, 16, Rc2)
	ROUND3(Rb, Rc, Rd, Ra,  2, 23, Rc3)

// a += (c^(b|^d)) + X[index] + const
// a = a<<shift | a>>(32-shift) + b
#define ROUND4(Ra, Rb, Rc, Rd, index, shift, Rconst) \
	MVN	Rd, Rt0			; \
	ORR	Rb, Rt0			; \
	EOR	Rc, Rt0			; \
	MOVW	(index<<2)(Rdata), Rt1	; \
	ADD	Rt1, Rt0			; \
	ADD	Rconst, Rt0			; \
	ADD	Rt0, Ra			; \
	ADD	Ra@>(32-shift), Rb, Ra	;

	MOVM.IA.W (Rtable), [Rc0,Rc1,Rc2,Rc3]
	ROUND4(Ra, Rb, Rc, Rd,  0,	6, Rc0)
	ROUND4(Rd, Ra, Rb, Rc,  7, 10, Rc1)
	ROUND4(Rc, Rd, Ra, Rb, 14, 15, Rc2)
	ROUND4(Rb, Rc, Rd, Ra,  5, 21, Rc3)

	MOVM.IA.W (Rtable), [Rc0,Rc1,Rc2,Rc3]
	ROUND4(Ra, Rb, Rc, Rd, 12,	6, Rc0)
	ROUND4(Rd, Ra, Rb, Rc,  3, 10, Rc1)
	ROUND4(Rc, Rd, Ra, Rb, 10, 15, Rc2)
	ROUND4(Rb, Rc, Rd, Ra,  1, 21, Rc3)

	MOVM.IA.W (Rtable), [Rc0,Rc1,Rc2,Rc3]
	ROUND4(Ra, Rb, Rc, Rd,  8,	6, Rc0)
	ROUND4(Rd, Ra, Rb, Rc, 15, 10, Rc1)
	ROUND4(Rc, Rd, Ra, Rb,  6, 15, Rc2)
	ROUND4(Rb, Rc, Rd, Ra, 13, 21, Rc3)

	MOVM.IA.W (Rtable), [Rc0,Rc1,Rc2,Rc3]
	ROUND4(Ra, Rb, Rc, Rd,  4,	6, Rc0)
	ROUND4(Rd, Ra, Rb, Rc, 11, 10, Rc1)
	ROUND4(Rc, Rd, Ra, Rb,  2, 15, Rc2)
	ROUND4(Rb, Rc, Rd, Ra,  9, 21, Rc3)

	MOVW	dig+0(FP), Rt0
	MOVM.IA (Rt0), [Rc0,Rc1,Rc2,Rc3]

	ADD	Rc0, Ra
	ADD	Rc1, Rb
	ADD	Rc2, Rc
	ADD	Rc3, Rd

	MOVM.IA [Ra,Rb,Rc,Rd], (Rt0)

	MOVW	p_data, Rdata
	MOVW	p_end, Rt0
	ADD	$64, Rdata
	CMP	Rt0, Rdata
	BLO	loop

	RET

// MD5 constants table

	// Round 1
	DATA	·table+0x00(SB)/4, $0xd76aa478
	DATA	·table+0x04(SB)/4, $0xe8c7b756
	DATA	·table+0x08(SB)/4, $0x242070db
	DATA	·table+0x0c(SB)/4, $0xc1bdceee
	DATA	·table+0x10(SB)/4, $0xf57c0faf
	DATA	·table+0x14(SB)/4, $0x4787c62a
	DATA	·table+0x18(SB)/4, $0xa8304613
	DATA	·table+0x1c(SB)/4, $0xfd469501
	DATA	·table+0x20(SB)/4, $0x698098d8
	DATA	·table+0x24(SB)/4, $0x8b44f7af
	DATA	·table+0x28(SB)/4, $0xffff5bb1
	DATA	·table+0x2c(SB)/4, $0x895cd7be
	DATA	·table+0x30(SB)/4, $0x6b901122
	DATA	·table+0x34(SB)/4, $0xfd987193
	DATA	·table+0x38(SB)/4, $0xa679438e
	DATA	·table+0x3c(SB)/4, $0x49b40821
	// Round 2
	DATA	·table+0x40(SB)/4, $0xf61e2562
	DATA	·table+0x44(SB)/4, $0xc040b340
	DATA	·table+0x48(SB)/4, $0x265e5a51
	DATA	·table+0x4c(SB)/4, $0xe9b6c7aa
	DATA	·table+0x50(SB)/4, $0xd62f105d
	DATA	·table+0x54(SB)/4, $0x02441453
	DATA	·table+0x58(SB)/4, $0xd8a1e681
	DATA	·table+0x5c(SB)/4, $0xe7d3fbc8
	DATA	·table+0x60(SB)/4, $0x21e1cde6
	DATA	·table+0x64(SB)/4, $0xc33707d6
	DATA	·table+0x68(SB)/4, $0xf4d50d87
	DATA	·table+0x6c(SB)/4, $0x455a14ed
	DATA	·table+0x70(SB)/4, $0xa9e3e905
	DATA	·table+0x74(SB)/4, $0xfcefa3f8
	DATA	·table+0x78(SB)/4, $0x676f02d9
	DATA	·table+0x7c(SB)/4, $0x8d2a4c8a
	// Round 3
	DATA	·table+0x80(SB)/4, $0xfffa3942
	DATA	·table+0x84(SB)/4, $0x8771f681
	DATA	·table+0x88(SB)/4, $0x6d9d6122
	DATA	·table+0x8c(SB)/4, $0xfde5380c
	DATA	·table+0x90(SB)/4, $0xa4beea44
	DATA	·table+0x94(SB)/4, $0x4bdecfa9
	DATA	·table+0x98(SB)/4, $0xf6bb4b60
	DATA	·table+0x9c(SB)/4, $0xbebfbc70
	DATA	·table+0xa0(SB)/4, $0x289b7ec6
	DATA	·table+0xa4(SB)/4, $0xeaa127fa
	DATA	·table+0xa8(SB)/4, $0xd4ef3085
	DATA	·table+0xac(SB)/4, $0x04881d05
	DATA	·table+0xb0(SB)/4, $0xd9d4d039
	DATA	·table+0xb4(SB)/4, $0xe6db99e5
	DATA	·table+0xb8(SB)/4, $0x1fa27cf8
	DATA	·table+0xbc(SB)/4, $0xc4ac5665
	// Round 4
	DATA	·table+0xc0(SB)/4, $0xf4292244
	DATA	·table+0xc4(SB)/4, $0x432aff97
	DATA	·table+0xc8(SB)/4, $0xab9423a7
	DATA	·table+0xcc(SB)/4, $0xfc93a039
	DATA	·table+0xd0(SB)/4, $0x655b59c3
	DATA	·table+0xd4(SB)/4, $0x8f0ccc92
	DATA	·table+0xd8(SB)/4, $0xffeff47d
	DATA	·table+0xdc(SB)/4, $0x85845dd1
	DATA	·table+0xe0(SB)/4, $0x6fa87e4f
	DATA	·table+0xe4(SB)/4, $0xfe2ce6e0
	DATA	·table+0xe8(SB)/4, $0xa3014314
	DATA	·table+0xec(SB)/4, $0x4e0811a1
	DATA	·table+0xf0(SB)/4, $0xf7537e82
	DATA	·table+0xf4(SB)/4, $0xbd3af235
	DATA	·table+0xf8(SB)/4, $0x2ad7d2bb
	DATA	·table+0xfc(SB)/4, $0xeb86d391
	// Global definition
	GLOBL	·table(SB),8,$256
