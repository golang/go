// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// ARM version of md5block.go

#include "textflag.h"

// Register definitions
table = 0	// Pointer to MD5 constants table
data = 1	// Pointer to data to hash
a = 2		// MD5 accumulator
b = 3		// MD5 accumulator
c = 4		// MD5 accumulator
d = 5		// MD5 accumulator
c0 = 6		// MD5 constant
c1 = 7		// MD5 constant
c2 = 8		// MD5 constant
// r9, r10 are forbidden
// r11 is OK provided you check the assembler that no synthetic instructions use it
c3 = 11		// MD5 constant
t0 = 12		// temporary
t1 = 14		// temporary

// func block(dig *digest, p []byte)
// 0(FP) is *digest
// 4(FP) is p.array (struct Slice)
// 8(FP) is p.len
//12(FP) is p.cap
//
// Stack frame
p_end = -4	// -4(SP) pointer to the end of data
p_data = -8	// -8(SP) current data pointer
buf = -8-4*16	//-72(SP) 16 words temporary buffer
		// 3 words at 4..12(R13) for called routine parameters

TEXT	·block(SB), NOSPLIT, $84-16
	MOVW	p+4(FP), R(data)	// pointer to the data
	MOVW	p_len+8(FP), R(t0)	// number of bytes
	ADD	R(data), R(t0)
	MOVW	R(t0), p_end(SP)	// pointer to end of data

loop:
	MOVW	R(data), p_data(SP)	// Save R(data)
	AND.S	$3, R(data), R(t0)	// TST $3, R(data) not working see issue 5921
	BEQ	aligned			// aligned detected - skip copy

	// Copy the unaligned source data into the aligned temporary buffer
	// memove(to=4(R13), from=8(R13), n=12(R13)) - Corrupts all registers
	MOVW	$buf(SP), R(table)	// to
	MOVW	$64, R(c0)		// n
	MOVM.IB	[R(table),R(data),R(c0)], (R13)
	BL	runtime·memmove(SB)

	// Point to the local aligned copy of the data
	MOVW	$buf(SP), R(data)

aligned:
	// Point to the table of constants
	// A PC relative add would be cheaper than this
	MOVW	$·table(SB), R(table)

	// Load up initial MD5 accumulator
	MOVW	dig+0(FP), R(c0)
	MOVM.IA (R(c0)), [R(a),R(b),R(c),R(d)]

// a += (((c^d)&b)^d) + X[index] + const
// a = a<<shift | a>>(32-shift) + b
#define ROUND1(a, b, c, d, index, shift, const) \
	EOR	R(c), R(d), R(t0)		; \
	AND	R(b), R(t0)			; \
	EOR	R(d), R(t0)			; \
	MOVW	(index<<2)(R(data)), R(t1)	; \
	ADD	R(t1), R(t0)			; \
	ADD	R(const), R(t0)			; \
	ADD	R(t0), R(a)			; \
	ADD	R(a)@>(32-shift), R(b), R(a)	;

	MOVM.IA.W (R(table)), [R(c0),R(c1),R(c2),R(c3)]
	ROUND1(a, b, c, d,  0,	7, c0)
	ROUND1(d, a, b, c,  1, 12, c1)
	ROUND1(c, d, a, b,  2, 17, c2)
	ROUND1(b, c, d, a,  3, 22, c3)

	MOVM.IA.W (R(table)), [R(c0),R(c1),R(c2),R(c3)]
	ROUND1(a, b, c, d,  4,	7, c0)
	ROUND1(d, a, b, c,  5, 12, c1)
	ROUND1(c, d, a, b,  6, 17, c2)
	ROUND1(b, c, d, a,  7, 22, c3)

	MOVM.IA.W (R(table)), [R(c0),R(c1),R(c2),R(c3)]
	ROUND1(a, b, c, d,  8,	7, c0)
	ROUND1(d, a, b, c,  9, 12, c1)
	ROUND1(c, d, a, b, 10, 17, c2)
	ROUND1(b, c, d, a, 11, 22, c3)

	MOVM.IA.W (R(table)), [R(c0),R(c1),R(c2),R(c3)]
	ROUND1(a, b, c, d, 12,	7, c0)
	ROUND1(d, a, b, c, 13, 12, c1)
	ROUND1(c, d, a, b, 14, 17, c2)
	ROUND1(b, c, d, a, 15, 22, c3)

// a += (((b^c)&d)^c) + X[index] + const
// a = a<<shift | a>>(32-shift) + b
#define ROUND2(a, b, c, d, index, shift, const) \
	EOR	R(b), R(c), R(t0)		; \
	AND	R(d), R(t0)			; \
	EOR	R(c), R(t0)			; \
	MOVW	(index<<2)(R(data)), R(t1)	; \
	ADD	R(t1), R(t0)			; \
	ADD	R(const), R(t0)			; \
	ADD	R(t0), R(a)			; \
	ADD	R(a)@>(32-shift), R(b), R(a)	;

	MOVM.IA.W (R(table)), [R(c0),R(c1),R(c2),R(c3)]
	ROUND2(a, b, c, d,  1,	5, c0)
	ROUND2(d, a, b, c,  6,	9, c1)
	ROUND2(c, d, a, b, 11, 14, c2)
	ROUND2(b, c, d, a,  0, 20, c3)

	MOVM.IA.W (R(table)), [R(c0),R(c1),R(c2),R(c3)]
	ROUND2(a, b, c, d,  5,	5, c0)
	ROUND2(d, a, b, c, 10,	9, c1)
	ROUND2(c, d, a, b, 15, 14, c2)
	ROUND2(b, c, d, a,  4, 20, c3)

	MOVM.IA.W (R(table)), [R(c0),R(c1),R(c2),R(c3)]
	ROUND2(a, b, c, d,  9,	5, c0)
	ROUND2(d, a, b, c, 14,	9, c1)
	ROUND2(c, d, a, b,  3, 14, c2)
	ROUND2(b, c, d, a,  8, 20, c3)

	MOVM.IA.W (R(table)), [R(c0),R(c1),R(c2),R(c3)]
	ROUND2(a, b, c, d, 13,	5, c0)
	ROUND2(d, a, b, c,  2,	9, c1)
	ROUND2(c, d, a, b,  7, 14, c2)
	ROUND2(b, c, d, a, 12, 20, c3)

// a += (b^c^d) + X[index] + const
// a = a<<shift | a>>(32-shift) + b
#define ROUND3(a, b, c, d, index, shift, const) \
	EOR	R(b), R(c), R(t0)		; \
	EOR	R(d), R(t0)			; \
	MOVW	(index<<2)(R(data)), R(t1)	; \
	ADD	R(t1), R(t0)			; \
	ADD	R(const), R(t0)			; \
	ADD	R(t0), R(a)			; \
	ADD	R(a)@>(32-shift), R(b), R(a)	;

	MOVM.IA.W (R(table)), [R(c0),R(c1),R(c2),R(c3)]
	ROUND3(a, b, c, d,  5,	4, c0)
	ROUND3(d, a, b, c,  8, 11, c1)
	ROUND3(c, d, a, b, 11, 16, c2)
	ROUND3(b, c, d, a, 14, 23, c3)

	MOVM.IA.W (R(table)), [R(c0),R(c1),R(c2),R(c3)]
	ROUND3(a, b, c, d,  1,	4, c0)
	ROUND3(d, a, b, c,  4, 11, c1)
	ROUND3(c, d, a, b,  7, 16, c2)
	ROUND3(b, c, d, a, 10, 23, c3)

	MOVM.IA.W (R(table)), [R(c0),R(c1),R(c2),R(c3)]
	ROUND3(a, b, c, d, 13,	4, c0)
	ROUND3(d, a, b, c,  0, 11, c1)
	ROUND3(c, d, a, b,  3, 16, c2)
	ROUND3(b, c, d, a,  6, 23, c3)

	MOVM.IA.W (R(table)), [R(c0),R(c1),R(c2),R(c3)]
	ROUND3(a, b, c, d,  9,	4, c0)
	ROUND3(d, a, b, c, 12, 11, c1)
	ROUND3(c, d, a, b, 15, 16, c2)
	ROUND3(b, c, d, a,  2, 23, c3)

// a += (c^(b|^d)) + X[index] + const
// a = a<<shift | a>>(32-shift) + b
#define ROUND4(a, b, c, d, index, shift, const) \
	MVN	R(d), R(t0)			; \
	ORR	R(b), R(t0)			; \
	EOR	R(c), R(t0)			; \
	MOVW	(index<<2)(R(data)), R(t1)	; \
	ADD	R(t1), R(t0)			; \
	ADD	R(const), R(t0)			; \
	ADD	R(t0), R(a)			; \
	ADD	R(a)@>(32-shift), R(b), R(a)	;

	MOVM.IA.W (R(table)), [R(c0),R(c1),R(c2),R(c3)]
	ROUND4(a, b, c, d,  0,	6, c0)
	ROUND4(d, a, b, c,  7, 10, c1)
	ROUND4(c, d, a, b, 14, 15, c2)
	ROUND4(b, c, d, a,  5, 21, c3)

	MOVM.IA.W (R(table)), [R(c0),R(c1),R(c2),R(c3)]
	ROUND4(a, b, c, d, 12,	6, c0)
	ROUND4(d, a, b, c,  3, 10, c1)
	ROUND4(c, d, a, b, 10, 15, c2)
	ROUND4(b, c, d, a,  1, 21, c3)

	MOVM.IA.W (R(table)), [R(c0),R(c1),R(c2),R(c3)]
	ROUND4(a, b, c, d,  8,	6, c0)
	ROUND4(d, a, b, c, 15, 10, c1)
	ROUND4(c, d, a, b,  6, 15, c2)
	ROUND4(b, c, d, a, 13, 21, c3)

	MOVM.IA.W (R(table)), [R(c0),R(c1),R(c2),R(c3)]
	ROUND4(a, b, c, d,  4,	6, c0)
	ROUND4(d, a, b, c, 11, 10, c1)
	ROUND4(c, d, a, b,  2, 15, c2)
	ROUND4(b, c, d, a,  9, 21, c3)

	MOVW	dig+0(FP), R(t0)
	MOVM.IA (R(t0)), [R(c0),R(c1),R(c2),R(c3)]

	ADD	R(c0), R(a)
	ADD	R(c1), R(b)
	ADD	R(c2), R(c)
	ADD	R(c3), R(d)

	MOVM.IA [R(a),R(b),R(c),R(d)], (R(t0))

	MOVW	p_data(SP), R(data)
	MOVW	p_end(SP), R(t0)
	ADD	$64, R(data)
	CMP	R(t0), R(data)
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
