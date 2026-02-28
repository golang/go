// Original source:
//	http://www.zorinaq.com/papers/md5-amd64.html
//	http://www.zorinaq.com/papers/md5-amd64.tar.bz2
//
// MD5 adapted for s390x using Go's assembler for
// s390x, based on md5block_amd64.s implementation by
// the Go authors.
//
// Author: Marc Bevand <bevand_m (at) epita.fr>
// Licence: I hereby disclaim the copyright on this code and place it
// in the public domain.

//go:build !purego

#include "textflag.h"

// func block(dig *digest, p []byte)
TEXT ·block(SB),NOSPLIT,$16-32
	MOVD	dig+0(FP), R1
	MOVD	p+8(FP), R6
	MOVD	p_len+16(FP), R5
	AND	$-64, R5
	LAY	(R6)(R5*1), R7

	LMY	0(R1), R2, R5
	CMPBEQ	R6, R7, end

loop:
	STMY	R2, R5, tmp-16(SP)

	MOVWBR	0(R6), R8
	MOVWZ	R5, R9

#define ROUND1(a, b, c, d, index, const, shift) \
	XOR	c, R9; \
	ADD	$const, a; \
	ADD	R8, a; \
	MOVWBR	(index*4)(R6), R8; \
	AND	b, R9; \
	XOR	d, R9; \
	ADD	R9, a; \
	RLL	$shift, a; \
	MOVWZ	c, R9; \
	ADD	b, a

	ROUND1(R2,R3,R4,R5, 1,0xd76aa478, 7);
	ROUND1(R5,R2,R3,R4, 2,0xe8c7b756,12);
	ROUND1(R4,R5,R2,R3, 3,0x242070db,17);
	ROUND1(R3,R4,R5,R2, 4,0xc1bdceee,22);
	ROUND1(R2,R3,R4,R5, 5,0xf57c0faf, 7);
	ROUND1(R5,R2,R3,R4, 6,0x4787c62a,12);
	ROUND1(R4,R5,R2,R3, 7,0xa8304613,17);
	ROUND1(R3,R4,R5,R2, 8,0xfd469501,22);
	ROUND1(R2,R3,R4,R5, 9,0x698098d8, 7);
	ROUND1(R5,R2,R3,R4,10,0x8b44f7af,12);
	ROUND1(R4,R5,R2,R3,11,0xffff5bb1,17);
	ROUND1(R3,R4,R5,R2,12,0x895cd7be,22);
	ROUND1(R2,R3,R4,R5,13,0x6b901122, 7);
	ROUND1(R5,R2,R3,R4,14,0xfd987193,12);
	ROUND1(R4,R5,R2,R3,15,0xa679438e,17);
	ROUND1(R3,R4,R5,R2, 0,0x49b40821,22);

	MOVWBR	(1*4)(R6), R8
	MOVWZ	R5, R9
	MOVWZ	R5, R1

#define ROUND2(a, b, c, d, index, const, shift) \
	XOR	$0xffffffff, R9; \ // NOTW R9
	ADD	$const, a; \
	ADD	R8, a; \
	MOVWBR	(index*4)(R6), R8; \
	AND	b, R1; \
	AND	c, R9; \
	OR	R9, R1; \
	MOVWZ	c, R9; \
	ADD	R1, a; \
	MOVWZ	c, R1; \
	RLL	$shift,	a; \
	ADD	b, a

	ROUND2(R2,R3,R4,R5, 6,0xf61e2562, 5);
	ROUND2(R5,R2,R3,R4,11,0xc040b340, 9);
	ROUND2(R4,R5,R2,R3, 0,0x265e5a51,14);
	ROUND2(R3,R4,R5,R2, 5,0xe9b6c7aa,20);
	ROUND2(R2,R3,R4,R5,10,0xd62f105d, 5);
	ROUND2(R5,R2,R3,R4,15, 0x2441453, 9);
	ROUND2(R4,R5,R2,R3, 4,0xd8a1e681,14);
	ROUND2(R3,R4,R5,R2, 9,0xe7d3fbc8,20);
	ROUND2(R2,R3,R4,R5,14,0x21e1cde6, 5);
	ROUND2(R5,R2,R3,R4, 3,0xc33707d6, 9);
	ROUND2(R4,R5,R2,R3, 8,0xf4d50d87,14);
	ROUND2(R3,R4,R5,R2,13,0x455a14ed,20);
	ROUND2(R2,R3,R4,R5, 2,0xa9e3e905, 5);
	ROUND2(R5,R2,R3,R4, 7,0xfcefa3f8, 9);
	ROUND2(R4,R5,R2,R3,12,0x676f02d9,14);
	ROUND2(R3,R4,R5,R2, 0,0x8d2a4c8a,20);

	MOVWBR	(5*4)(R6), R8
	MOVWZ	R4, R9

#define ROUND3(a, b, c, d, index, const, shift) \
	ADD	$const, a; \
	ADD	R8, a; \
	MOVWBR	(index*4)(R6), R8; \
	XOR	d, R9; \
	XOR	b, R9; \
	ADD	R9, a; \
	RLL	$shift, a; \
	MOVWZ	b, R9; \
	ADD	b, a

	ROUND3(R2,R3,R4,R5, 8,0xfffa3942, 4);
	ROUND3(R5,R2,R3,R4,11,0x8771f681,11);
	ROUND3(R4,R5,R2,R3,14,0x6d9d6122,16);
	ROUND3(R3,R4,R5,R2, 1,0xfde5380c,23);
	ROUND3(R2,R3,R4,R5, 4,0xa4beea44, 4);
	ROUND3(R5,R2,R3,R4, 7,0x4bdecfa9,11);
	ROUND3(R4,R5,R2,R3,10,0xf6bb4b60,16);
	ROUND3(R3,R4,R5,R2,13,0xbebfbc70,23);
	ROUND3(R2,R3,R4,R5, 0,0x289b7ec6, 4);
	ROUND3(R5,R2,R3,R4, 3,0xeaa127fa,11);
	ROUND3(R4,R5,R2,R3, 6,0xd4ef3085,16);
	ROUND3(R3,R4,R5,R2, 9, 0x4881d05,23);
	ROUND3(R2,R3,R4,R5,12,0xd9d4d039, 4);
	ROUND3(R5,R2,R3,R4,15,0xe6db99e5,11);
	ROUND3(R4,R5,R2,R3, 2,0x1fa27cf8,16);
	ROUND3(R3,R4,R5,R2, 0,0xc4ac5665,23);

	MOVWBR	(0*4)(R6), R8
	MOVWZ	$0xffffffff, R9
	XOR	R5, R9

#define ROUND4(a, b, c, d, index, const, shift) \
	ADD	$const, a; \
	ADD	R8, a; \
	MOVWBR	(index*4)(R6), R8; \
	OR	b, R9; \
	XOR	c, R9; \
	ADD	R9, a; \
	MOVWZ	$0xffffffff, R9; \
	RLL	$shift,	a; \
	XOR	c, R9; \
	ADD	b, a

	ROUND4(R2,R3,R4,R5, 7,0xf4292244, 6);
	ROUND4(R5,R2,R3,R4,14,0x432aff97,10);
	ROUND4(R4,R5,R2,R3, 5,0xab9423a7,15);
	ROUND4(R3,R4,R5,R2,12,0xfc93a039,21);
	ROUND4(R2,R3,R4,R5, 3,0x655b59c3, 6);
	ROUND4(R5,R2,R3,R4,10,0x8f0ccc92,10);
	ROUND4(R4,R5,R2,R3, 1,0xffeff47d,15);
	ROUND4(R3,R4,R5,R2, 8,0x85845dd1,21);
	ROUND4(R2,R3,R4,R5,15,0x6fa87e4f, 6);
	ROUND4(R5,R2,R3,R4, 6,0xfe2ce6e0,10);
	ROUND4(R4,R5,R2,R3,13,0xa3014314,15);
	ROUND4(R3,R4,R5,R2, 4,0x4e0811a1,21);
	ROUND4(R2,R3,R4,R5,11,0xf7537e82, 6);
	ROUND4(R5,R2,R3,R4, 2,0xbd3af235,10);
	ROUND4(R4,R5,R2,R3, 9,0x2ad7d2bb,15);
	ROUND4(R3,R4,R5,R2, 0,0xeb86d391,21);

	MOVWZ	tmp-16(SP), R1
	ADD	R1, R2
	MOVWZ	tmp-12(SP), R1
	ADD	R1, R3
	MOVWZ	tmp-8(SP), R1
	ADD	R1, R4
	MOVWZ	tmp-4(SP), R1
	ADD	R1, R5

	LA	64(R6), R6
	CMPBLT	R6, R7, loop

end:
	MOVD	dig+0(FP), R1
	STMY	R2, R5, 0(R1)
	RET
