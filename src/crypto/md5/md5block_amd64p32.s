// Original source:
//	http://www.zorinaq.com/papers/md5-amd64.html
//	http://www.zorinaq.com/papers/md5-amd64.tar.bz2
//
// Translated from Perl generating GNU assembly into
// #defines generating 6a assembly by the Go Authors.
//
// Restrictions to make code safe for Native Client:
// replace BP with R11, reloaded before use at return.
// replace R15 with R11.

#include "textflag.h"

// MD5 optimized for AMD64.
//
// Author: Marc Bevand <bevand_m (at) epita.fr>
// Licence: I hereby disclaim the copyright on this code and place it
// in the public domain.

TEXT	·block(SB),NOSPLIT,$0-32
	MOVL	dig+0(FP),	R11
	MOVL	p+4(FP),	SI
	MOVL	p_len+8(FP), DX
	SHRQ	$6,		DX
	SHLQ	$6,		DX

	LEAQ	(SI)(DX*1),	DI
	MOVL	(0*4)(R11),	AX
	MOVL	(1*4)(R11),	BX
	MOVL	(2*4)(R11),	CX
	MOVL	(3*4)(R11),	DX

	CMPQ	SI,		DI
	JEQ	end

loop:
	MOVL	AX,		R12
	MOVL	BX,		R13
	MOVL	CX,		R14
	MOVL	DX,		R11

	MOVL	(0*4)(SI),	R8
	MOVL	DX,		R9

#define ROUND1(a, b, c, d, index, const, shift) \
	XORL	c, R9; \
	LEAL	const(a)(R8*1), a; \
	ANDL	b, R9; \
	XORL d, R9; \
	MOVL (index*4)(SI), R8; \
	ADDL R9, a; \
	ROLL $shift, a; \
	MOVL c, R9; \
	ADDL b, a

	ROUND1(AX,BX,CX,DX, 1,0xd76aa478, 7);
	ROUND1(DX,AX,BX,CX, 2,0xe8c7b756,12);
	ROUND1(CX,DX,AX,BX, 3,0x242070db,17);
	ROUND1(BX,CX,DX,AX, 4,0xc1bdceee,22);
	ROUND1(AX,BX,CX,DX, 5,0xf57c0faf, 7);
	ROUND1(DX,AX,BX,CX, 6,0x4787c62a,12);
	ROUND1(CX,DX,AX,BX, 7,0xa8304613,17);
	ROUND1(BX,CX,DX,AX, 8,0xfd469501,22);
	ROUND1(AX,BX,CX,DX, 9,0x698098d8, 7);
	ROUND1(DX,AX,BX,CX,10,0x8b44f7af,12);
	ROUND1(CX,DX,AX,BX,11,0xffff5bb1,17);
	ROUND1(BX,CX,DX,AX,12,0x895cd7be,22);
	ROUND1(AX,BX,CX,DX,13,0x6b901122, 7);
	ROUND1(DX,AX,BX,CX,14,0xfd987193,12);
	ROUND1(CX,DX,AX,BX,15,0xa679438e,17);
	ROUND1(BX,CX,DX,AX, 0,0x49b40821,22);

	MOVL	(1*4)(SI),	R8
	MOVL	DX,		R9
	MOVL	DX,		R10

#define ROUND2(a, b, c, d, index, const, shift) \
	NOTL	R9; \
	LEAL	const(a)(R8*1),a; \
	ANDL	b,		R10; \
	ANDL	c,		R9; \
	MOVL	(index*4)(SI),R8; \
	ORL	R9,		R10; \
	MOVL	c,		R9; \
	ADDL	R10,		a; \
	MOVL	c,		R10; \
	ROLL	$shift,	a; \
	ADDL	b,		a

	ROUND2(AX,BX,CX,DX, 6,0xf61e2562, 5);
	ROUND2(DX,AX,BX,CX,11,0xc040b340, 9);
	ROUND2(CX,DX,AX,BX, 0,0x265e5a51,14);
	ROUND2(BX,CX,DX,AX, 5,0xe9b6c7aa,20);
	ROUND2(AX,BX,CX,DX,10,0xd62f105d, 5);
	ROUND2(DX,AX,BX,CX,15, 0x2441453, 9);
	ROUND2(CX,DX,AX,BX, 4,0xd8a1e681,14);
	ROUND2(BX,CX,DX,AX, 9,0xe7d3fbc8,20);
	ROUND2(AX,BX,CX,DX,14,0x21e1cde6, 5);
	ROUND2(DX,AX,BX,CX, 3,0xc33707d6, 9);
	ROUND2(CX,DX,AX,BX, 8,0xf4d50d87,14);
	ROUND2(BX,CX,DX,AX,13,0x455a14ed,20);
	ROUND2(AX,BX,CX,DX, 2,0xa9e3e905, 5);
	ROUND2(DX,AX,BX,CX, 7,0xfcefa3f8, 9);
	ROUND2(CX,DX,AX,BX,12,0x676f02d9,14);
	ROUND2(BX,CX,DX,AX, 0,0x8d2a4c8a,20);
 
	MOVL	(5*4)(SI),	R8
	MOVL	CX,		R9

#define ROUND3(a, b, c, d, index, const, shift) \
	LEAL	const(a)(R8*1),a; \
	MOVL	(index*4)(SI),R8; \
	XORL	d,		R9; \
	XORL	b,		R9; \
	ADDL	R9,		a; \
	ROLL	$shift,		a; \
	MOVL	b,		R9; \
	ADDL	b,		a

	ROUND3(AX,BX,CX,DX, 8,0xfffa3942, 4);
	ROUND3(DX,AX,BX,CX,11,0x8771f681,11);
	ROUND3(CX,DX,AX,BX,14,0x6d9d6122,16);
	ROUND3(BX,CX,DX,AX, 1,0xfde5380c,23);
	ROUND3(AX,BX,CX,DX, 4,0xa4beea44, 4);
	ROUND3(DX,AX,BX,CX, 7,0x4bdecfa9,11);
	ROUND3(CX,DX,AX,BX,10,0xf6bb4b60,16);
	ROUND3(BX,CX,DX,AX,13,0xbebfbc70,23);
	ROUND3(AX,BX,CX,DX, 0,0x289b7ec6, 4);
	ROUND3(DX,AX,BX,CX, 3,0xeaa127fa,11);
	ROUND3(CX,DX,AX,BX, 6,0xd4ef3085,16);
	ROUND3(BX,CX,DX,AX, 9, 0x4881d05,23);
	ROUND3(AX,BX,CX,DX,12,0xd9d4d039, 4);
	ROUND3(DX,AX,BX,CX,15,0xe6db99e5,11);
	ROUND3(CX,DX,AX,BX, 2,0x1fa27cf8,16);
	ROUND3(BX,CX,DX,AX, 0,0xc4ac5665,23);

	MOVL	(0*4)(SI),	R8
	MOVL	$0xffffffff,	R9
	XORL	DX,		R9

#define ROUND4(a, b, c, d, index, const, shift) \
	LEAL	const(a)(R8*1),a; \
	ORL	b,		R9; \
	XORL	c,		R9; \
	ADDL	R9,		a; \
	MOVL	(index*4)(SI),R8; \
	MOVL	$0xffffffff,	R9; \
	ROLL	$shift,		a; \
	XORL	c,		R9; \
	ADDL	b,		a
	
	ROUND4(AX,BX,CX,DX, 7,0xf4292244, 6);
	ROUND4(DX,AX,BX,CX,14,0x432aff97,10);
	ROUND4(CX,DX,AX,BX, 5,0xab9423a7,15);
	ROUND4(BX,CX,DX,AX,12,0xfc93a039,21);
	ROUND4(AX,BX,CX,DX, 3,0x655b59c3, 6);
	ROUND4(DX,AX,BX,CX,10,0x8f0ccc92,10);
	ROUND4(CX,DX,AX,BX, 1,0xffeff47d,15);
	ROUND4(BX,CX,DX,AX, 8,0x85845dd1,21);
	ROUND4(AX,BX,CX,DX,15,0x6fa87e4f, 6);
	ROUND4(DX,AX,BX,CX, 6,0xfe2ce6e0,10);
	ROUND4(CX,DX,AX,BX,13,0xa3014314,15);
	ROUND4(BX,CX,DX,AX, 4,0x4e0811a1,21);
	ROUND4(AX,BX,CX,DX,11,0xf7537e82, 6);
	ROUND4(DX,AX,BX,CX, 2,0xbd3af235,10);
	ROUND4(CX,DX,AX,BX, 9,0x2ad7d2bb,15);
	ROUND4(BX,CX,DX,AX, 0,0xeb86d391,21);

	ADDL	R12,	AX
	ADDL	R13,	BX
	ADDL	R14,	CX
	ADDL	R11,	DX

	ADDQ	$64,		SI
	CMPQ	SI,		DI
	JB	loop

end:
	MOVL	dig+0(FP),	R11
	MOVL	AX,		(0*4)(R11)
	MOVL	BX,		(1*4)(R11)
	MOVL	CX,		(2*4)(R11)
	MOVL	DX,		(3*4)(R11)
	RET
