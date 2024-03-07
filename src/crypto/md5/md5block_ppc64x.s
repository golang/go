// Original source:
//	http://www.zorinaq.com/papers/md5-amd64.html
//	http://www.zorinaq.com/papers/md5-amd64.tar.bz2
//
// MD5 optimized for ppc64le using Go's assembler for
// ppc64le, based on md5block_amd64.s implementation by
// the Go authors.
//
// Author: Marc Bevand <bevand_m (at) epita.fr>
// Licence: I hereby disclaim the copyright on this code and place it
// in the public domain.

//go:build (ppc64 || ppc64le) && !purego

#include "textflag.h"

// ENDIAN_MOVE generates the appropriate
// 4 byte load for big or little endian.
// The 4 bytes at ptr+off is loaded into dst.
// The idx reg is only needed for big endian
// and is clobbered when used.
#ifdef GOARCH_ppc64le
#define ENDIAN_MOVE(off, ptr, dst, idx) \
	MOVWZ	off(ptr),dst
#else
#define ENDIAN_MOVE(off, ptr, dst, idx) \
	MOVD	$off,idx; \
	MOVWBR	(idx)(ptr), dst
#endif

#define M00 R18
#define M01 R19
#define M02 R20
#define M03 R24
#define M04 R25
#define M05 R26
#define M06 R27
#define M07 R28
#define M08 R29
#define M09 R21
#define M10 R11
#define M11 R8
#define M12 R7
#define M13 R12
#define M14 R23
#define M15 R10

#define ROUND1(a, b, c, d, index, const, shift) \
	ADD	$const, index, R9; \
	ADD	R9, a; \
	AND     b, c, R9; \
	ANDN    b, d, R31; \
	OR	R9, R31, R9; \
	ADD	R9, a; \
	ROTLW	$shift, a; \
	ADD	b, a;

#define ROUND2(a, b, c, d, index, const, shift) \
	ADD	$const, index, R9; \
	ADD	R9, a; \
	AND	b, d, R31; \
	ANDN	d, c, R9; \
	OR	R9, R31; \
	ADD	R31, a; \
	ROTLW	$shift, a; \
	ADD	b, a;

#define ROUND3(a, b, c, d, index, const, shift) \
	ADD	$const, index, R9; \
	ADD	R9, a; \
	XOR	d, c, R31; \
	XOR	b, R31; \
	ADD	R31, a; \
	ROTLW	$shift, a; \
	ADD	b, a;

#define ROUND4(a, b, c, d, index, const, shift) \
	ADD	$const, index, R9; \
	ADD	R9, a; \
	ORN     d, b, R31; \
	XOR	c, R31; \
	ADD	R31, a; \
	ROTLW	$shift, a; \
	ADD	b, a;


TEXT ·block(SB),NOSPLIT,$0-32
	MOVD	dig+0(FP), R10
	MOVD	p+8(FP), R6
	MOVD	p_len+16(FP), R5

	// We assume p_len >= 64
	SRD 	$6, R5
	MOVD	R5, CTR

	MOVWZ	0(R10), R22
	MOVWZ	4(R10), R3
	MOVWZ	8(R10), R4
	MOVWZ	12(R10), R5

loop:
	MOVD	R22, R14
	MOVD	R3, R15
	MOVD	R4, R16
	MOVD	R5, R17

	ENDIAN_MOVE( 0,R6,M00,M15)
	ENDIAN_MOVE( 4,R6,M01,M15)
	ENDIAN_MOVE( 8,R6,M02,M15)
	ENDIAN_MOVE(12,R6,M03,M15)

	ROUND1(R22,R3,R4,R5,M00,0xd76aa478, 7);
	ROUND1(R5,R22,R3,R4,M01,0xe8c7b756,12);
	ROUND1(R4,R5,R22,R3,M02,0x242070db,17);
	ROUND1(R3,R4,R5,R22,M03,0xc1bdceee,22);

	ENDIAN_MOVE(16,R6,M04,M15)
	ENDIAN_MOVE(20,R6,M05,M15)
	ENDIAN_MOVE(24,R6,M06,M15)
	ENDIAN_MOVE(28,R6,M07,M15)

	ROUND1(R22,R3,R4,R5,M04,0xf57c0faf, 7);
	ROUND1(R5,R22,R3,R4,M05,0x4787c62a,12);
	ROUND1(R4,R5,R22,R3,M06,0xa8304613,17);
	ROUND1(R3,R4,R5,R22,M07,0xfd469501,22);

	ENDIAN_MOVE(32,R6,M08,M15)
	ENDIAN_MOVE(36,R6,M09,M15)
	ENDIAN_MOVE(40,R6,M10,M15)
	ENDIAN_MOVE(44,R6,M11,M15)

	ROUND1(R22,R3,R4,R5,M08,0x698098d8, 7);
	ROUND1(R5,R22,R3,R4,M09,0x8b44f7af,12);
	ROUND1(R4,R5,R22,R3,M10,0xffff5bb1,17);
	ROUND1(R3,R4,R5,R22,M11,0x895cd7be,22);

	ENDIAN_MOVE(48,R6,M12,M15)
	ENDIAN_MOVE(52,R6,M13,M15)
	ENDIAN_MOVE(56,R6,M14,M15)
	ENDIAN_MOVE(60,R6,M15,M15)

	ROUND1(R22,R3,R4,R5,M12,0x6b901122, 7);
	ROUND1(R5,R22,R3,R4,M13,0xfd987193,12);
	ROUND1(R4,R5,R22,R3,M14,0xa679438e,17);
	ROUND1(R3,R4,R5,R22,M15,0x49b40821,22);

	ROUND2(R22,R3,R4,R5,M01,0xf61e2562, 5);
	ROUND2(R5,R22,R3,R4,M06,0xc040b340, 9);
	ROUND2(R4,R5,R22,R3,M11,0x265e5a51,14);
	ROUND2(R3,R4,R5,R22,M00,0xe9b6c7aa,20);
	ROUND2(R22,R3,R4,R5,M05,0xd62f105d, 5);
	ROUND2(R5,R22,R3,R4,M10, 0x2441453, 9);
	ROUND2(R4,R5,R22,R3,M15,0xd8a1e681,14);
	ROUND2(R3,R4,R5,R22,M04,0xe7d3fbc8,20);
	ROUND2(R22,R3,R4,R5,M09,0x21e1cde6, 5);
	ROUND2(R5,R22,R3,R4,M14,0xc33707d6, 9);
	ROUND2(R4,R5,R22,R3,M03,0xf4d50d87,14);
	ROUND2(R3,R4,R5,R22,M08,0x455a14ed,20);
	ROUND2(R22,R3,R4,R5,M13,0xa9e3e905, 5);
	ROUND2(R5,R22,R3,R4,M02,0xfcefa3f8, 9);
	ROUND2(R4,R5,R22,R3,M07,0x676f02d9,14);
	ROUND2(R3,R4,R5,R22,M12,0x8d2a4c8a,20);

	ROUND3(R22,R3,R4,R5,M05,0xfffa3942, 4);
	ROUND3(R5,R22,R3,R4,M08,0x8771f681,11);
	ROUND3(R4,R5,R22,R3,M11,0x6d9d6122,16);
	ROUND3(R3,R4,R5,R22,M14,0xfde5380c,23);
	ROUND3(R22,R3,R4,R5,M01,0xa4beea44, 4);
	ROUND3(R5,R22,R3,R4,M04,0x4bdecfa9,11);
	ROUND3(R4,R5,R22,R3,M07,0xf6bb4b60,16);
	ROUND3(R3,R4,R5,R22,M10,0xbebfbc70,23);
	ROUND3(R22,R3,R4,R5,M13,0x289b7ec6, 4);
	ROUND3(R5,R22,R3,R4,M00,0xeaa127fa,11);
	ROUND3(R4,R5,R22,R3,M03,0xd4ef3085,16);
	ROUND3(R3,R4,R5,R22,M06, 0x4881d05,23);
	ROUND3(R22,R3,R4,R5,M09,0xd9d4d039, 4);
	ROUND3(R5,R22,R3,R4,M12,0xe6db99e5,11);
	ROUND3(R4,R5,R22,R3,M15,0x1fa27cf8,16);
	ROUND3(R3,R4,R5,R22,M02,0xc4ac5665,23);

	ROUND4(R22,R3,R4,R5,M00,0xf4292244, 6);
	ROUND4(R5,R22,R3,R4,M07,0x432aff97,10);
	ROUND4(R4,R5,R22,R3,M14,0xab9423a7,15);
	ROUND4(R3,R4,R5,R22,M05,0xfc93a039,21);
	ROUND4(R22,R3,R4,R5,M12,0x655b59c3, 6);
	ROUND4(R5,R22,R3,R4,M03,0x8f0ccc92,10);
	ROUND4(R4,R5,R22,R3,M10,0xffeff47d,15);
	ROUND4(R3,R4,R5,R22,M01,0x85845dd1,21);
	ROUND4(R22,R3,R4,R5,M08,0x6fa87e4f, 6);
	ROUND4(R5,R22,R3,R4,M15,0xfe2ce6e0,10);
	ROUND4(R4,R5,R22,R3,M06,0xa3014314,15);
	ROUND4(R3,R4,R5,R22,M13,0x4e0811a1,21);
	ROUND4(R22,R3,R4,R5,M04,0xf7537e82, 6);
	ROUND4(R5,R22,R3,R4,M11,0xbd3af235,10);
	ROUND4(R4,R5,R22,R3,M02,0x2ad7d2bb,15);
	ROUND4(R3,R4,R5,R22,M09,0xeb86d391,21);

	ADD	R14, R22
	ADD	R15, R3
	ADD	R16, R4
	ADD	R17, R5
	ADD	$64, R6
	BC	16, 0, loop // bdnz

end:
	MOVD	dig+0(FP), R10
	MOVWZ	R22, 0(R10)
	MOVWZ	R3, 4(R10)
	MOVWZ	R4, 8(R10)
	MOVWZ	R5, 12(R10)

	RET
