// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// SHA256 block routine. See sha256block.go for Go equivalent.
//
// The algorithm is detailed in FIPS 180-4:
//
//  http://csrc.nist.gov/publications/fips/fips180-4/fips-180-4.pdf

// The avx2-version is described in an Intel White-Paper:
// "Fast SHA-256 Implementations on Intel Architecture Processors"
// To find it, surf to http://www.intel.com/p/en_US/embedded
// and search for that title.
// AVX2 version by Intel, same algorithm as code in Linux kernel:
// https://github.com/torvalds/linux/blob/master/arch/x86/crypto/sha256-avx2-asm.S
// by
//     James Guilford <james.guilford@intel.com>
//     Kirk Yap <kirk.s.yap@intel.com>
//     Tim Chen <tim.c.chen@linux.intel.com>

// Wt = Mt; for 0 <= t <= 15
// Wt = SIGMA1(Wt-2) + SIGMA0(Wt-15) + Wt-16; for 16 <= t <= 63
//
// a = H0
// b = H1
// c = H2
// d = H3
// e = H4
// f = H5
// g = H6
// h = H7
//
// for t = 0 to 63 {
//    T1 = h + BIGSIGMA1(e) + Ch(e,f,g) + Kt + Wt
//    T2 = BIGSIGMA0(a) + Maj(a,b,c)
//    h = g
//    g = f
//    f = e
//    e = d + T1
//    d = c
//    c = b
//    b = a
//    a = T1 + T2
// }
//
// H0 = a + H0
// H1 = b + H1
// H2 = c + H2
// H3 = d + H3
// H4 = e + H4
// H5 = f + H5
// H6 = g + H6
// H7 = h + H7

// Wt = Mt; for 0 <= t <= 15
#define MSGSCHEDULE0(index) \
	MOVL	(index*4)(SI), AX; \
	BSWAPL	AX; \
	MOVL	AX, (index*4)(BP)

// Wt = SIGMA1(Wt-2) + Wt-7 + SIGMA0(Wt-15) + Wt-16; for 16 <= t <= 63
//   SIGMA0(x) = ROTR(7,x) XOR ROTR(18,x) XOR SHR(3,x)
//   SIGMA1(x) = ROTR(17,x) XOR ROTR(19,x) XOR SHR(10,x)
#define MSGSCHEDULE1(index) \
	MOVL	((index-2)*4)(BP), AX; \
	MOVL	AX, CX; \
	RORL	$17, AX; \
	MOVL	CX, DX; \
	RORL	$19, CX; \
	SHRL	$10, DX; \
	MOVL	((index-15)*4)(BP), BX; \
	XORL	CX, AX; \
	MOVL	BX, CX; \
	XORL	DX, AX; \
	RORL	$7, BX; \
	MOVL	CX, DX; \
	SHRL	$3, DX; \
	RORL	$18, CX; \
	ADDL	((index-7)*4)(BP), AX; \
	XORL	CX, BX; \
	XORL	DX, BX; \
	ADDL	((index-16)*4)(BP), BX; \
	ADDL	BX, AX; \
	MOVL	AX, ((index)*4)(BP)

// Calculate T1 in AX - uses AX, CX and DX registers.
// h is also used as an accumulator. Wt is passed in AX.
//   T1 = h + BIGSIGMA1(e) + Ch(e, f, g) + Kt + Wt
//     BIGSIGMA1(x) = ROTR(6,x) XOR ROTR(11,x) XOR ROTR(25,x)
//     Ch(x, y, z) = (x AND y) XOR (NOT x AND z)
#define SHA256T1(const, e, f, g, h) \
	ADDL	AX, h; \
	MOVL	e, AX; \
	ADDL	$const, h; \
	MOVL	e, CX; \
	RORL	$6, AX; \
	MOVL	e, DX; \
	RORL	$11, CX; \
	XORL	CX, AX; \
	MOVL	e, CX; \
	RORL	$25, DX; \
	ANDL	f, CX; \
	XORL	AX, DX; \
	MOVL	e, AX; \
	NOTL	AX; \
	ADDL	DX, h; \
	ANDL	g, AX; \
	XORL	CX, AX; \
	ADDL	h, AX

// Calculate T2 in BX - uses BX, CX, DX and DI registers.
//   T2 = BIGSIGMA0(a) + Maj(a, b, c)
//     BIGSIGMA0(x) = ROTR(2,x) XOR ROTR(13,x) XOR ROTR(22,x)
//     Maj(x, y, z) = (x AND y) XOR (x AND z) XOR (y AND z)
#define SHA256T2(a, b, c) \
	MOVL	a, DI; \
	MOVL	c, BX; \
	RORL	$2, DI; \
	MOVL	a, DX; \
	ANDL	b, BX; \
	RORL	$13, DX; \
	MOVL	a, CX; \
	ANDL	c, CX; \
	XORL	DX, DI; \
	XORL	CX, BX; \
	MOVL	a, DX; \
	MOVL	b, CX; \
	RORL	$22, DX; \
	ANDL	a, CX; \
	XORL	CX, BX; \
	XORL	DX, DI; \
	ADDL	DI, BX

// Calculate T1 and T2, then e = d + T1 and a = T1 + T2.
// The values for e and a are stored in d and h, ready for rotation.
#define SHA256ROUND(index, const, a, b, c, d, e, f, g, h) \
	SHA256T1(const, e, f, g, h); \
	SHA256T2(a, b, c); \
	MOVL	BX, h; \
	ADDL	AX, d; \
	ADDL	AX, h

#define SHA256ROUND0(index, const, a, b, c, d, e, f, g, h) \
	MSGSCHEDULE0(index); \
	SHA256ROUND(index, const, a, b, c, d, e, f, g, h)

#define SHA256ROUND1(index, const, a, b, c, d, e, f, g, h) \
	MSGSCHEDULE1(index); \
	SHA256ROUND(index, const, a, b, c, d, e, f, g, h)


// Definitions for AVX2 version

// addm (mem), reg
// Add reg to mem using reg-mem add and store
#define addm(P1, P2) \
	ADDL P2, P1; \
	MOVL P1, P2

#define XDWORD0 Y4
#define XDWORD1 Y5
#define XDWORD2 Y6
#define XDWORD3 Y7

#define XWORD0 X4
#define XWORD1 X5
#define XWORD2 X6
#define XWORD3 X7

#define XTMP0 Y0
#define XTMP1 Y1
#define XTMP2 Y2
#define XTMP3 Y3
#define XTMP4 Y8
#define XTMP5 Y11

#define XFER  Y9

#define BYTE_FLIP_MASK 	Y13 // mask to convert LE -> BE
#define X_BYTE_FLIP_MASK X13

#define NUM_BYTES DX
#define INP	DI

#define CTX SI // Beginning of digest in memory (a, b, c, ... , h)

#define a AX
#define b BX
#define c CX
#define d R8
#define e DX
#define f R9
#define g R10
#define h R11

#define old_h R11

#define TBL BP

#define SRND SI // SRND is same register as CTX

#define T1 R12

#define y0 R13
#define y1 R14
#define y2 R15
#define y3 DI

// Offsets
#define XFER_SIZE 2*64*4
#define INP_END_SIZE 8
#define INP_SIZE 8
#define TMP_SIZE 4

#define _XFER 0
#define _INP_END _XFER + XFER_SIZE
#define _INP _INP_END + INP_END_SIZE
#define _TMP _INP + INP_SIZE
#define STACK_SIZE _TMP + TMP_SIZE

#define ROUND_AND_SCHED_N_0(disp, a, b, c, d, e, f, g, h, XDWORD0, XDWORD1, XDWORD2, XDWORD3) \
	;                                     \ // #############################  RND N + 0 ############################//
	MOVL     a, y3;                       \ // y3 = a					// MAJA
	RORXL    $25, e, y0;                  \ // y0 = e >> 25				// S1A
	RORXL    $11, e, y1;                  \ // y1 = e >> 11				// S1B
	;                                     \
	ADDL     (disp + 0*4)(SP)(SRND*1), h; \ // h = k + w + h        // disp = k + w
	ORL      c, y3;                       \ // y3 = a|c				// MAJA
	VPALIGNR $4, XDWORD2, XDWORD3, XTMP0; \ // XTMP0 = W[-7]
	MOVL     f, y2;                       \ // y2 = f				// CH
	RORXL    $13, a, T1;                  \ // T1 = a >> 13			// S0B
	;                                     \
	XORL     y1, y0;                      \ // y0 = (e>>25) ^ (e>>11)					// S1
	XORL     g, y2;                       \ // y2 = f^g                              	// CH
	VPADDD   XDWORD0, XTMP0, XTMP0;       \ // XTMP0 = W[-7] + W[-16]	// y1 = (e >> 6)	// S1
	RORXL    $6, e, y1;                   \ // y1 = (e >> 6)						// S1
	;                                     \
	ANDL     e, y2;                       \ // y2 = (f^g)&e                         // CH
	XORL     y1, y0;                      \ // y0 = (e>>25) ^ (e>>11) ^ (e>>6)		// S1
	RORXL    $22, a, y1;                  \ // y1 = a >> 22							// S0A
	ADDL     h, d;                        \ // d = k + w + h + d                     	// --
	;                                     \
	ANDL     b, y3;                       \ // y3 = (a|c)&b							// MAJA
	VPALIGNR $4, XDWORD0, XDWORD1, XTMP1; \ // XTMP1 = W[-15]
	XORL     T1, y1;                      \ // y1 = (a>>22) ^ (a>>13)				// S0
	RORXL    $2, a, T1;                   \ // T1 = (a >> 2)						// S0
	;                                     \
	XORL     g, y2;                       \ // y2 = CH = ((f^g)&e)^g				// CH
	VPSRLD   $7, XTMP1, XTMP2;            \
	XORL     T1, y1;                      \ // y1 = (a>>22) ^ (a>>13) ^ (a>>2)		// S0
	MOVL     a, T1;                       \ // T1 = a								// MAJB
	ANDL     c, T1;                       \ // T1 = a&c								// MAJB
	;                                     \
	ADDL     y0, y2;                      \ // y2 = S1 + CH							// --
	VPSLLD   $(32-7), XTMP1, XTMP3;       \
	ORL      T1, y3;                      \ // y3 = MAJ = (a|c)&b)|(a&c)			// MAJ
	ADDL     y1, h;                       \ // h = k + w + h + S0					// --
	;                                     \
	ADDL     y2, d;                       \ // d = k + w + h + d + S1 + CH = d + t1  // --
	VPOR     XTMP2, XTMP3, XTMP3;         \ // XTMP3 = W[-15] ror 7
	;                                     \
	VPSRLD   $18, XTMP1, XTMP2;           \
	ADDL     y2, h;                       \ // h = k + w + h + S0 + S1 + CH = t1 + S0// --
	ADDL     y3, h                        // h = t1 + S0 + MAJ                     // --

#define ROUND_AND_SCHED_N_1(disp, a, b, c, d, e, f, g, h, XDWORD0, XDWORD1, XDWORD2, XDWORD3) \
	;                                    \ // ################################### RND N + 1 ############################
	;                                    \
	MOVL    a, y3;                       \ // y3 = a                       // MAJA
	RORXL   $25, e, y0;                  \ // y0 = e >> 25					// S1A
	RORXL   $11, e, y1;                  \ // y1 = e >> 11					// S1B
	ADDL    (disp + 1*4)(SP)(SRND*1), h; \ // h = k + w + h         		// --
	ORL     c, y3;                       \ // y3 = a|c						// MAJA
	;                                    \
	VPSRLD  $3, XTMP1, XTMP4;            \ // XTMP4 = W[-15] >> 3
	MOVL    f, y2;                       \ // y2 = f						// CH
	RORXL   $13, a, T1;                  \ // T1 = a >> 13					// S0B
	XORL    y1, y0;                      \ // y0 = (e>>25) ^ (e>>11)		// S1
	XORL    g, y2;                       \ // y2 = f^g						// CH
	;                                    \
	RORXL   $6, e, y1;                   \ // y1 = (e >> 6)				// S1
	XORL    y1, y0;                      \ // y0 = (e>>25) ^ (e>>11) ^ (e>>6)	// S1
	RORXL   $22, a, y1;                  \ // y1 = a >> 22						// S0A
	ANDL    e, y2;                       \ // y2 = (f^g)&e						// CH
	ADDL    h, d;                        \ // d = k + w + h + d				// --
	;                                    \
	VPSLLD  $(32-18), XTMP1, XTMP1;      \
	ANDL    b, y3;                       \ // y3 = (a|c)&b					// MAJA
	XORL    T1, y1;                      \ // y1 = (a>>22) ^ (a>>13)		// S0
	;                                    \
	VPXOR   XTMP1, XTMP3, XTMP3;         \
	RORXL   $2, a, T1;                   \ // T1 = (a >> 2)				// S0
	XORL    g, y2;                       \ // y2 = CH = ((f^g)&e)^g		// CH
	;                                    \
	VPXOR   XTMP2, XTMP3, XTMP3;         \ // XTMP3 = W[-15] ror 7 ^ W[-15] ror 18
	XORL    T1, y1;                      \ // y1 = (a>>22) ^ (a>>13) ^ (a>>2)		// S0
	MOVL    a, T1;                       \ // T1 = a						// MAJB
	ANDL    c, T1;                       \ // T1 = a&c						// MAJB
	ADDL    y0, y2;                      \ // y2 = S1 + CH					// --
	;                                    \
	VPXOR   XTMP4, XTMP3, XTMP1;         \ // XTMP1 = s0
	VPSHUFD $0xFA, XDWORD3, XTMP2;       \ // XTMP2 = W[-2] {BBAA}
	ORL     T1, y3;                      \ // y3 = MAJ = (a|c)&b)|(a&c)             // MAJ
	ADDL    y1, h;                       \ // h = k + w + h + S0                    // --
	;                                    \
	VPADDD  XTMP1, XTMP0, XTMP0;         \ // XTMP0 = W[-16] + W[-7] + s0
	ADDL    y2, d;                       \ // d = k + w + h + d + S1 + CH = d + t1  // --
	ADDL    y2, h;                       \ // h = k + w + h + S0 + S1 + CH = t1 + S0// --
	ADDL    y3, h;                       \ // h = t1 + S0 + MAJ                     // --
	;                                    \
	VPSRLD  $10, XTMP2, XTMP4            // XTMP4 = W[-2] >> 10 {BBAA}

#define ROUND_AND_SCHED_N_2(disp, a, b, c, d, e, f, g, h, XDWORD0, XDWORD1, XDWORD2, XDWORD3) \
	;                                    \ // ################################### RND N + 2 ############################
	;                                    \
	MOVL    a, y3;                       \ // y3 = a							// MAJA
	RORXL   $25, e, y0;                  \ // y0 = e >> 25						// S1A
	ADDL    (disp + 2*4)(SP)(SRND*1), h; \ // h = k + w + h        			// --
	;                                    \
	VPSRLQ  $19, XTMP2, XTMP3;           \ // XTMP3 = W[-2] ror 19 {xBxA}
	RORXL   $11, e, y1;                  \ // y1 = e >> 11						// S1B
	ORL     c, y3;                       \ // y3 = a|c                         // MAJA
	MOVL    f, y2;                       \ // y2 = f                           // CH
	XORL    g, y2;                       \ // y2 = f^g                         // CH
	;                                    \
	RORXL   $13, a, T1;                  \ // T1 = a >> 13						// S0B
	XORL    y1, y0;                      \ // y0 = (e>>25) ^ (e>>11)			// S1
	VPSRLQ  $17, XTMP2, XTMP2;           \ // XTMP2 = W[-2] ror 17 {xBxA}
	ANDL    e, y2;                       \ // y2 = (f^g)&e						// CH
	;                                    \
	RORXL   $6, e, y1;                   \ // y1 = (e >> 6)					// S1
	VPXOR   XTMP3, XTMP2, XTMP2;         \
	ADDL    h, d;                        \ // d = k + w + h + d				// --
	ANDL    b, y3;                       \ // y3 = (a|c)&b						// MAJA
	;                                    \
	XORL    y1, y0;                      \ // y0 = (e>>25) ^ (e>>11) ^ (e>>6)	// S1
	RORXL   $22, a, y1;                  \ // y1 = a >> 22						// S0A
	VPXOR   XTMP2, XTMP4, XTMP4;         \ // XTMP4 = s1 {xBxA}
	XORL    g, y2;                       \ // y2 = CH = ((f^g)&e)^g			// CH
	;                                    \
	MOVL    f, _TMP(SP);                 \
	MOVQ    $shuff_00BA<>(SB), f;        \ // f is used to keep SHUF_00BA
	VPSHUFB (f), XTMP4, XTMP4;           \ // XTMP4 = s1 {00BA}
	MOVL    _TMP(SP), f;                 \ // f is restored
	;                                    \
	XORL    T1, y1;                      \ // y1 = (a>>22) ^ (a>>13)		// S0
	RORXL   $2, a, T1;                   \ // T1 = (a >> 2)				// S0
	VPADDD  XTMP4, XTMP0, XTMP0;         \ // XTMP0 = {..., ..., W[1], W[0]}
	;                                    \
	XORL    T1, y1;                      \ // y1 = (a>>22) ^ (a>>13) ^ (a>>2)	// S0
	MOVL    a, T1;                       \ // T1 = a                                // MAJB
	ANDL    c, T1;                       \ // T1 = a&c                              // MAJB
	ADDL    y0, y2;                      \ // y2 = S1 + CH                          // --
	VPSHUFD $80, XTMP0, XTMP2;           \ // XTMP2 = W[-2] {DDCC}
	;                                    \
	ORL     T1, y3;                      \ // y3 = MAJ = (a|c)&b)|(a&c)             // MAJ
	ADDL    y1, h;                       \ // h = k + w + h + S0                    // --
	ADDL    y2, d;                       \ // d = k + w + h + d + S1 + CH = d + t1  // --
	ADDL    y2, h;                       \ // h = k + w + h + S0 + S1 + CH = t1 + S0// --
	;                                    \
	ADDL    y3, h                        // h = t1 + S0 + MAJ                     // --

#define ROUND_AND_SCHED_N_3(disp, a, b, c, d, e, f, g, h, XDWORD0, XDWORD1, XDWORD2, XDWORD3) \
	;                                    \ // ################################### RND N + 3 ############################
	;                                    \
	MOVL    a, y3;                       \ // y3 = a						// MAJA
	RORXL   $25, e, y0;                  \ // y0 = e >> 25					// S1A
	RORXL   $11, e, y1;                  \ // y1 = e >> 11					// S1B
	ADDL    (disp + 3*4)(SP)(SRND*1), h; \ // h = k + w + h				// --
	ORL     c, y3;                       \ // y3 = a|c                     // MAJA
	;                                    \
	VPSRLD  $10, XTMP2, XTMP5;           \ // XTMP5 = W[-2] >> 10 {DDCC}
	MOVL    f, y2;                       \ // y2 = f						// CH
	RORXL   $13, a, T1;                  \ // T1 = a >> 13					// S0B
	XORL    y1, y0;                      \ // y0 = (e>>25) ^ (e>>11)		// S1
	XORL    g, y2;                       \ // y2 = f^g						// CH
	;                                    \
	VPSRLQ  $19, XTMP2, XTMP3;           \ // XTMP3 = W[-2] ror 19 {xDxC}
	RORXL   $6, e, y1;                   \ // y1 = (e >> 6)				// S1
	ANDL    e, y2;                       \ // y2 = (f^g)&e					// CH
	ADDL    h, d;                        \ // d = k + w + h + d			// --
	ANDL    b, y3;                       \ // y3 = (a|c)&b					// MAJA
	;                                    \
	VPSRLQ  $17, XTMP2, XTMP2;           \ // XTMP2 = W[-2] ror 17 {xDxC}
	XORL    y1, y0;                      \ // y0 = (e>>25) ^ (e>>11) ^ (e>>6)	// S1
	XORL    g, y2;                       \ // y2 = CH = ((f^g)&e)^g			// CH
	;                                    \
	VPXOR   XTMP3, XTMP2, XTMP2;         \
	RORXL   $22, a, y1;                  \ // y1 = a >> 22					// S0A
	ADDL    y0, y2;                      \ // y2 = S1 + CH					// --
	;                                    \
	VPXOR   XTMP2, XTMP5, XTMP5;         \ // XTMP5 = s1 {xDxC}
	XORL    T1, y1;                      \ // y1 = (a>>22) ^ (a>>13)		// S0
	ADDL    y2, d;                       \ // d = k + w + h + d + S1 + CH = d + t1  // --
	;                                    \
	RORXL   $2, a, T1;                   \ // T1 = (a >> 2)				// S0
	;                                    \
	MOVL    f, _TMP(SP);                 \ // Save f
	MOVQ    $shuff_DC00<>(SB), f;        \ // SHUF_00DC
	VPSHUFB (f), XTMP5, XTMP5;           \ // XTMP5 = s1 {DC00}
	MOVL    _TMP(SP), f;                 \ // Restore f
	;                                    \
	VPADDD  XTMP0, XTMP5, XDWORD0;       \ // XDWORD0 = {W[3], W[2], W[1], W[0]}
	XORL    T1, y1;                      \ // y1 = (a>>22) ^ (a>>13) ^ (a>>2)	// S0
	MOVL    a, T1;                       \ // T1 = a							// MAJB
	ANDL    c, T1;                       \ // T1 = a&c							// MAJB
	ORL     T1, y3;                      \ // y3 = MAJ = (a|c)&b)|(a&c)		// MAJ
	;                                    \
	ADDL    y1, h;                       \ // h = k + w + h + S0				// --
	ADDL    y2, h;                       \ // h = k + w + h + S0 + S1 + CH = t1 + S0// --
	ADDL    y3, h                        // h = t1 + S0 + MAJ				// --

#define DO_ROUND_N_0(disp, a, b, c, d, e, f, g, h, old_h) \
	;                                  \ // ################################### RND N + 0 ###########################
	MOVL  f, y2;                       \ // y2 = f					// CH
	RORXL $25, e, y0;                  \ // y0 = e >> 25				// S1A
	RORXL $11, e, y1;                  \ // y1 = e >> 11				// S1B
	XORL  g, y2;                       \ // y2 = f^g					// CH
	;                                  \
	XORL  y1, y0;                      \ // y0 = (e>>25) ^ (e>>11)	// S1
	RORXL $6, e, y1;                   \ // y1 = (e >> 6)			// S1
	ANDL  e, y2;                       \ // y2 = (f^g)&e				// CH
	;                                  \
	XORL  y1, y0;                      \ // y0 = (e>>25) ^ (e>>11) ^ (e>>6)	// S1
	RORXL $13, a, T1;                  \ // T1 = a >> 13						// S0B
	XORL  g, y2;                       \ // y2 = CH = ((f^g)&e)^g			// CH
	RORXL $22, a, y1;                  \ // y1 = a >> 22						// S0A
	MOVL  a, y3;                       \ // y3 = a							// MAJA
	;                                  \
	XORL  T1, y1;                      \ // y1 = (a>>22) ^ (a>>13)			// S0
	RORXL $2, a, T1;                   \ // T1 = (a >> 2)					// S0
	ADDL  (disp + 0*4)(SP)(SRND*1), h; \ // h = k + w + h // --
	ORL   c, y3;                       \ // y3 = a|c							// MAJA
	;                                  \
	XORL  T1, y1;                      \ // y1 = (a>>22) ^ (a>>13) ^ (a>>2)	// S0
	MOVL  a, T1;                       \ // T1 = a							// MAJB
	ANDL  b, y3;                       \ // y3 = (a|c)&b						// MAJA
	ANDL  c, T1;                       \ // T1 = a&c							// MAJB
	ADDL  y0, y2;                      \ // y2 = S1 + CH						// --
	;                                  \
	ADDL  h, d;                        \ // d = k + w + h + d					// --
	ORL   T1, y3;                      \ // y3 = MAJ = (a|c)&b)|(a&c)			// MAJ
	ADDL  y1, h;                       \ // h = k + w + h + S0					// --
	ADDL  y2, d                        // d = k + w + h + d + S1 + CH = d + t1	// --

#define DO_ROUND_N_1(disp, a, b, c, d, e, f, g, h, old_h) \
	;                                  \ // ################################### RND N + 1 ###########################
	ADDL  y2, old_h;                   \ // h = k + w + h + S0 + S1 + CH = t1 + S0 // --
	MOVL  f, y2;                       \ // y2 = f                                // CH
	RORXL $25, e, y0;                  \ // y0 = e >> 25				// S1A
	RORXL $11, e, y1;                  \ // y1 = e >> 11				// S1B
	XORL  g, y2;                       \ // y2 = f^g                             // CH
	;                                  \
	XORL  y1, y0;                      \ // y0 = (e>>25) ^ (e>>11)				// S1
	RORXL $6, e, y1;                   \ // y1 = (e >> 6)						// S1
	ANDL  e, y2;                       \ // y2 = (f^g)&e                         // CH
	ADDL  y3, old_h;                   \ // h = t1 + S0 + MAJ                    // --
	;                                  \
	XORL  y1, y0;                      \ // y0 = (e>>25) ^ (e>>11) ^ (e>>6)		// S1
	RORXL $13, a, T1;                  \ // T1 = a >> 13							// S0B
	XORL  g, y2;                       \ // y2 = CH = ((f^g)&e)^g                // CH
	RORXL $22, a, y1;                  \ // y1 = a >> 22							// S0A
	MOVL  a, y3;                       \ // y3 = a                               // MAJA
	;                                  \
	XORL  T1, y1;                      \ // y1 = (a>>22) ^ (a>>13)				// S0
	RORXL $2, a, T1;                   \ // T1 = (a >> 2)						// S0
	ADDL  (disp + 1*4)(SP)(SRND*1), h; \ // h = k + w + h // --
	ORL   c, y3;                       \ // y3 = a|c                             // MAJA
	;                                  \
	XORL  T1, y1;                      \ // y1 = (a>>22) ^ (a>>13) ^ (a>>2)		// S0
	MOVL  a, T1;                       \ // T1 = a                               // MAJB
	ANDL  b, y3;                       \ // y3 = (a|c)&b                         // MAJA
	ANDL  c, T1;                       \ // T1 = a&c                             // MAJB
	ADDL  y0, y2;                      \ // y2 = S1 + CH                         // --
	;                                  \
	ADDL  h, d;                        \ // d = k + w + h + d                    // --
	ORL   T1, y3;                      \ // y3 = MAJ = (a|c)&b)|(a&c)            // MAJ
	ADDL  y1, h;                       \ // h = k + w + h + S0                   // --
	;                                  \
	ADDL  y2, d                        // d = k + w + h + d + S1 + CH = d + t1 // --

#define DO_ROUND_N_2(disp, a, b, c, d, e, f, g, h, old_h) \
	;                                  \ // ################################### RND N + 2 ##############################
	ADDL  y2, old_h;                   \ // h = k + w + h + S0 + S1 + CH = t1 + S0// --
	MOVL  f, y2;                       \ // y2 = f								// CH
	RORXL $25, e, y0;                  \ // y0 = e >> 25							// S1A
	RORXL $11, e, y1;                  \ // y1 = e >> 11							// S1B
	XORL  g, y2;                       \ // y2 = f^g								// CH
	;                                  \
	XORL  y1, y0;                      \ // y0 = (e>>25) ^ (e>>11)				// S1
	RORXL $6, e, y1;                   \ // y1 = (e >> 6)						// S1
	ANDL  e, y2;                       \ // y2 = (f^g)&e							// CH
	ADDL  y3, old_h;                   \ // h = t1 + S0 + MAJ					// --
	;                                  \
	XORL  y1, y0;                      \ // y0 = (e>>25) ^ (e>>11) ^ (e>>6)		// S1
	RORXL $13, a, T1;                  \ // T1 = a >> 13							// S0B
	XORL  g, y2;                       \ // y2 = CH = ((f^g)&e)^g                // CH
	RORXL $22, a, y1;                  \ // y1 = a >> 22							// S0A
	MOVL  a, y3;                       \ // y3 = a								// MAJA
	;                                  \
	XORL  T1, y1;                      \ // y1 = (a>>22) ^ (a>>13)				// S0
	RORXL $2, a, T1;                   \ // T1 = (a >> 2)						// S0
	ADDL  (disp + 2*4)(SP)(SRND*1), h; \ // h = k + w + h 	// --
	ORL   c, y3;                       \ // y3 = a|c								// MAJA
	;                                  \
	XORL  T1, y1;                      \ // y1 = (a>>22) ^ (a>>13) ^ (a>>2)		// S0
	MOVL  a, T1;                       \ // T1 = a								// MAJB
	ANDL  b, y3;                       \ // y3 = (a|c)&b							// MAJA
	ANDL  c, T1;                       \ // T1 = a&c								// MAJB
	ADDL  y0, y2;                      \ // y2 = S1 + CH							// --
	;                                  \
	ADDL  h, d;                        \ // d = k + w + h + d					// --
	ORL   T1, y3;                      \ // y3 = MAJ = (a|c)&b)|(a&c)			// MAJ
	ADDL  y1, h;                       \ // h = k + w + h + S0					// --
	;                                  \
	ADDL  y2, d                        // d = k + w + h + d + S1 + CH = d + t1 // --

#define DO_ROUND_N_3(disp, a, b, c, d, e, f, g, h, old_h) \
	;                                  \ // ################################### RND N + 3 ###########################
	ADDL  y2, old_h;                   \ // h = k + w + h + S0 + S1 + CH = t1 + S0// --
	MOVL  f, y2;                       \ // y2 = f								// CH
	RORXL $25, e, y0;                  \ // y0 = e >> 25							// S1A
	RORXL $11, e, y1;                  \ // y1 = e >> 11							// S1B
	XORL  g, y2;                       \ // y2 = f^g								// CH
	;                                  \
	XORL  y1, y0;                      \ // y0 = (e>>25) ^ (e>>11)				// S1
	RORXL $6, e, y1;                   \ // y1 = (e >> 6)						// S1
	ANDL  e, y2;                       \ // y2 = (f^g)&e							// CH
	ADDL  y3, old_h;                   \ // h = t1 + S0 + MAJ					// --
	;                                  \
	XORL  y1, y0;                      \ // y0 = (e>>25) ^ (e>>11) ^ (e>>6)		// S1
	RORXL $13, a, T1;                  \ // T1 = a >> 13							// S0B
	XORL  g, y2;                       \ // y2 = CH = ((f^g)&e)^g				// CH
	RORXL $22, a, y1;                  \ // y1 = a >> 22							// S0A
	MOVL  a, y3;                       \ // y3 = a								// MAJA
	;                                  \
	XORL  T1, y1;                      \ // y1 = (a>>22) ^ (a>>13)				// S0
	RORXL $2, a, T1;                   \ // T1 = (a >> 2)						// S0
	ADDL  (disp + 3*4)(SP)(SRND*1), h; \ // h = k + w + h 	// --
	ORL   c, y3;                       \ // y3 = a|c								// MAJA
	;                                  \
	XORL  T1, y1;                      \ // y1 = (a>>22) ^ (a>>13) ^ (a>>2)		// S0
	MOVL  a, T1;                       \ // T1 = a								// MAJB
	ANDL  b, y3;                       \ // y3 = (a|c)&b							// MAJA
	ANDL  c, T1;                       \ // T1 = a&c								// MAJB
	ADDL  y0, y2;                      \ // y2 = S1 + CH							// --
	;                                  \
	ADDL  h, d;                        \ // d = k + w + h + d					// --
	ORL   T1, y3;                      \ // y3 = MAJ = (a|c)&b)|(a&c)			// MAJ
	ADDL  y1, h;                       \ // h = k + w + h + S0					// --
	;                                  \
	ADDL  y2, d;                       \ // d = k + w + h + d + S1 + CH = d + t1	// --
	;                                  \
	ADDL  y2, h;                       \ // h = k + w + h + S0 + S1 + CH = t1 + S0// --
	;                                  \
	ADDL  y3, h                        // h = t1 + S0 + MAJ					// --

TEXT ·block(SB), 0, $536-32
	CMPB ·useAVX2(SB), $1
	JE   avx2

	MOVQ p_base+8(FP), SI
	MOVQ p_len+16(FP), DX
	SHRQ $6, DX
	SHLQ $6, DX

	LEAQ (SI)(DX*1), DI
	MOVQ DI, 256(SP)
	CMPQ SI, DI
	JEQ  end

	MOVQ dig+0(FP), BP
	MOVL (0*4)(BP), R8  // a = H0
	MOVL (1*4)(BP), R9  // b = H1
	MOVL (2*4)(BP), R10 // c = H2
	MOVL (3*4)(BP), R11 // d = H3
	MOVL (4*4)(BP), R12 // e = H4
	MOVL (5*4)(BP), R13 // f = H5
	MOVL (6*4)(BP), R14 // g = H6
	MOVL (7*4)(BP), R15 // h = H7

loop:
	MOVQ SP, BP

	SHA256ROUND0(0, 0x428a2f98, R8, R9, R10, R11, R12, R13, R14, R15)
	SHA256ROUND0(1, 0x71374491, R15, R8, R9, R10, R11, R12, R13, R14)
	SHA256ROUND0(2, 0xb5c0fbcf, R14, R15, R8, R9, R10, R11, R12, R13)
	SHA256ROUND0(3, 0xe9b5dba5, R13, R14, R15, R8, R9, R10, R11, R12)
	SHA256ROUND0(4, 0x3956c25b, R12, R13, R14, R15, R8, R9, R10, R11)
	SHA256ROUND0(5, 0x59f111f1, R11, R12, R13, R14, R15, R8, R9, R10)
	SHA256ROUND0(6, 0x923f82a4, R10, R11, R12, R13, R14, R15, R8, R9)
	SHA256ROUND0(7, 0xab1c5ed5, R9, R10, R11, R12, R13, R14, R15, R8)
	SHA256ROUND0(8, 0xd807aa98, R8, R9, R10, R11, R12, R13, R14, R15)
	SHA256ROUND0(9, 0x12835b01, R15, R8, R9, R10, R11, R12, R13, R14)
	SHA256ROUND0(10, 0x243185be, R14, R15, R8, R9, R10, R11, R12, R13)
	SHA256ROUND0(11, 0x550c7dc3, R13, R14, R15, R8, R9, R10, R11, R12)
	SHA256ROUND0(12, 0x72be5d74, R12, R13, R14, R15, R8, R9, R10, R11)
	SHA256ROUND0(13, 0x80deb1fe, R11, R12, R13, R14, R15, R8, R9, R10)
	SHA256ROUND0(14, 0x9bdc06a7, R10, R11, R12, R13, R14, R15, R8, R9)
	SHA256ROUND0(15, 0xc19bf174, R9, R10, R11, R12, R13, R14, R15, R8)

	SHA256ROUND1(16, 0xe49b69c1, R8, R9, R10, R11, R12, R13, R14, R15)
	SHA256ROUND1(17, 0xefbe4786, R15, R8, R9, R10, R11, R12, R13, R14)
	SHA256ROUND1(18, 0x0fc19dc6, R14, R15, R8, R9, R10, R11, R12, R13)
	SHA256ROUND1(19, 0x240ca1cc, R13, R14, R15, R8, R9, R10, R11, R12)
	SHA256ROUND1(20, 0x2de92c6f, R12, R13, R14, R15, R8, R9, R10, R11)
	SHA256ROUND1(21, 0x4a7484aa, R11, R12, R13, R14, R15, R8, R9, R10)
	SHA256ROUND1(22, 0x5cb0a9dc, R10, R11, R12, R13, R14, R15, R8, R9)
	SHA256ROUND1(23, 0x76f988da, R9, R10, R11, R12, R13, R14, R15, R8)
	SHA256ROUND1(24, 0x983e5152, R8, R9, R10, R11, R12, R13, R14, R15)
	SHA256ROUND1(25, 0xa831c66d, R15, R8, R9, R10, R11, R12, R13, R14)
	SHA256ROUND1(26, 0xb00327c8, R14, R15, R8, R9, R10, R11, R12, R13)
	SHA256ROUND1(27, 0xbf597fc7, R13, R14, R15, R8, R9, R10, R11, R12)
	SHA256ROUND1(28, 0xc6e00bf3, R12, R13, R14, R15, R8, R9, R10, R11)
	SHA256ROUND1(29, 0xd5a79147, R11, R12, R13, R14, R15, R8, R9, R10)
	SHA256ROUND1(30, 0x06ca6351, R10, R11, R12, R13, R14, R15, R8, R9)
	SHA256ROUND1(31, 0x14292967, R9, R10, R11, R12, R13, R14, R15, R8)
	SHA256ROUND1(32, 0x27b70a85, R8, R9, R10, R11, R12, R13, R14, R15)
	SHA256ROUND1(33, 0x2e1b2138, R15, R8, R9, R10, R11, R12, R13, R14)
	SHA256ROUND1(34, 0x4d2c6dfc, R14, R15, R8, R9, R10, R11, R12, R13)
	SHA256ROUND1(35, 0x53380d13, R13, R14, R15, R8, R9, R10, R11, R12)
	SHA256ROUND1(36, 0x650a7354, R12, R13, R14, R15, R8, R9, R10, R11)
	SHA256ROUND1(37, 0x766a0abb, R11, R12, R13, R14, R15, R8, R9, R10)
	SHA256ROUND1(38, 0x81c2c92e, R10, R11, R12, R13, R14, R15, R8, R9)
	SHA256ROUND1(39, 0x92722c85, R9, R10, R11, R12, R13, R14, R15, R8)
	SHA256ROUND1(40, 0xa2bfe8a1, R8, R9, R10, R11, R12, R13, R14, R15)
	SHA256ROUND1(41, 0xa81a664b, R15, R8, R9, R10, R11, R12, R13, R14)
	SHA256ROUND1(42, 0xc24b8b70, R14, R15, R8, R9, R10, R11, R12, R13)
	SHA256ROUND1(43, 0xc76c51a3, R13, R14, R15, R8, R9, R10, R11, R12)
	SHA256ROUND1(44, 0xd192e819, R12, R13, R14, R15, R8, R9, R10, R11)
	SHA256ROUND1(45, 0xd6990624, R11, R12, R13, R14, R15, R8, R9, R10)
	SHA256ROUND1(46, 0xf40e3585, R10, R11, R12, R13, R14, R15, R8, R9)
	SHA256ROUND1(47, 0x106aa070, R9, R10, R11, R12, R13, R14, R15, R8)
	SHA256ROUND1(48, 0x19a4c116, R8, R9, R10, R11, R12, R13, R14, R15)
	SHA256ROUND1(49, 0x1e376c08, R15, R8, R9, R10, R11, R12, R13, R14)
	SHA256ROUND1(50, 0x2748774c, R14, R15, R8, R9, R10, R11, R12, R13)
	SHA256ROUND1(51, 0x34b0bcb5, R13, R14, R15, R8, R9, R10, R11, R12)
	SHA256ROUND1(52, 0x391c0cb3, R12, R13, R14, R15, R8, R9, R10, R11)
	SHA256ROUND1(53, 0x4ed8aa4a, R11, R12, R13, R14, R15, R8, R9, R10)
	SHA256ROUND1(54, 0x5b9cca4f, R10, R11, R12, R13, R14, R15, R8, R9)
	SHA256ROUND1(55, 0x682e6ff3, R9, R10, R11, R12, R13, R14, R15, R8)
	SHA256ROUND1(56, 0x748f82ee, R8, R9, R10, R11, R12, R13, R14, R15)
	SHA256ROUND1(57, 0x78a5636f, R15, R8, R9, R10, R11, R12, R13, R14)
	SHA256ROUND1(58, 0x84c87814, R14, R15, R8, R9, R10, R11, R12, R13)
	SHA256ROUND1(59, 0x8cc70208, R13, R14, R15, R8, R9, R10, R11, R12)
	SHA256ROUND1(60, 0x90befffa, R12, R13, R14, R15, R8, R9, R10, R11)
	SHA256ROUND1(61, 0xa4506ceb, R11, R12, R13, R14, R15, R8, R9, R10)
	SHA256ROUND1(62, 0xbef9a3f7, R10, R11, R12, R13, R14, R15, R8, R9)
	SHA256ROUND1(63, 0xc67178f2, R9, R10, R11, R12, R13, R14, R15, R8)

	MOVQ dig+0(FP), BP
	ADDL (0*4)(BP), R8  // H0 = a + H0
	MOVL R8, (0*4)(BP)
	ADDL (1*4)(BP), R9  // H1 = b + H1
	MOVL R9, (1*4)(BP)
	ADDL (2*4)(BP), R10 // H2 = c + H2
	MOVL R10, (2*4)(BP)
	ADDL (3*4)(BP), R11 // H3 = d + H3
	MOVL R11, (3*4)(BP)
	ADDL (4*4)(BP), R12 // H4 = e + H4
	MOVL R12, (4*4)(BP)
	ADDL (5*4)(BP), R13 // H5 = f + H5
	MOVL R13, (5*4)(BP)
	ADDL (6*4)(BP), R14 // H6 = g + H6
	MOVL R14, (6*4)(BP)
	ADDL (7*4)(BP), R15 // H7 = h + H7
	MOVL R15, (7*4)(BP)

	ADDQ $64, SI
	CMPQ SI, 256(SP)
	JB   loop

end:
	RET

avx2:
	MOVQ dig+0(FP), CTX          // d.h[8]
	MOVQ p_base+8(FP), INP
	MOVQ p_len+16(FP), NUM_BYTES

	LEAQ -64(INP)(NUM_BYTES*1), NUM_BYTES // Pointer to the last block
	MOVQ NUM_BYTES, _INP_END(SP)

	CMPQ NUM_BYTES, INP
	JE   avx2_only_one_block

	// Load initial digest
	MOVL 0(CTX), a  // a = H0
	MOVL 4(CTX), b  // b = H1
	MOVL 8(CTX), c  // c = H2
	MOVL 12(CTX), d // d = H3
	MOVL 16(CTX), e // e = H4
	MOVL 20(CTX), f // f = H5
	MOVL 24(CTX), g // g = H6
	MOVL 28(CTX), h // h = H7

avx2_loop0: // at each iteration works with one block (512 bit)

	VMOVDQU (0*32)(INP), XTMP0
	VMOVDQU (1*32)(INP), XTMP1
	VMOVDQU (2*32)(INP), XTMP2
	VMOVDQU (3*32)(INP), XTMP3

	MOVQ    $flip_mask<>(SB), BP // BYTE_FLIP_MASK
	VMOVDQU (BP), BYTE_FLIP_MASK

	// Apply Byte Flip Mask: LE -> BE
	VPSHUFB BYTE_FLIP_MASK, XTMP0, XTMP0
	VPSHUFB BYTE_FLIP_MASK, XTMP1, XTMP1
	VPSHUFB BYTE_FLIP_MASK, XTMP2, XTMP2
	VPSHUFB BYTE_FLIP_MASK, XTMP3, XTMP3

	// Transpose data into high/low parts
	VPERM2I128 $0x20, XTMP2, XTMP0, XDWORD0 // w3, w2, w1, w0
	VPERM2I128 $0x31, XTMP2, XTMP0, XDWORD1 // w7, w6, w5, w4
	VPERM2I128 $0x20, XTMP3, XTMP1, XDWORD2 // w11, w10, w9, w8
	VPERM2I128 $0x31, XTMP3, XTMP1, XDWORD3 // w15, w14, w13, w12

	MOVQ $K256<>(SB), TBL // Loading address of table with round-specific constants

avx2_last_block_enter:
	ADDQ $64, INP
	MOVQ INP, _INP(SP)
	XORQ SRND, SRND

avx2_loop1: // for w0 - w47
	// Do 4 rounds and scheduling
	VPADDD  0*32(TBL)(SRND*1), XDWORD0, XFER
	VMOVDQU XFER, (_XFER + 0*32)(SP)(SRND*1)
	ROUND_AND_SCHED_N_0(_XFER + 0*32, a, b, c, d, e, f, g, h, XDWORD0, XDWORD1, XDWORD2, XDWORD3)
	ROUND_AND_SCHED_N_1(_XFER + 0*32, h, a, b, c, d, e, f, g, XDWORD0, XDWORD1, XDWORD2, XDWORD3)
	ROUND_AND_SCHED_N_2(_XFER + 0*32, g, h, a, b, c, d, e, f, XDWORD0, XDWORD1, XDWORD2, XDWORD3)
	ROUND_AND_SCHED_N_3(_XFER + 0*32, f, g, h, a, b, c, d, e, XDWORD0, XDWORD1, XDWORD2, XDWORD3)

	// Do 4 rounds and scheduling
	VPADDD  1*32(TBL)(SRND*1), XDWORD1, XFER
	VMOVDQU XFER, (_XFER + 1*32)(SP)(SRND*1)
	ROUND_AND_SCHED_N_0(_XFER + 1*32, e, f, g, h, a, b, c, d, XDWORD1, XDWORD2, XDWORD3, XDWORD0)
	ROUND_AND_SCHED_N_1(_XFER + 1*32, d, e, f, g, h, a, b, c, XDWORD1, XDWORD2, XDWORD3, XDWORD0)
	ROUND_AND_SCHED_N_2(_XFER + 1*32, c, d, e, f, g, h, a, b, XDWORD1, XDWORD2, XDWORD3, XDWORD0)
	ROUND_AND_SCHED_N_3(_XFER + 1*32, b, c, d, e, f, g, h, a, XDWORD1, XDWORD2, XDWORD3, XDWORD0)

	// Do 4 rounds and scheduling
	VPADDD  2*32(TBL)(SRND*1), XDWORD2, XFER
	VMOVDQU XFER, (_XFER + 2*32)(SP)(SRND*1)
	ROUND_AND_SCHED_N_0(_XFER + 2*32, a, b, c, d, e, f, g, h, XDWORD2, XDWORD3, XDWORD0, XDWORD1)
	ROUND_AND_SCHED_N_1(_XFER + 2*32, h, a, b, c, d, e, f, g, XDWORD2, XDWORD3, XDWORD0, XDWORD1)
	ROUND_AND_SCHED_N_2(_XFER + 2*32, g, h, a, b, c, d, e, f, XDWORD2, XDWORD3, XDWORD0, XDWORD1)
	ROUND_AND_SCHED_N_3(_XFER + 2*32, f, g, h, a, b, c, d, e, XDWORD2, XDWORD3, XDWORD0, XDWORD1)

	// Do 4 rounds and scheduling
	VPADDD  3*32(TBL)(SRND*1), XDWORD3, XFER
	VMOVDQU XFER, (_XFER + 3*32)(SP)(SRND*1)
	ROUND_AND_SCHED_N_0(_XFER + 3*32, e, f, g, h, a, b, c, d, XDWORD3, XDWORD0, XDWORD1, XDWORD2)
	ROUND_AND_SCHED_N_1(_XFER + 3*32, d, e, f, g, h, a, b, c, XDWORD3, XDWORD0, XDWORD1, XDWORD2)
	ROUND_AND_SCHED_N_2(_XFER + 3*32, c, d, e, f, g, h, a, b, XDWORD3, XDWORD0, XDWORD1, XDWORD2)
	ROUND_AND_SCHED_N_3(_XFER + 3*32, b, c, d, e, f, g, h, a, XDWORD3, XDWORD0, XDWORD1, XDWORD2)

	ADDQ $4*32, SRND
	CMPQ SRND, $3*4*32
	JB   avx2_loop1

avx2_loop2:
	// w48 - w63 processed with no scheduliung (last 16 rounds)
	VPADDD  0*32(TBL)(SRND*1), XDWORD0, XFER
	VMOVDQU XFER, (_XFER + 0*32)(SP)(SRND*1)
	DO_ROUND_N_0(_XFER + 0*32, a, b, c, d, e, f, g, h, h)
	DO_ROUND_N_1(_XFER + 0*32, h, a, b, c, d, e, f, g, h)
	DO_ROUND_N_2(_XFER + 0*32, g, h, a, b, c, d, e, f, g)
	DO_ROUND_N_3(_XFER + 0*32, f, g, h, a, b, c, d, e, f)

	VPADDD  1*32(TBL)(SRND*1), XDWORD1, XFER
	VMOVDQU XFER, (_XFER + 1*32)(SP)(SRND*1)
	DO_ROUND_N_0(_XFER + 1*32, e, f, g, h, a, b, c, d, e)
	DO_ROUND_N_1(_XFER + 1*32, d, e, f, g, h, a, b, c, d)
	DO_ROUND_N_2(_XFER + 1*32, c, d, e, f, g, h, a, b, c)
	DO_ROUND_N_3(_XFER + 1*32, b, c, d, e, f, g, h, a, b)

	ADDQ $2*32, SRND

	VMOVDQU XDWORD2, XDWORD0
	VMOVDQU XDWORD3, XDWORD1

	CMPQ SRND, $4*4*32
	JB   avx2_loop2

	MOVQ dig+0(FP), CTX // d.h[8]
	MOVQ _INP(SP), INP

	addm(  0(CTX), a)
	addm(  4(CTX), b)
	addm(  8(CTX), c)
	addm( 12(CTX), d)
	addm( 16(CTX), e)
	addm( 20(CTX), f)
	addm( 24(CTX), g)
	addm( 28(CTX), h)

	CMPQ _INP_END(SP), INP
	JB   done_hash

	XORQ SRND, SRND

avx2_loop3: // Do second block using previously scheduled results
	DO_ROUND_N_0(_XFER + 0*32 + 16, a, b, c, d, e, f, g, h, a)
	DO_ROUND_N_1(_XFER + 0*32 + 16, h, a, b, c, d, e, f, g, h)
	DO_ROUND_N_2(_XFER + 0*32 + 16, g, h, a, b, c, d, e, f, g)
	DO_ROUND_N_3(_XFER + 0*32 + 16, f, g, h, a, b, c, d, e, f)

	DO_ROUND_N_0(_XFER + 1*32 + 16, e, f, g, h, a, b, c, d, e)
	DO_ROUND_N_1(_XFER + 1*32 + 16, d, e, f, g, h, a, b, c, d)
	DO_ROUND_N_2(_XFER + 1*32 + 16, c, d, e, f, g, h, a, b, c)
	DO_ROUND_N_3(_XFER + 1*32 + 16, b, c, d, e, f, g, h, a, b)

	ADDQ $2*32, SRND
	CMPQ SRND, $4*4*32
	JB   avx2_loop3

	MOVQ dig+0(FP), CTX // d.h[8]
	MOVQ _INP(SP), INP
	ADDQ $64, INP

	addm(  0(CTX), a)
	addm(  4(CTX), b)
	addm(  8(CTX), c)
	addm( 12(CTX), d)
	addm( 16(CTX), e)
	addm( 20(CTX), f)
	addm( 24(CTX), g)
	addm( 28(CTX), h)

	CMPQ _INP_END(SP), INP
	JA   avx2_loop0
	JB   done_hash

avx2_do_last_block:

	VMOVDQU 0(INP), XWORD0
	VMOVDQU 16(INP), XWORD1
	VMOVDQU 32(INP), XWORD2
	VMOVDQU 48(INP), XWORD3

	MOVQ    $flip_mask<>(SB), BP
	VMOVDQU (BP), X_BYTE_FLIP_MASK

	VPSHUFB X_BYTE_FLIP_MASK, XWORD0, XWORD0
	VPSHUFB X_BYTE_FLIP_MASK, XWORD1, XWORD1
	VPSHUFB X_BYTE_FLIP_MASK, XWORD2, XWORD2
	VPSHUFB X_BYTE_FLIP_MASK, XWORD3, XWORD3

	MOVQ $K256<>(SB), TBL

	JMP avx2_last_block_enter

avx2_only_one_block:
	// Load initial digest
	MOVL 0(CTX), a  // a = H0
	MOVL 4(CTX), b  // b = H1
	MOVL 8(CTX), c  // c = H2
	MOVL 12(CTX), d // d = H3
	MOVL 16(CTX), e // e = H4
	MOVL 20(CTX), f // f = H5
	MOVL 24(CTX), g // g = H6
	MOVL 28(CTX), h // h = H7

	JMP avx2_do_last_block

done_hash:
	VZEROUPPER
	RET

// shuffle byte order from LE to BE
DATA flip_mask<>+0x00(SB)/8, $0x0405060700010203
DATA flip_mask<>+0x08(SB)/8, $0x0c0d0e0f08090a0b
DATA flip_mask<>+0x10(SB)/8, $0x0405060700010203
DATA flip_mask<>+0x18(SB)/8, $0x0c0d0e0f08090a0b
GLOBL flip_mask<>(SB), 8, $32

// shuffle xBxA -> 00BA
DATA shuff_00BA<>+0x00(SB)/8, $0x0b0a090803020100
DATA shuff_00BA<>+0x08(SB)/8, $0xFFFFFFFFFFFFFFFF
DATA shuff_00BA<>+0x10(SB)/8, $0x0b0a090803020100
DATA shuff_00BA<>+0x18(SB)/8, $0xFFFFFFFFFFFFFFFF
GLOBL shuff_00BA<>(SB), 8, $32

// shuffle xDxC -> DC00
DATA shuff_DC00<>+0x00(SB)/8, $0xFFFFFFFFFFFFFFFF
DATA shuff_DC00<>+0x08(SB)/8, $0x0b0a090803020100
DATA shuff_DC00<>+0x10(SB)/8, $0xFFFFFFFFFFFFFFFF
DATA shuff_DC00<>+0x18(SB)/8, $0x0b0a090803020100
GLOBL shuff_DC00<>(SB), 8, $32

// Round specific constants
DATA K256<>+0x00(SB)/4, $0x428a2f98 // k1
DATA K256<>+0x04(SB)/4, $0x71374491 // k2
DATA K256<>+0x08(SB)/4, $0xb5c0fbcf // k3
DATA K256<>+0x0c(SB)/4, $0xe9b5dba5 // k4
DATA K256<>+0x10(SB)/4, $0x428a2f98 // k1
DATA K256<>+0x14(SB)/4, $0x71374491 // k2
DATA K256<>+0x18(SB)/4, $0xb5c0fbcf // k3
DATA K256<>+0x1c(SB)/4, $0xe9b5dba5 // k4

DATA K256<>+0x20(SB)/4, $0x3956c25b // k5 - k8
DATA K256<>+0x24(SB)/4, $0x59f111f1
DATA K256<>+0x28(SB)/4, $0x923f82a4
DATA K256<>+0x2c(SB)/4, $0xab1c5ed5
DATA K256<>+0x30(SB)/4, $0x3956c25b
DATA K256<>+0x34(SB)/4, $0x59f111f1
DATA K256<>+0x38(SB)/4, $0x923f82a4
DATA K256<>+0x3c(SB)/4, $0xab1c5ed5

DATA K256<>+0x40(SB)/4, $0xd807aa98 // k9 - k12
DATA K256<>+0x44(SB)/4, $0x12835b01
DATA K256<>+0x48(SB)/4, $0x243185be
DATA K256<>+0x4c(SB)/4, $0x550c7dc3
DATA K256<>+0x50(SB)/4, $0xd807aa98
DATA K256<>+0x54(SB)/4, $0x12835b01
DATA K256<>+0x58(SB)/4, $0x243185be
DATA K256<>+0x5c(SB)/4, $0x550c7dc3

DATA K256<>+0x60(SB)/4, $0x72be5d74 // k13 - k16
DATA K256<>+0x64(SB)/4, $0x80deb1fe
DATA K256<>+0x68(SB)/4, $0x9bdc06a7
DATA K256<>+0x6c(SB)/4, $0xc19bf174
DATA K256<>+0x70(SB)/4, $0x72be5d74
DATA K256<>+0x74(SB)/4, $0x80deb1fe
DATA K256<>+0x78(SB)/4, $0x9bdc06a7
DATA K256<>+0x7c(SB)/4, $0xc19bf174

DATA K256<>+0x80(SB)/4, $0xe49b69c1 // k17 - k20
DATA K256<>+0x84(SB)/4, $0xefbe4786
DATA K256<>+0x88(SB)/4, $0x0fc19dc6
DATA K256<>+0x8c(SB)/4, $0x240ca1cc
DATA K256<>+0x90(SB)/4, $0xe49b69c1
DATA K256<>+0x94(SB)/4, $0xefbe4786
DATA K256<>+0x98(SB)/4, $0x0fc19dc6
DATA K256<>+0x9c(SB)/4, $0x240ca1cc

DATA K256<>+0xa0(SB)/4, $0x2de92c6f // k21 - k24
DATA K256<>+0xa4(SB)/4, $0x4a7484aa
DATA K256<>+0xa8(SB)/4, $0x5cb0a9dc
DATA K256<>+0xac(SB)/4, $0x76f988da
DATA K256<>+0xb0(SB)/4, $0x2de92c6f
DATA K256<>+0xb4(SB)/4, $0x4a7484aa
DATA K256<>+0xb8(SB)/4, $0x5cb0a9dc
DATA K256<>+0xbc(SB)/4, $0x76f988da

DATA K256<>+0xc0(SB)/4, $0x983e5152 // k25 - k28
DATA K256<>+0xc4(SB)/4, $0xa831c66d
DATA K256<>+0xc8(SB)/4, $0xb00327c8
DATA K256<>+0xcc(SB)/4, $0xbf597fc7
DATA K256<>+0xd0(SB)/4, $0x983e5152
DATA K256<>+0xd4(SB)/4, $0xa831c66d
DATA K256<>+0xd8(SB)/4, $0xb00327c8
DATA K256<>+0xdc(SB)/4, $0xbf597fc7

DATA K256<>+0xe0(SB)/4, $0xc6e00bf3 // k29 - k32
DATA K256<>+0xe4(SB)/4, $0xd5a79147
DATA K256<>+0xe8(SB)/4, $0x06ca6351
DATA K256<>+0xec(SB)/4, $0x14292967
DATA K256<>+0xf0(SB)/4, $0xc6e00bf3
DATA K256<>+0xf4(SB)/4, $0xd5a79147
DATA K256<>+0xf8(SB)/4, $0x06ca6351
DATA K256<>+0xfc(SB)/4, $0x14292967

DATA K256<>+0x100(SB)/4, $0x27b70a85
DATA K256<>+0x104(SB)/4, $0x2e1b2138
DATA K256<>+0x108(SB)/4, $0x4d2c6dfc
DATA K256<>+0x10c(SB)/4, $0x53380d13
DATA K256<>+0x110(SB)/4, $0x27b70a85
DATA K256<>+0x114(SB)/4, $0x2e1b2138
DATA K256<>+0x118(SB)/4, $0x4d2c6dfc
DATA K256<>+0x11c(SB)/4, $0x53380d13

DATA K256<>+0x120(SB)/4, $0x650a7354
DATA K256<>+0x124(SB)/4, $0x766a0abb
DATA K256<>+0x128(SB)/4, $0x81c2c92e
DATA K256<>+0x12c(SB)/4, $0x92722c85
DATA K256<>+0x130(SB)/4, $0x650a7354
DATA K256<>+0x134(SB)/4, $0x766a0abb
DATA K256<>+0x138(SB)/4, $0x81c2c92e
DATA K256<>+0x13c(SB)/4, $0x92722c85

DATA K256<>+0x140(SB)/4, $0xa2bfe8a1
DATA K256<>+0x144(SB)/4, $0xa81a664b
DATA K256<>+0x148(SB)/4, $0xc24b8b70
DATA K256<>+0x14c(SB)/4, $0xc76c51a3
DATA K256<>+0x150(SB)/4, $0xa2bfe8a1
DATA K256<>+0x154(SB)/4, $0xa81a664b
DATA K256<>+0x158(SB)/4, $0xc24b8b70
DATA K256<>+0x15c(SB)/4, $0xc76c51a3

DATA K256<>+0x160(SB)/4, $0xd192e819
DATA K256<>+0x164(SB)/4, $0xd6990624
DATA K256<>+0x168(SB)/4, $0xf40e3585
DATA K256<>+0x16c(SB)/4, $0x106aa070
DATA K256<>+0x170(SB)/4, $0xd192e819
DATA K256<>+0x174(SB)/4, $0xd6990624
DATA K256<>+0x178(SB)/4, $0xf40e3585
DATA K256<>+0x17c(SB)/4, $0x106aa070

DATA K256<>+0x180(SB)/4, $0x19a4c116
DATA K256<>+0x184(SB)/4, $0x1e376c08
DATA K256<>+0x188(SB)/4, $0x2748774c
DATA K256<>+0x18c(SB)/4, $0x34b0bcb5
DATA K256<>+0x190(SB)/4, $0x19a4c116
DATA K256<>+0x194(SB)/4, $0x1e376c08
DATA K256<>+0x198(SB)/4, $0x2748774c
DATA K256<>+0x19c(SB)/4, $0x34b0bcb5

DATA K256<>+0x1a0(SB)/4, $0x391c0cb3
DATA K256<>+0x1a4(SB)/4, $0x4ed8aa4a
DATA K256<>+0x1a8(SB)/4, $0x5b9cca4f
DATA K256<>+0x1ac(SB)/4, $0x682e6ff3
DATA K256<>+0x1b0(SB)/4, $0x391c0cb3
DATA K256<>+0x1b4(SB)/4, $0x4ed8aa4a
DATA K256<>+0x1b8(SB)/4, $0x5b9cca4f
DATA K256<>+0x1bc(SB)/4, $0x682e6ff3

DATA K256<>+0x1c0(SB)/4, $0x748f82ee
DATA K256<>+0x1c4(SB)/4, $0x78a5636f
DATA K256<>+0x1c8(SB)/4, $0x84c87814
DATA K256<>+0x1cc(SB)/4, $0x8cc70208
DATA K256<>+0x1d0(SB)/4, $0x748f82ee
DATA K256<>+0x1d4(SB)/4, $0x78a5636f
DATA K256<>+0x1d8(SB)/4, $0x84c87814
DATA K256<>+0x1dc(SB)/4, $0x8cc70208

DATA K256<>+0x1e0(SB)/4, $0x90befffa
DATA K256<>+0x1e4(SB)/4, $0xa4506ceb
DATA K256<>+0x1e8(SB)/4, $0xbef9a3f7
DATA K256<>+0x1ec(SB)/4, $0xc67178f2
DATA K256<>+0x1f0(SB)/4, $0x90befffa
DATA K256<>+0x1f4(SB)/4, $0xa4506ceb
DATA K256<>+0x1f8(SB)/4, $0xbef9a3f7
DATA K256<>+0x1fc(SB)/4, $0xc67178f2

GLOBL K256<>(SB), (NOPTR + RODATA), $512
