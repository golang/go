// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.11,!gccgo,!purego

#include "textflag.h"

// Implementation of Poly1305 using the vector facility (vx).

// constants
#define MOD26 V0
#define EX0   V1
#define EX1   V2
#define EX2   V3

// temporaries
#define T_0 V4
#define T_1 V5
#define T_2 V6
#define T_3 V7
#define T_4 V8

// key (r)
#define R_0  V9
#define R_1  V10
#define R_2  V11
#define R_3  V12
#define R_4  V13
#define R5_1 V14
#define R5_2 V15
#define R5_3 V16
#define R5_4 V17
#define RSAVE_0 R5
#define RSAVE_1 R6
#define RSAVE_2 R7
#define RSAVE_3 R8
#define RSAVE_4 R9
#define R5SAVE_1 V28
#define R5SAVE_2 V29
#define R5SAVE_3 V30
#define R5SAVE_4 V31

// message block
#define F_0 V18
#define F_1 V19
#define F_2 V20
#define F_3 V21
#define F_4 V22

// accumulator
#define H_0 V23
#define H_1 V24
#define H_2 V25
#define H_3 V26
#define H_4 V27

GLOBL ·keyMask<>(SB), RODATA, $16
DATA ·keyMask<>+0(SB)/8, $0xffffff0ffcffff0f
DATA ·keyMask<>+8(SB)/8, $0xfcffff0ffcffff0f

GLOBL ·bswapMask<>(SB), RODATA, $16
DATA ·bswapMask<>+0(SB)/8, $0x0f0e0d0c0b0a0908
DATA ·bswapMask<>+8(SB)/8, $0x0706050403020100

GLOBL ·constants<>(SB), RODATA, $64
// MOD26
DATA ·constants<>+0(SB)/8, $0x3ffffff
DATA ·constants<>+8(SB)/8, $0x3ffffff
// EX0
DATA ·constants<>+16(SB)/8, $0x0006050403020100
DATA ·constants<>+24(SB)/8, $0x1016151413121110
// EX1
DATA ·constants<>+32(SB)/8, $0x060c0b0a09080706
DATA ·constants<>+40(SB)/8, $0x161c1b1a19181716
// EX2
DATA ·constants<>+48(SB)/8, $0x0d0d0d0d0d0f0e0d
DATA ·constants<>+56(SB)/8, $0x1d1d1d1d1d1f1e1d

// h = (f*g) % (2**130-5) [partial reduction]
#define MULTIPLY(f0, f1, f2, f3, f4, g0, g1, g2, g3, g4, g51, g52, g53, g54, h0, h1, h2, h3, h4) \
	VMLOF  f0, g0, h0        \
	VMLOF  f0, g1, h1        \
	VMLOF  f0, g2, h2        \
	VMLOF  f0, g3, h3        \
	VMLOF  f0, g4, h4        \
	VMLOF  f1, g54, T_0      \
	VMLOF  f1, g0, T_1       \
	VMLOF  f1, g1, T_2       \
	VMLOF  f1, g2, T_3       \
	VMLOF  f1, g3, T_4       \
	VMALOF f2, g53, h0, h0   \
	VMALOF f2, g54, h1, h1   \
	VMALOF f2, g0, h2, h2    \
	VMALOF f2, g1, h3, h3    \
	VMALOF f2, g2, h4, h4    \
	VMALOF f3, g52, T_0, T_0 \
	VMALOF f3, g53, T_1, T_1 \
	VMALOF f3, g54, T_2, T_2 \
	VMALOF f3, g0, T_3, T_3  \
	VMALOF f3, g1, T_4, T_4  \
	VMALOF f4, g51, h0, h0   \
	VMALOF f4, g52, h1, h1   \
	VMALOF f4, g53, h2, h2   \
	VMALOF f4, g54, h3, h3   \
	VMALOF f4, g0, h4, h4    \
	VAG    T_0, h0, h0       \
	VAG    T_1, h1, h1       \
	VAG    T_2, h2, h2       \
	VAG    T_3, h3, h3       \
	VAG    T_4, h4, h4

// carry h0->h1 h3->h4, h1->h2 h4->h0, h0->h1 h2->h3, h3->h4
#define REDUCE(h0, h1, h2, h3, h4) \
	VESRLG $26, h0, T_0  \
	VESRLG $26, h3, T_1  \
	VN     MOD26, h0, h0 \
	VN     MOD26, h3, h3 \
	VAG    T_0, h1, h1   \
	VAG    T_1, h4, h4   \
	VESRLG $26, h1, T_2  \
	VESRLG $26, h4, T_3  \
	VN     MOD26, h1, h1 \
	VN     MOD26, h4, h4 \
	VESLG  $2, T_3, T_4  \
	VAG    T_3, T_4, T_4 \
	VAG    T_2, h2, h2   \
	VAG    T_4, h0, h0   \
	VESRLG $26, h2, T_0  \
	VESRLG $26, h0, T_1  \
	VN     MOD26, h2, h2 \
	VN     MOD26, h0, h0 \
	VAG    T_0, h3, h3   \
	VAG    T_1, h1, h1   \
	VESRLG $26, h3, T_2  \
	VN     MOD26, h3, h3 \
	VAG    T_2, h4, h4

// expand in0 into d[0] and in1 into d[1]
#define EXPAND(in0, in1, d0, d1, d2, d3, d4) \
	VGBM   $0x0707, d1       \ // d1=tmp
	VPERM  in0, in1, EX2, d4 \
	VPERM  in0, in1, EX0, d0 \
	VPERM  in0, in1, EX1, d2 \
	VN     d1, d4, d4        \
	VESRLG $26, d0, d1       \
	VESRLG $30, d2, d3       \
	VESRLG $4, d2, d2        \
	VN     MOD26, d0, d0     \
	VN     MOD26, d1, d1     \
	VN     MOD26, d2, d2     \
	VN     MOD26, d3, d3

// pack h4:h0 into h1:h0 (no carry)
#define PACK(h0, h1, h2, h3, h4) \
	VESLG $26, h1, h1  \
	VESLG $26, h3, h3  \
	VO    h0, h1, h0   \
	VO    h2, h3, h2   \
	VESLG $4, h2, h2   \
	VLEIB $7, $48, h1  \
	VSLB  h1, h2, h2   \
	VO    h0, h2, h0   \
	VLEIB $7, $104, h1 \
	VSLB  h1, h4, h3   \
	VO    h3, h0, h0   \
	VLEIB $7, $24, h1  \
	VSRLB h1, h4, h1

// if h > 2**130-5 then h -= 2**130-5
#define MOD(h0, h1, t0, t1, t2) \
	VZERO t0          \
	VLEIG $1, $5, t0  \
	VACCQ h0, t0, t1  \
	VAQ   h0, t0, t0  \
	VONE  t2          \
	VLEIG $1, $-4, t2 \
	VAQ   t2, t1, t1  \
	VACCQ h1, t1, t1  \
	VONE  t2          \
	VAQ   t2, t1, t1  \
	VN    h0, t1, t2  \
	VNC   t0, t1, t1  \
	VO    t1, t2, h0

// func poly1305vx(out *[16]byte, m *byte, mlen uint64, key *[32]key)
TEXT ·poly1305vx(SB), $0-32
	// This code processes up to 2 blocks (32 bytes) per iteration
	// using the algorithm described in:
	// NEON crypto, Daniel J. Bernstein & Peter Schwabe
	// https://cryptojedi.org/papers/neoncrypto-20120320.pdf
	LMG out+0(FP), R1, R4 // R1=out, R2=m, R3=mlen, R4=key

	// load MOD26, EX0, EX1 and EX2
	MOVD $·constants<>(SB), R5
	VLM  (R5), MOD26, EX2

	// setup r
	VL   (R4), T_0
	MOVD $·keyMask<>(SB), R6
	VL   (R6), T_1
	VN   T_0, T_1, T_0
	EXPAND(T_0, T_0, R_0, R_1, R_2, R_3, R_4)

	// setup r*5
	VLEIG $0, $5, T_0
	VLEIG $1, $5, T_0

	// store r (for final block)
	VMLOF T_0, R_1, R5SAVE_1
	VMLOF T_0, R_2, R5SAVE_2
	VMLOF T_0, R_3, R5SAVE_3
	VMLOF T_0, R_4, R5SAVE_4
	VLGVG $0, R_0, RSAVE_0
	VLGVG $0, R_1, RSAVE_1
	VLGVG $0, R_2, RSAVE_2
	VLGVG $0, R_3, RSAVE_3
	VLGVG $0, R_4, RSAVE_4

	// skip r**2 calculation
	CMPBLE R3, $16, skip

	// calculate r**2
	MULTIPLY(R_0, R_1, R_2, R_3, R_4, R_0, R_1, R_2, R_3, R_4, R5SAVE_1, R5SAVE_2, R5SAVE_3, R5SAVE_4, H_0, H_1, H_2, H_3, H_4)
	REDUCE(H_0, H_1, H_2, H_3, H_4)
	VLEIG $0, $5, T_0
	VLEIG $1, $5, T_0
	VMLOF T_0, H_1, R5_1
	VMLOF T_0, H_2, R5_2
	VMLOF T_0, H_3, R5_3
	VMLOF T_0, H_4, R5_4
	VLR   H_0, R_0
	VLR   H_1, R_1
	VLR   H_2, R_2
	VLR   H_3, R_3
	VLR   H_4, R_4

	// initialize h
	VZERO H_0
	VZERO H_1
	VZERO H_2
	VZERO H_3
	VZERO H_4

loop:
	CMPBLE R3, $32, b2
	VLM    (R2), T_0, T_1
	SUB    $32, R3
	MOVD   $32(R2), R2
	EXPAND(T_0, T_1, F_0, F_1, F_2, F_3, F_4)
	VLEIB  $4, $1, F_4
	VLEIB  $12, $1, F_4

multiply:
	VAG    H_0, F_0, F_0
	VAG    H_1, F_1, F_1
	VAG    H_2, F_2, F_2
	VAG    H_3, F_3, F_3
	VAG    H_4, F_4, F_4
	MULTIPLY(F_0, F_1, F_2, F_3, F_4, R_0, R_1, R_2, R_3, R_4, R5_1, R5_2, R5_3, R5_4, H_0, H_1, H_2, H_3, H_4)
	REDUCE(H_0, H_1, H_2, H_3, H_4)
	CMPBNE R3, $0, loop

finish:
	// sum vectors
	VZERO  T_0
	VSUMQG H_0, T_0, H_0
	VSUMQG H_1, T_0, H_1
	VSUMQG H_2, T_0, H_2
	VSUMQG H_3, T_0, H_3
	VSUMQG H_4, T_0, H_4

	// h may be >= 2*(2**130-5) so we need to reduce it again
	REDUCE(H_0, H_1, H_2, H_3, H_4)

	// carry h1->h4
	VESRLG $26, H_1, T_1
	VN     MOD26, H_1, H_1
	VAQ    T_1, H_2, H_2
	VESRLG $26, H_2, T_2
	VN     MOD26, H_2, H_2
	VAQ    T_2, H_3, H_3
	VESRLG $26, H_3, T_3
	VN     MOD26, H_3, H_3
	VAQ    T_3, H_4, H_4

	// h is now < 2*(2**130-5)
	// pack h into h1 (hi) and h0 (lo)
	PACK(H_0, H_1, H_2, H_3, H_4)

	// if h > 2**130-5 then h -= 2**130-5
	MOD(H_0, H_1, T_0, T_1, T_2)

	// h += s
	MOVD  $·bswapMask<>(SB), R5
	VL    (R5), T_1
	VL    16(R4), T_0
	VPERM T_0, T_0, T_1, T_0    // reverse bytes (to big)
	VAQ   T_0, H_0, H_0
	VPERM H_0, H_0, T_1, H_0    // reverse bytes (to little)
	VST   H_0, (R1)

	RET

b2:
	CMPBLE R3, $16, b1

	// 2 blocks remaining
	SUB    $17, R3
	VL     (R2), T_0
	VLL    R3, 16(R2), T_1
	ADD    $1, R3
	MOVBZ  $1, R0
	CMPBEQ R3, $16, 2(PC)
	VLVGB  R3, R0, T_1
	EXPAND(T_0, T_1, F_0, F_1, F_2, F_3, F_4)
	CMPBNE R3, $16, 2(PC)
	VLEIB  $12, $1, F_4
	VLEIB  $4, $1, F_4

	// setup [r²,r]
	VLVGG $1, RSAVE_0, R_0
	VLVGG $1, RSAVE_1, R_1
	VLVGG $1, RSAVE_2, R_2
	VLVGG $1, RSAVE_3, R_3
	VLVGG $1, RSAVE_4, R_4
	VPDI  $0, R5_1, R5SAVE_1, R5_1
	VPDI  $0, R5_2, R5SAVE_2, R5_2
	VPDI  $0, R5_3, R5SAVE_3, R5_3
	VPDI  $0, R5_4, R5SAVE_4, R5_4

	MOVD $0, R3
	BR   multiply

skip:
	VZERO H_0
	VZERO H_1
	VZERO H_2
	VZERO H_3
	VZERO H_4

	CMPBEQ R3, $0, finish

b1:
	// 1 block remaining
	SUB    $1, R3
	VLL    R3, (R2), T_0
	ADD    $1, R3
	MOVBZ  $1, R0
	CMPBEQ R3, $16, 2(PC)
	VLVGB  R3, R0, T_0
	VZERO  T_1
	EXPAND(T_0, T_1, F_0, F_1, F_2, F_3, F_4)
	CMPBNE R3, $16, 2(PC)
	VLEIB  $4, $1, F_4
	VLEIG  $1, $1, R_0
	VZERO  R_1
	VZERO  R_2
	VZERO  R_3
	VZERO  R_4
	VZERO  R5_1
	VZERO  R5_2
	VZERO  R5_3
	VZERO  R5_4

	// setup [r, 1]
	VLVGG $0, RSAVE_0, R_0
	VLVGG $0, RSAVE_1, R_1
	VLVGG $0, RSAVE_2, R_2
	VLVGG $0, RSAVE_3, R_3
	VLVGG $0, RSAVE_4, R_4
	VPDI  $0, R5SAVE_1, R5_1, R5_1
	VPDI  $0, R5SAVE_2, R5_2, R5_2
	VPDI  $0, R5SAVE_3, R5_3, R5_3
	VPDI  $0, R5SAVE_4, R5_4, R5_4

	MOVD $0, R3
	BR   multiply
