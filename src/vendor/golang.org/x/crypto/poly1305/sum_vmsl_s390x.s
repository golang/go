// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build s390x,go1.11,!gccgo,!appengine

#include "textflag.h"

// Implementation of Poly1305 using the vector facility (vx) and the VMSL instruction.

// constants
#define EX0   V1
#define EX1   V2
#define EX2   V3

// temporaries
#define T_0 V4
#define T_1 V5
#define T_2 V6
#define T_3 V7
#define T_4 V8
#define T_5 V9
#define T_6 V10
#define T_7 V11
#define T_8 V12
#define T_9 V13
#define T_10 V14

// r**2 & r**4
#define R_0  V15
#define R_1  V16
#define R_2  V17
#define R5_1 V18
#define R5_2 V19
// key (r)
#define RSAVE_0 R7
#define RSAVE_1 R8
#define RSAVE_2 R9
#define R5SAVE_1 R10
#define R5SAVE_2 R11

// message block
#define M0 V20
#define M1 V21
#define M2 V22
#define M3 V23
#define M4 V24
#define M5 V25

// accumulator
#define H0_0 V26
#define H1_0 V27
#define H2_0 V28
#define H0_1 V29
#define H1_1 V30
#define H2_1 V31

GLOBL ·keyMask<>(SB), RODATA, $16
DATA ·keyMask<>+0(SB)/8, $0xffffff0ffcffff0f
DATA ·keyMask<>+8(SB)/8, $0xfcffff0ffcffff0f

GLOBL ·bswapMask<>(SB), RODATA, $16
DATA ·bswapMask<>+0(SB)/8, $0x0f0e0d0c0b0a0908
DATA ·bswapMask<>+8(SB)/8, $0x0706050403020100

GLOBL ·constants<>(SB), RODATA, $48
// EX0
DATA ·constants<>+0(SB)/8, $0x18191a1b1c1d1e1f
DATA ·constants<>+8(SB)/8, $0x0000050403020100
// EX1
DATA ·constants<>+16(SB)/8, $0x18191a1b1c1d1e1f
DATA ·constants<>+24(SB)/8, $0x00000a0908070605
// EX2
DATA ·constants<>+32(SB)/8, $0x18191a1b1c1d1e1f
DATA ·constants<>+40(SB)/8, $0x0000000f0e0d0c0b

GLOBL ·c<>(SB), RODATA, $48
// EX0
DATA ·c<>+0(SB)/8, $0x0000050403020100
DATA ·c<>+8(SB)/8, $0x0000151413121110
// EX1
DATA ·c<>+16(SB)/8, $0x00000a0908070605
DATA ·c<>+24(SB)/8, $0x00001a1918171615
// EX2
DATA ·c<>+32(SB)/8, $0x0000000f0e0d0c0b
DATA ·c<>+40(SB)/8, $0x0000001f1e1d1c1b

GLOBL ·reduce<>(SB), RODATA, $32
// 44 bit
DATA ·reduce<>+0(SB)/8, $0x0
DATA ·reduce<>+8(SB)/8, $0xfffffffffff
// 42 bit
DATA ·reduce<>+16(SB)/8, $0x0
DATA ·reduce<>+24(SB)/8, $0x3ffffffffff

// h = (f*g) % (2**130-5) [partial reduction]
// uses T_0...T_9 temporary registers
// input: m02_0, m02_1, m02_2, m13_0, m13_1, m13_2, r_0, r_1, r_2, r5_1, r5_2, m4_0, m4_1, m4_2, m5_0, m5_1, m5_2
// temp: t0, t1, t2, t3, t4, t5, t6, t7, t8, t9
// output: m02_0, m02_1, m02_2, m13_0, m13_1, m13_2
#define MULTIPLY(m02_0, m02_1, m02_2, m13_0, m13_1, m13_2, r_0, r_1, r_2, r5_1, r5_2, m4_0, m4_1, m4_2, m5_0, m5_1, m5_2, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9) \
	\ // Eliminate the dependency for the last 2 VMSLs
	VMSLG m02_0, r_2, m4_2, m4_2                       \
	VMSLG m13_0, r_2, m5_2, m5_2                       \ // 8 VMSLs pipelined
	VMSLG m02_0, r_0, m4_0, m4_0                       \
	VMSLG m02_1, r5_2, V0, T_0                         \
	VMSLG m02_0, r_1, m4_1, m4_1                       \
	VMSLG m02_1, r_0, V0, T_1                          \
	VMSLG m02_1, r_1, V0, T_2                          \
	VMSLG m02_2, r5_1, V0, T_3                         \
	VMSLG m02_2, r5_2, V0, T_4                         \
	VMSLG m13_0, r_0, m5_0, m5_0                       \
	VMSLG m13_1, r5_2, V0, T_5                         \
	VMSLG m13_0, r_1, m5_1, m5_1                       \
	VMSLG m13_1, r_0, V0, T_6                          \
	VMSLG m13_1, r_1, V0, T_7                          \
	VMSLG m13_2, r5_1, V0, T_8                         \
	VMSLG m13_2, r5_2, V0, T_9                         \
	VMSLG m02_2, r_0, m4_2, m4_2                       \
	VMSLG m13_2, r_0, m5_2, m5_2                       \
	VAQ   m4_0, T_0, m02_0                             \
	VAQ   m4_1, T_1, m02_1                             \
	VAQ   m5_0, T_5, m13_0                             \
	VAQ   m5_1, T_6, m13_1                             \
	VAQ   m02_0, T_3, m02_0                            \
	VAQ   m02_1, T_4, m02_1                            \
	VAQ   m13_0, T_8, m13_0                            \
	VAQ   m13_1, T_9, m13_1                            \
	VAQ   m4_2, T_2, m02_2                             \
	VAQ   m5_2, T_7, m13_2                             \

// SQUARE uses three limbs of r and r_2*5 to output square of r
// uses T_1, T_5 and T_7 temporary registers
// input: r_0, r_1, r_2, r5_2
// temp: TEMP0, TEMP1, TEMP2
// output: p0, p1, p2
#define SQUARE(r_0, r_1, r_2, r5_2, p0, p1, p2, TEMP0, TEMP1, TEMP2) \
	VMSLG r_0, r_0, p0, p0     \
	VMSLG r_1, r5_2, V0, TEMP0 \
	VMSLG r_2, r5_2, p1, p1    \
	VMSLG r_0, r_1, V0, TEMP1  \
	VMSLG r_1, r_1, p2, p2     \
	VMSLG r_0, r_2, V0, TEMP2  \
	VAQ   TEMP0, p0, p0        \
	VAQ   TEMP1, p1, p1        \
	VAQ   TEMP2, p2, p2        \
	VAQ   TEMP0, p0, p0        \
	VAQ   TEMP1, p1, p1        \
	VAQ   TEMP2, p2, p2        \

// carry h0->h1->h2->h0 || h3->h4->h5->h3
// uses T_2, T_4, T_5, T_7, T_8, T_9
//       t6,  t7,  t8,  t9, t10, t11
// input: h0, h1, h2, h3, h4, h5
// temp: t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11
// output: h0, h1, h2, h3, h4, h5
#define REDUCE(h0, h1, h2, h3, h4, h5, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11) \
	VLM    (R12), t6, t7  \ // 44 and 42 bit clear mask
	VLEIB  $7, $0x28, t10 \ // 5 byte shift mask
	VREPIB $4, t8         \ // 4 bit shift mask
	VREPIB $2, t11        \ // 2 bit shift mask
	VSRLB  t10, h0, t0    \ // h0 byte shift
	VSRLB  t10, h1, t1    \ // h1 byte shift
	VSRLB  t10, h2, t2    \ // h2 byte shift
	VSRLB  t10, h3, t3    \ // h3 byte shift
	VSRLB  t10, h4, t4    \ // h4 byte shift
	VSRLB  t10, h5, t5    \ // h5 byte shift
	VSRL   t8, t0, t0     \ // h0 bit shift
	VSRL   t8, t1, t1     \ // h2 bit shift
	VSRL   t11, t2, t2    \ // h2 bit shift
	VSRL   t8, t3, t3     \ // h3 bit shift
	VSRL   t8, t4, t4     \ // h4 bit shift
	VESLG  $2, t2, t9     \ // h2 carry x5
	VSRL   t11, t5, t5    \ // h5 bit shift
	VN     t6, h0, h0     \ // h0 clear carry
	VAQ    t2, t9, t2     \ // h2 carry x5
	VESLG  $2, t5, t9     \ // h5 carry x5
	VN     t6, h1, h1     \ // h1 clear carry
	VN     t7, h2, h2     \ // h2 clear carry
	VAQ    t5, t9, t5     \ // h5 carry x5
	VN     t6, h3, h3     \ // h3 clear carry
	VN     t6, h4, h4     \ // h4 clear carry
	VN     t7, h5, h5     \ // h5 clear carry
	VAQ    t0, h1, h1     \ // h0->h1
	VAQ    t3, h4, h4     \ // h3->h4
	VAQ    t1, h2, h2     \ // h1->h2
	VAQ    t4, h5, h5     \ // h4->h5
	VAQ    t2, h0, h0     \ // h2->h0
	VAQ    t5, h3, h3     \ // h5->h3
	VREPG  $1, t6, t6     \ // 44 and 42 bit masks across both halves
	VREPG  $1, t7, t7     \
	VSLDB  $8, h0, h0, h0 \ // set up [h0/1/2, h3/4/5]
	VSLDB  $8, h1, h1, h1 \
	VSLDB  $8, h2, h2, h2 \
	VO     h0, h3, h3     \
	VO     h1, h4, h4     \
	VO     h2, h5, h5     \
	VESRLG $44, h3, t0    \ // 44 bit shift right
	VESRLG $44, h4, t1    \
	VESRLG $42, h5, t2    \
	VN     t6, h3, h3     \ // clear carry bits
	VN     t6, h4, h4     \
	VN     t7, h5, h5     \
	VESLG  $2, t2, t9     \ // multiply carry by 5
	VAQ    t9, t2, t2     \
	VAQ    t0, h4, h4     \
	VAQ    t1, h5, h5     \
	VAQ    t2, h3, h3     \

// carry h0->h1->h2->h0
// input: h0, h1, h2
// temp: t0, t1, t2, t3, t4, t5, t6, t7, t8
// output: h0, h1, h2
#define REDUCE2(h0, h1, h2, t0, t1, t2, t3, t4, t5, t6, t7, t8) \
	VLEIB  $7, $0x28, t3 \ // 5 byte shift mask
	VREPIB $4, t4        \ // 4 bit shift mask
	VREPIB $2, t7        \ // 2 bit shift mask
	VGBM   $0x003F, t5   \ // mask to clear carry bits
	VSRLB  t3, h0, t0    \
	VSRLB  t3, h1, t1    \
	VSRLB  t3, h2, t2    \
	VESRLG $4, t5, t5    \ // 44 bit clear mask
	VSRL   t4, t0, t0    \
	VSRL   t4, t1, t1    \
	VSRL   t7, t2, t2    \
	VESRLG $2, t5, t6    \ // 42 bit clear mask
	VESLG  $2, t2, t8    \
	VAQ    t8, t2, t2    \
	VN     t5, h0, h0    \
	VN     t5, h1, h1    \
	VN     t6, h2, h2    \
	VAQ    t0, h1, h1    \
	VAQ    t1, h2, h2    \
	VAQ    t2, h0, h0    \
	VSRLB  t3, h0, t0    \
	VSRLB  t3, h1, t1    \
	VSRLB  t3, h2, t2    \
	VSRL   t4, t0, t0    \
	VSRL   t4, t1, t1    \
	VSRL   t7, t2, t2    \
	VN     t5, h0, h0    \
	VN     t5, h1, h1    \
	VESLG  $2, t2, t8    \
	VN     t6, h2, h2    \
	VAQ    t0, h1, h1    \
	VAQ    t8, t2, t2    \
	VAQ    t1, h2, h2    \
	VAQ    t2, h0, h0    \

// expands two message blocks into the lower halfs of the d registers
// moves the contents of the d registers into upper halfs
// input: in1, in2, d0, d1, d2, d3, d4, d5
// temp: TEMP0, TEMP1, TEMP2, TEMP3
// output: d0, d1, d2, d3, d4, d5
#define EXPACC(in1, in2, d0, d1, d2, d3, d4, d5, TEMP0, TEMP1, TEMP2, TEMP3) \
	VGBM   $0xff3f, TEMP0      \
	VGBM   $0xff1f, TEMP1      \
	VESLG  $4, d1, TEMP2       \
	VESLG  $4, d4, TEMP3       \
	VESRLG $4, TEMP0, TEMP0    \
	VPERM  in1, d0, EX0, d0    \
	VPERM  in2, d3, EX0, d3    \
	VPERM  in1, d2, EX2, d2    \
	VPERM  in2, d5, EX2, d5    \
	VPERM  in1, TEMP2, EX1, d1 \
	VPERM  in2, TEMP3, EX1, d4 \
	VN     TEMP0, d0, d0       \
	VN     TEMP0, d3, d3       \
	VESRLG $4, d1, d1          \
	VESRLG $4, d4, d4          \
	VN     TEMP1, d2, d2       \
	VN     TEMP1, d5, d5       \
	VN     TEMP0, d1, d1       \
	VN     TEMP0, d4, d4       \

// expands one message block into the lower halfs of the d registers
// moves the contents of the d registers into upper halfs
// input: in, d0, d1, d2
// temp: TEMP0, TEMP1, TEMP2
// output: d0, d1, d2
#define EXPACC2(in, d0, d1, d2, TEMP0, TEMP1, TEMP2) \
	VGBM   $0xff3f, TEMP0     \
	VESLG  $4, d1, TEMP2      \
	VGBM   $0xff1f, TEMP1     \
	VPERM  in, d0, EX0, d0    \
	VESRLG $4, TEMP0, TEMP0   \
	VPERM  in, d2, EX2, d2    \
	VPERM  in, TEMP2, EX1, d1 \
	VN     TEMP0, d0, d0      \
	VN     TEMP1, d2, d2      \
	VESRLG $4, d1, d1         \
	VN     TEMP0, d1, d1      \

// pack h2:h0 into h1:h0 (no carry)
// input: h0, h1, h2
// output: h0, h1, h2
#define PACK(h0, h1, h2) \
	VMRLG  h1, h2, h2  \ // copy h1 to upper half h2
	VESLG  $44, h1, h1 \ // shift limb 1 44 bits, leaving 20
	VO     h0, h1, h0  \ // combine h0 with 20 bits from limb 1
	VESRLG $20, h2, h1 \ // put top 24 bits of limb 1 into h1
	VLEIG  $1, $0, h1  \ // clear h2 stuff from lower half of h1
	VO     h0, h1, h0  \ // h0 now has 88 bits (limb 0 and 1)
	VLEIG  $0, $0, h2  \ // clear upper half of h2
	VESRLG $40, h2, h1 \ // h1 now has upper two bits of result
	VLEIB  $7, $88, h1 \ // for byte shift (11 bytes)
	VSLB   h1, h2, h2  \ // shift h2 11 bytes to the left
	VO     h0, h2, h0  \ // combine h0 with 20 bits from limb 1
	VLEIG  $0, $0, h1  \ // clear upper half of h1

// if h > 2**130-5 then h -= 2**130-5
// input: h0, h1
// temp: t0, t1, t2
// output: h0
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
	VO    t1, t2, h0  \

// func poly1305vmsl(out *[16]byte, m *byte, mlen uint64, key *[32]key)
TEXT ·poly1305vmsl(SB), $0-32
	// This code processes 6 + up to 4 blocks (32 bytes) per iteration
	// using the algorithm described in:
	// NEON crypto, Daniel J. Bernstein & Peter Schwabe
	// https://cryptojedi.org/papers/neoncrypto-20120320.pdf
	// And as moddified for VMSL as described in
	// Accelerating Poly1305 Cryptographic Message Authentication on the z14
	// O'Farrell et al, CASCON 2017, p48-55
	// https://ibm.ent.box.com/s/jf9gedj0e9d2vjctfyh186shaztavnht

	LMG   out+0(FP), R1, R4 // R1=out, R2=m, R3=mlen, R4=key
	VZERO V0                // c

	// load EX0, EX1 and EX2
	MOVD $·constants<>(SB), R5
	VLM  (R5), EX0, EX2        // c

	// setup r
	VL    (R4), T_0
	MOVD  $·keyMask<>(SB), R6
	VL    (R6), T_1
	VN    T_0, T_1, T_0
	VZERO T_2                 // limbs for r
	VZERO T_3
	VZERO T_4
	EXPACC2(T_0, T_2, T_3, T_4, T_1, T_5, T_7)

	// T_2, T_3, T_4: [0, r]

	// setup r*20
	VLEIG $0, $0, T_0
	VLEIG $1, $20, T_0       // T_0: [0, 20]
	VZERO T_5
	VZERO T_6
	VMSLG T_0, T_3, T_5, T_5
	VMSLG T_0, T_4, T_6, T_6

	// store r for final block in GR
	VLGVG $1, T_2, RSAVE_0  // c
	VLGVG $1, T_3, RSAVE_1  // c
	VLGVG $1, T_4, RSAVE_2  // c
	VLGVG $1, T_5, R5SAVE_1 // c
	VLGVG $1, T_6, R5SAVE_2 // c

	// initialize h
	VZERO H0_0
	VZERO H1_0
	VZERO H2_0
	VZERO H0_1
	VZERO H1_1
	VZERO H2_1

	// initialize pointer for reduce constants
	MOVD $·reduce<>(SB), R12

	// calculate r**2 and 20*(r**2)
	VZERO R_0
	VZERO R_1
	VZERO R_2
	SQUARE(T_2, T_3, T_4, T_6, R_0, R_1, R_2, T_1, T_5, T_7)
	REDUCE2(R_0, R_1, R_2, M0, M1, M2, M3, M4, R5_1, R5_2, M5, T_1)
	VZERO R5_1
	VZERO R5_2
	VMSLG T_0, R_1, R5_1, R5_1
	VMSLG T_0, R_2, R5_2, R5_2

	// skip r**4 calculation if 3 blocks or less
	CMPBLE R3, $48, b4

	// calculate r**4 and 20*(r**4)
	VZERO T_8
	VZERO T_9
	VZERO T_10
	SQUARE(R_0, R_1, R_2, R5_2, T_8, T_9, T_10, T_1, T_5, T_7)
	REDUCE2(T_8, T_9, T_10, M0, M1, M2, M3, M4, T_2, T_3, M5, T_1)
	VZERO T_2
	VZERO T_3
	VMSLG T_0, T_9, T_2, T_2
	VMSLG T_0, T_10, T_3, T_3

	// put r**2 to the right and r**4 to the left of R_0, R_1, R_2
	VSLDB $8, T_8, T_8, T_8
	VSLDB $8, T_9, T_9, T_9
	VSLDB $8, T_10, T_10, T_10
	VSLDB $8, T_2, T_2, T_2
	VSLDB $8, T_3, T_3, T_3

	VO T_8, R_0, R_0
	VO T_9, R_1, R_1
	VO T_10, R_2, R_2
	VO T_2, R5_1, R5_1
	VO T_3, R5_2, R5_2

	CMPBLE R3, $80, load // less than or equal to 5 blocks in message

	// 6(or 5+1) blocks
	SUB    $81, R3
	VLM    (R2), M0, M4
	VLL    R3, 80(R2), M5
	ADD    $1, R3
	MOVBZ  $1, R0
	CMPBGE R3, $16, 2(PC)
	VLVGB  R3, R0, M5
	MOVD   $96(R2), R2
	EXPACC(M0, M1, H0_0, H1_0, H2_0, H0_1, H1_1, H2_1, T_0, T_1, T_2, T_3)
	EXPACC(M2, M3, H0_0, H1_0, H2_0, H0_1, H1_1, H2_1, T_0, T_1, T_2, T_3)
	VLEIB  $2, $1, H2_0
	VLEIB  $2, $1, H2_1
	VLEIB  $10, $1, H2_0
	VLEIB  $10, $1, H2_1

	VZERO  M0
	VZERO  M1
	VZERO  M2
	VZERO  M3
	VZERO  T_4
	VZERO  T_10
	EXPACC(M4, M5, M0, M1, M2, M3, T_4, T_10, T_0, T_1, T_2, T_3)
	VLR    T_4, M4
	VLEIB  $10, $1, M2
	CMPBLT R3, $16, 2(PC)
	VLEIB  $10, $1, T_10
	MULTIPLY(H0_0, H1_0, H2_0, H0_1, H1_1, H2_1, R_0, R_1, R_2, R5_1, R5_2, M0, M1, M2, M3, M4, T_10, T_0, T_1, T_2, T_3, T_4, T_5, T_6, T_7, T_8, T_9)
	REDUCE(H0_0, H1_0, H2_0, H0_1, H1_1, H2_1, T_10, M0, M1, M2, M3, M4, T_4, T_5, T_2, T_7, T_8, T_9)
	VMRHG  V0, H0_1, H0_0
	VMRHG  V0, H1_1, H1_0
	VMRHG  V0, H2_1, H2_0
	VMRLG  V0, H0_1, H0_1
	VMRLG  V0, H1_1, H1_1
	VMRLG  V0, H2_1, H2_1

	SUB    $16, R3
	CMPBLE R3, $0, square

load:
	// load EX0, EX1 and EX2
	MOVD $·c<>(SB), R5
	VLM  (R5), EX0, EX2

loop:
	CMPBLE R3, $64, add // b4	// last 4 or less blocks left

	// next 4 full blocks
	VLM  (R2), M2, M5
	SUB  $64, R3
	MOVD $64(R2), R2
	REDUCE(H0_0, H1_0, H2_0, H0_1, H1_1, H2_1, T_10, M0, M1, T_0, T_1, T_3, T_4, T_5, T_2, T_7, T_8, T_9)

	// expacc in-lined to create [m2, m3] limbs
	VGBM   $0x3f3f, T_0     // 44 bit clear mask
	VGBM   $0x1f1f, T_1     // 40 bit clear mask
	VPERM  M2, M3, EX0, T_3
	VESRLG $4, T_0, T_0     // 44 bit clear mask ready
	VPERM  M2, M3, EX1, T_4
	VPERM  M2, M3, EX2, T_5
	VN     T_0, T_3, T_3
	VESRLG $4, T_4, T_4
	VN     T_1, T_5, T_5
	VN     T_0, T_4, T_4
	VMRHG  H0_1, T_3, H0_0
	VMRHG  H1_1, T_4, H1_0
	VMRHG  H2_1, T_5, H2_0
	VMRLG  H0_1, T_3, H0_1
	VMRLG  H1_1, T_4, H1_1
	VMRLG  H2_1, T_5, H2_1
	VLEIB  $10, $1, H2_0
	VLEIB  $10, $1, H2_1
	VPERM  M4, M5, EX0, T_3
	VPERM  M4, M5, EX1, T_4
	VPERM  M4, M5, EX2, T_5
	VN     T_0, T_3, T_3
	VESRLG $4, T_4, T_4
	VN     T_1, T_5, T_5
	VN     T_0, T_4, T_4
	VMRHG  V0, T_3, M0
	VMRHG  V0, T_4, M1
	VMRHG  V0, T_5, M2
	VMRLG  V0, T_3, M3
	VMRLG  V0, T_4, M4
	VMRLG  V0, T_5, M5
	VLEIB  $10, $1, M2
	VLEIB  $10, $1, M5

	MULTIPLY(H0_0, H1_0, H2_0, H0_1, H1_1, H2_1, R_0, R_1, R_2, R5_1, R5_2, M0, M1, M2, M3, M4, M5, T_0, T_1, T_2, T_3, T_4, T_5, T_6, T_7, T_8, T_9)
	CMPBNE R3, $0, loop
	REDUCE(H0_0, H1_0, H2_0, H0_1, H1_1, H2_1, T_10, M0, M1, M3, M4, M5, T_4, T_5, T_2, T_7, T_8, T_9)
	VMRHG  V0, H0_1, H0_0
	VMRHG  V0, H1_1, H1_0
	VMRHG  V0, H2_1, H2_0
	VMRLG  V0, H0_1, H0_1
	VMRLG  V0, H1_1, H1_1
	VMRLG  V0, H2_1, H2_1

	// load EX0, EX1, EX2
	MOVD $·constants<>(SB), R5
	VLM  (R5), EX0, EX2

	// sum vectors
	VAQ H0_0, H0_1, H0_0
	VAQ H1_0, H1_1, H1_0
	VAQ H2_0, H2_1, H2_0

	// h may be >= 2*(2**130-5) so we need to reduce it again
	// M0...M4 are used as temps here
	REDUCE2(H0_0, H1_0, H2_0, M0, M1, M2, M3, M4, T_9, T_10, H0_1, M5)

next:  // carry h1->h2
	VLEIB  $7, $0x28, T_1
	VREPIB $4, T_2
	VGBM   $0x003F, T_3
	VESRLG $4, T_3

	// byte shift
	VSRLB T_1, H1_0, T_4

	// bit shift
	VSRL T_2, T_4, T_4

	// clear h1 carry bits
	VN T_3, H1_0, H1_0

	// add carry
	VAQ T_4, H2_0, H2_0

	// h is now < 2*(2**130-5)
	// pack h into h1 (hi) and h0 (lo)
	PACK(H0_0, H1_0, H2_0)

	// if h > 2**130-5 then h -= 2**130-5
	MOD(H0_0, H1_0, T_0, T_1, T_2)

	// h += s
	MOVD  $·bswapMask<>(SB), R5
	VL    (R5), T_1
	VL    16(R4), T_0
	VPERM T_0, T_0, T_1, T_0    // reverse bytes (to big)
	VAQ   T_0, H0_0, H0_0
	VPERM H0_0, H0_0, T_1, H0_0 // reverse bytes (to little)
	VST   H0_0, (R1)
	RET

add:
	// load EX0, EX1, EX2
	MOVD $·constants<>(SB), R5
	VLM  (R5), EX0, EX2

	REDUCE(H0_0, H1_0, H2_0, H0_1, H1_1, H2_1, T_10, M0, M1, M3, M4, M5, T_4, T_5, T_2, T_7, T_8, T_9)
	VMRHG  V0, H0_1, H0_0
	VMRHG  V0, H1_1, H1_0
	VMRHG  V0, H2_1, H2_0
	VMRLG  V0, H0_1, H0_1
	VMRLG  V0, H1_1, H1_1
	VMRLG  V0, H2_1, H2_1
	CMPBLE R3, $64, b4

b4:
	CMPBLE R3, $48, b3 // 3 blocks or less

	// 4(3+1) blocks remaining
	SUB    $49, R3
	VLM    (R2), M0, M2
	VLL    R3, 48(R2), M3
	ADD    $1, R3
	MOVBZ  $1, R0
	CMPBEQ R3, $16, 2(PC)
	VLVGB  R3, R0, M3
	MOVD   $64(R2), R2
	EXPACC(M0, M1, H0_0, H1_0, H2_0, H0_1, H1_1, H2_1, T_0, T_1, T_2, T_3)
	VLEIB  $10, $1, H2_0
	VLEIB  $10, $1, H2_1
	VZERO  M0
	VZERO  M1
	VZERO  M4
	VZERO  M5
	VZERO  T_4
	VZERO  T_10
	EXPACC(M2, M3, M0, M1, M4, M5, T_4, T_10, T_0, T_1, T_2, T_3)
	VLR    T_4, M2
	VLEIB  $10, $1, M4
	CMPBNE R3, $16, 2(PC)
	VLEIB  $10, $1, T_10
	MULTIPLY(H0_0, H1_0, H2_0, H0_1, H1_1, H2_1, R_0, R_1, R_2, R5_1, R5_2, M0, M1, M4, M5, M2, T_10, T_0, T_1, T_2, T_3, T_4, T_5, T_6, T_7, T_8, T_9)
	REDUCE(H0_0, H1_0, H2_0, H0_1, H1_1, H2_1, T_10, M0, M1, M3, M4, M5, T_4, T_5, T_2, T_7, T_8, T_9)
	VMRHG  V0, H0_1, H0_0
	VMRHG  V0, H1_1, H1_0
	VMRHG  V0, H2_1, H2_0
	VMRLG  V0, H0_1, H0_1
	VMRLG  V0, H1_1, H1_1
	VMRLG  V0, H2_1, H2_1
	SUB    $16, R3
	CMPBLE R3, $0, square // this condition must always hold true!

b3:
	CMPBLE R3, $32, b2

	// 3 blocks remaining

	// setup [r²,r]
	VSLDB $8, R_0, R_0, R_0
	VSLDB $8, R_1, R_1, R_1
	VSLDB $8, R_2, R_2, R_2
	VSLDB $8, R5_1, R5_1, R5_1
	VSLDB $8, R5_2, R5_2, R5_2

	VLVGG $1, RSAVE_0, R_0
	VLVGG $1, RSAVE_1, R_1
	VLVGG $1, RSAVE_2, R_2
	VLVGG $1, R5SAVE_1, R5_1
	VLVGG $1, R5SAVE_2, R5_2

	// setup [h0, h1]
	VSLDB $8, H0_0, H0_0, H0_0
	VSLDB $8, H1_0, H1_0, H1_0
	VSLDB $8, H2_0, H2_0, H2_0
	VO    H0_1, H0_0, H0_0
	VO    H1_1, H1_0, H1_0
	VO    H2_1, H2_0, H2_0
	VZERO H0_1
	VZERO H1_1
	VZERO H2_1

	VZERO M0
	VZERO M1
	VZERO M2
	VZERO M3
	VZERO M4
	VZERO M5

	// H*[r**2, r]
	MULTIPLY(H0_0, H1_0, H2_0, H0_1, H1_1, H2_1, R_0, R_1, R_2, R5_1, R5_2, M0, M1, M2, M3, M4, M5, T_0, T_1, T_2, T_3, T_4, T_5, T_6, T_7, T_8, T_9)
	REDUCE2(H0_0, H1_0, H2_0, M0, M1, M2, M3, M4, H0_1, H1_1, T_10, M5)

	SUB    $33, R3
	VLM    (R2), M0, M1
	VLL    R3, 32(R2), M2
	ADD    $1, R3
	MOVBZ  $1, R0
	CMPBEQ R3, $16, 2(PC)
	VLVGB  R3, R0, M2

	// H += m0
	VZERO T_1
	VZERO T_2
	VZERO T_3
	EXPACC2(M0, T_1, T_2, T_3, T_4, T_5, T_6)
	VLEIB $10, $1, T_3
	VAG   H0_0, T_1, H0_0
	VAG   H1_0, T_2, H1_0
	VAG   H2_0, T_3, H2_0

	VZERO M0
	VZERO M3
	VZERO M4
	VZERO M5
	VZERO T_10

	// (H+m0)*r
	MULTIPLY(H0_0, H1_0, H2_0, H0_1, H1_1, H2_1, R_0, R_1, R_2, R5_1, R5_2, M0, M3, M4, M5, V0, T_10, T_0, T_1, T_2, T_3, T_4, T_5, T_6, T_7, T_8, T_9)
	REDUCE2(H0_0, H1_0, H2_0, M0, M3, M4, M5, T_10, H0_1, H1_1, H2_1, T_9)

	// H += m1
	VZERO V0
	VZERO T_1
	VZERO T_2
	VZERO T_3
	EXPACC2(M1, T_1, T_2, T_3, T_4, T_5, T_6)
	VLEIB $10, $1, T_3
	VAQ   H0_0, T_1, H0_0
	VAQ   H1_0, T_2, H1_0
	VAQ   H2_0, T_3, H2_0
	REDUCE2(H0_0, H1_0, H2_0, M0, M3, M4, M5, T_9, H0_1, H1_1, H2_1, T_10)

	// [H, m2] * [r**2, r]
	EXPACC2(M2, H0_0, H1_0, H2_0, T_1, T_2, T_3)
	CMPBNE R3, $16, 2(PC)
	VLEIB  $10, $1, H2_0
	VZERO  M0
	VZERO  M1
	VZERO  M2
	VZERO  M3
	VZERO  M4
	VZERO  M5
	MULTIPLY(H0_0, H1_0, H2_0, H0_1, H1_1, H2_1, R_0, R_1, R_2, R5_1, R5_2, M0, M1, M2, M3, M4, M5, T_0, T_1, T_2, T_3, T_4, T_5, T_6, T_7, T_8, T_9)
	REDUCE2(H0_0, H1_0, H2_0, M0, M1, M2, M3, M4, H0_1, H1_1, M5, T_10)
	SUB    $16, R3
	CMPBLE R3, $0, next   // this condition must always hold true!

b2:
	CMPBLE R3, $16, b1

	// 2 blocks remaining

	// setup [r²,r]
	VSLDB $8, R_0, R_0, R_0
	VSLDB $8, R_1, R_1, R_1
	VSLDB $8, R_2, R_2, R_2
	VSLDB $8, R5_1, R5_1, R5_1
	VSLDB $8, R5_2, R5_2, R5_2

	VLVGG $1, RSAVE_0, R_0
	VLVGG $1, RSAVE_1, R_1
	VLVGG $1, RSAVE_2, R_2
	VLVGG $1, R5SAVE_1, R5_1
	VLVGG $1, R5SAVE_2, R5_2

	// setup [h0, h1]
	VSLDB $8, H0_0, H0_0, H0_0
	VSLDB $8, H1_0, H1_0, H1_0
	VSLDB $8, H2_0, H2_0, H2_0
	VO    H0_1, H0_0, H0_0
	VO    H1_1, H1_0, H1_0
	VO    H2_1, H2_0, H2_0
	VZERO H0_1
	VZERO H1_1
	VZERO H2_1

	VZERO M0
	VZERO M1
	VZERO M2
	VZERO M3
	VZERO M4
	VZERO M5

	// H*[r**2, r]
	MULTIPLY(H0_0, H1_0, H2_0, H0_1, H1_1, H2_1, R_0, R_1, R_2, R5_1, R5_2, M0, M1, M2, M3, M4, M5, T_0, T_1, T_2, T_3, T_4, T_5, T_6, T_7, T_8, T_9)
	REDUCE(H0_0, H1_0, H2_0, H0_1, H1_1, H2_1, T_10, M0, M1, M2, M3, M4, T_4, T_5, T_2, T_7, T_8, T_9)
	VMRHG V0, H0_1, H0_0
	VMRHG V0, H1_1, H1_0
	VMRHG V0, H2_1, H2_0
	VMRLG V0, H0_1, H0_1
	VMRLG V0, H1_1, H1_1
	VMRLG V0, H2_1, H2_1

	// move h to the left and 0s at the right
	VSLDB $8, H0_0, H0_0, H0_0
	VSLDB $8, H1_0, H1_0, H1_0
	VSLDB $8, H2_0, H2_0, H2_0

	// get message blocks and append 1 to start
	SUB    $17, R3
	VL     (R2), M0
	VLL    R3, 16(R2), M1
	ADD    $1, R3
	MOVBZ  $1, R0
	CMPBEQ R3, $16, 2(PC)
	VLVGB  R3, R0, M1
	VZERO  T_6
	VZERO  T_7
	VZERO  T_8
	EXPACC2(M0, T_6, T_7, T_8, T_1, T_2, T_3)
	EXPACC2(M1, T_6, T_7, T_8, T_1, T_2, T_3)
	VLEIB  $2, $1, T_8
	CMPBNE R3, $16, 2(PC)
	VLEIB  $10, $1, T_8

	// add [m0, m1] to h
	VAG H0_0, T_6, H0_0
	VAG H1_0, T_7, H1_0
	VAG H2_0, T_8, H2_0

	VZERO M2
	VZERO M3
	VZERO M4
	VZERO M5
	VZERO T_10
	VZERO M0

	// at this point R_0 .. R5_2 look like [r**2, r]
	MULTIPLY(H0_0, H1_0, H2_0, H0_1, H1_1, H2_1, R_0, R_1, R_2, R5_1, R5_2, M2, M3, M4, M5, T_10, M0, T_0, T_1, T_2, T_3, T_4, T_5, T_6, T_7, T_8, T_9)
	REDUCE2(H0_0, H1_0, H2_0, M2, M3, M4, M5, T_9, H0_1, H1_1, H2_1, T_10)
	SUB    $16, R3, R3
	CMPBLE R3, $0, next

b1:
	CMPBLE R3, $0, next

	// 1 block remaining

	// setup [r²,r]
	VSLDB $8, R_0, R_0, R_0
	VSLDB $8, R_1, R_1, R_1
	VSLDB $8, R_2, R_2, R_2
	VSLDB $8, R5_1, R5_1, R5_1
	VSLDB $8, R5_2, R5_2, R5_2

	VLVGG $1, RSAVE_0, R_0
	VLVGG $1, RSAVE_1, R_1
	VLVGG $1, RSAVE_2, R_2
	VLVGG $1, R5SAVE_1, R5_1
	VLVGG $1, R5SAVE_2, R5_2

	// setup [h0, h1]
	VSLDB $8, H0_0, H0_0, H0_0
	VSLDB $8, H1_0, H1_0, H1_0
	VSLDB $8, H2_0, H2_0, H2_0
	VO    H0_1, H0_0, H0_0
	VO    H1_1, H1_0, H1_0
	VO    H2_1, H2_0, H2_0
	VZERO H0_1
	VZERO H1_1
	VZERO H2_1

	VZERO M0
	VZERO M1
	VZERO M2
	VZERO M3
	VZERO M4
	VZERO M5

	// H*[r**2, r]
	MULTIPLY(H0_0, H1_0, H2_0, H0_1, H1_1, H2_1, R_0, R_1, R_2, R5_1, R5_2, M0, M1, M2, M3, M4, M5, T_0, T_1, T_2, T_3, T_4, T_5, T_6, T_7, T_8, T_9)
	REDUCE2(H0_0, H1_0, H2_0, M0, M1, M2, M3, M4, T_9, T_10, H0_1, M5)

	// set up [0, m0] limbs
	SUB    $1, R3
	VLL    R3, (R2), M0
	ADD    $1, R3
	MOVBZ  $1, R0
	CMPBEQ R3, $16, 2(PC)
	VLVGB  R3, R0, M0
	VZERO  T_1
	VZERO  T_2
	VZERO  T_3
	EXPACC2(M0, T_1, T_2, T_3, T_4, T_5, T_6)// limbs: [0, m]
	CMPBNE R3, $16, 2(PC)
	VLEIB  $10, $1, T_3

	// h+m0
	VAQ H0_0, T_1, H0_0
	VAQ H1_0, T_2, H1_0
	VAQ H2_0, T_3, H2_0

	VZERO M0
	VZERO M1
	VZERO M2
	VZERO M3
	VZERO M4
	VZERO M5
	MULTIPLY(H0_0, H1_0, H2_0, H0_1, H1_1, H2_1, R_0, R_1, R_2, R5_1, R5_2, M0, M1, M2, M3, M4, M5, T_0, T_1, T_2, T_3, T_4, T_5, T_6, T_7, T_8, T_9)
	REDUCE2(H0_0, H1_0, H2_0, M0, M1, M2, M3, M4, T_9, T_10, H0_1, M5)

	BR next

square:
	// setup [r²,r]
	VSLDB $8, R_0, R_0, R_0
	VSLDB $8, R_1, R_1, R_1
	VSLDB $8, R_2, R_2, R_2
	VSLDB $8, R5_1, R5_1, R5_1
	VSLDB $8, R5_2, R5_2, R5_2

	VLVGG $1, RSAVE_0, R_0
	VLVGG $1, RSAVE_1, R_1
	VLVGG $1, RSAVE_2, R_2
	VLVGG $1, R5SAVE_1, R5_1
	VLVGG $1, R5SAVE_2, R5_2

	// setup [h0, h1]
	VSLDB $8, H0_0, H0_0, H0_0
	VSLDB $8, H1_0, H1_0, H1_0
	VSLDB $8, H2_0, H2_0, H2_0
	VO    H0_1, H0_0, H0_0
	VO    H1_1, H1_0, H1_0
	VO    H2_1, H2_0, H2_0
	VZERO H0_1
	VZERO H1_1
	VZERO H2_1

	VZERO M0
	VZERO M1
	VZERO M2
	VZERO M3
	VZERO M4
	VZERO M5

	// (h0*r**2) + (h1*r)
	MULTIPLY(H0_0, H1_0, H2_0, H0_1, H1_1, H2_1, R_0, R_1, R_2, R5_1, R5_2, M0, M1, M2, M3, M4, M5, T_0, T_1, T_2, T_3, T_4, T_5, T_6, T_7, T_8, T_9)
	REDUCE2(H0_0, H1_0, H2_0, M0, M1, M2, M3, M4, T_9, T_10, H0_1, M5)
	BR next

TEXT ·hasVMSLFacility(SB), NOSPLIT, $24-1
	MOVD  $x-24(SP), R1
	XC    $24, 0(R1), 0(R1) // clear the storage
	MOVD  $2, R0            // R0 is the number of double words stored -1
	WORD  $0xB2B01000       // STFLE 0(R1)
	XOR   R0, R0            // reset the value of R0
	MOVBZ z-8(SP), R1
	AND   $0x01, R1
	BEQ   novmsl

vectorinstalled:
	// check if the vector instruction has been enabled
	VLEIB  $0, $0xF, V16
	VLGVB  $0, V16, R1
	CMPBNE R1, $0xF, novmsl
	MOVB   $1, ret+0(FP)    // have vx
	RET

novmsl:
	MOVB $0, ret+0(FP) // no vx
	RET
