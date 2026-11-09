// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

// SHA-3 (Keccak-f[1600]) for LoongArch LSX, two lanes per register.
// Column x is stored as:
//   V(0+x)  = [L(x,0), L(x,1)]
//   V(5+x)  = [L(x,2), L(x,3)]
//   V(10+x) = [L(x,4), x]      (element 1 undefined; masked by V31 in theta)
// V15-V19 parity/pi-A, V20-V24 D/pi-B, V25-V29 pi-C, V30 tmp, V31 mask.

#include "textflag.h"

TEXT ·keccakF1600LSX(SB), NOSPLIT, $200-8
	MOVV	a+0(FP), R4
	MOVV	$round_consts<>(SB), R6
	MOVV	$rho_pairs<>(SB), R8
	MOVV	$24, R7

	// V31 = [-1, 0] mask (element 0 set, element 1 clear).
	VXORV	V31, V31, V31
	MOVV	$-1, R5
	VMOVQ	R5, V31.V[0]

	// Load state: A_x = lanes (x, x+5), B_x = (x+10, x+15), C_x = (x+20, 0).
	MOVV	0(R4), R5
	VMOVQ	R5, V0.V[0]
	MOVV	40(R4), R5
	VMOVQ	R5, V0.V[1]
	MOVV	8(R4), R5
	VMOVQ	R5, V1.V[0]
	MOVV	48(R4), R5
	VMOVQ	R5, V1.V[1]
	MOVV	16(R4), R5
	VMOVQ	R5, V2.V[0]
	MOVV	56(R4), R5
	VMOVQ	R5, V2.V[1]
	MOVV	24(R4), R5
	VMOVQ	R5, V3.V[0]
	MOVV	64(R4), R5
	VMOVQ	R5, V3.V[1]
	MOVV	32(R4), R5
	VMOVQ	R5, V4.V[0]
	MOVV	72(R4), R5
	VMOVQ	R5, V4.V[1]

	MOVV	80(R4), R5
	VMOVQ	R5, V5.V[0]
	MOVV	120(R4), R5
	VMOVQ	R5, V5.V[1]
	MOVV	88(R4), R5
	VMOVQ	R5, V6.V[0]
	MOVV	128(R4), R5
	VMOVQ	R5, V6.V[1]
	MOVV	96(R4), R5
	VMOVQ	R5, V7.V[0]
	MOVV	136(R4), R5
	VMOVQ	R5, V7.V[1]
	MOVV	104(R4), R5
	VMOVQ	R5, V8.V[0]
	MOVV	144(R4), R5
	VMOVQ	R5, V8.V[1]
	MOVV	112(R4), R5
	VMOVQ	R5, V9.V[0]
	MOVV	152(R4), R5
	VMOVQ	R5, V9.V[1]

	VXORV	V10, V10, V10
	VXORV	V11, V11, V11
	VXORV	V12, V12, V12
	VXORV	V13, V13, V13
	VXORV	V14, V14, V14
	MOVV	160(R4), R5
	VMOVQ	R5, V10.V[0]
	MOVV	168(R4), R5
	VMOVQ	R5, V11.V[0]
	MOVV	176(R4), R5
	VMOVQ	R5, V12.V[0]
	MOVV	184(R4), R5
	VMOVQ	R5, V13.V[0]
	MOVV	192(R4), R5
	VMOVQ	R5, V14.V[0]

loop:
	// Theta: C[x] = parity of column x -> V15+x, both elements.
	// T = A^B^(C&mask); PC = T ^ swap(T).
	VXORV	V0, V5, V30
	VANDV	V31, V10, V29
	VXORV	V30, V29, V30
	VILVHV	V30, V30, V29
	VILVLV	V29, V30, V29
	VXORV	V30, V29, V15	// C[0]

	VXORV	V1, V6, V30
	VANDV	V31, V11, V29
	VXORV	V30, V29, V30
	VILVHV	V30, V30, V29
	VILVLV	V29, V30, V29
	VXORV	V30, V29, V16	// C[1]

	VXORV	V2, V7, V30
	VANDV	V31, V12, V29
	VXORV	V30, V29, V30
	VILVHV	V30, V30, V29
	VILVLV	V29, V30, V29
	VXORV	V30, V29, V17	// C[2]

	VXORV	V3, V8, V30
	VANDV	V31, V13, V29
	VXORV	V30, V29, V30
	VILVHV	V30, V30, V29
	VILVLV	V29, V30, V29
	VXORV	V30, V29, V18	// C[3]

	VXORV	V4, V9, V30
	VANDV	V31, V14, V29
	VXORV	V30, V29, V30
	VILVHV	V30, V30, V29
	VILVLV	V29, V30, V29
	VXORV	V30, V29, V19	// C[4]

	// D_x = C[x-1] ^ rotl(C[x+1],1) -> V20+x (broadcast both elements).
	VROTRV	$63, V16, V30
	VXORV	V19, V30, V20	// D0
	VROTRV	$63, V17, V30
	VXORV	V15, V30, V21	// D1
	VROTRV	$63, V18, V30
	VXORV	V16, V30, V22	// D2
	VROTRV	$63, V19, V30
	VXORV	V17, V30, V23	// D3
	VROTRV	$63, V15, V30
	VXORV	V18, V30, V24	// D4

	// Theta apply + rho: per reg, R = rotr(R ^ D_x, rho-pair).
	VXORV	V20, V0, V0
	VMOVQ	0(R8), V30
	VROTRV	V30, V0, V0
	VXORV	V21, V1, V1
	VMOVQ	16(R8), V30
	VROTRV	V30, V1, V1
	VXORV	V22, V2, V2
	VMOVQ	32(R8), V30
	VROTRV	V30, V2, V2
	VXORV	V23, V3, V3
	VMOVQ	48(R8), V30
	VROTRV	V30, V3, V3
	VXORV	V24, V4, V4
	VMOVQ	64(R8), V30
	VROTRV	V30, V4, V4

	VXORV	V20, V5, V5
	VMOVQ	80(R8), V30
	VROTRV	V30, V5, V5
	VXORV	V21, V6, V6
	VMOVQ	96(R8), V30
	VROTRV	V30, V6, V6
	VXORV	V22, V7, V7
	VMOVQ	112(R8), V30
	VROTRV	V30, V7, V7
	VXORV	V23, V8, V8
	VMOVQ	128(R8), V30
	VROTRV	V30, V8, V8
	VXORV	V24, V9, V9
	VMOVQ	144(R8), V30
	VROTRV	V30, V9, V9

	VXORV	V20, V10, V10
	VMOVQ	160(R8), V30
	VROTRV	V30, V10, V10
	VXORV	V21, V11, V11
	VMOVQ	176(R8), V30
	VROTRV	V30, V11, V11
	VXORV	V22, V12, V12
	VMOVQ	192(R8), V30
	VROTRV	V30, V12, V12
	VXORV	V23, V13, V13
	VMOVQ	208(R8), V30
	VROTRV	V30, V13, V13
	VXORV	V24, V14, V14
	VMOVQ	224(R8), V30
	VROTRV	V30, V14, V14

	// Pi: gather post-rho lanes into V15-V29.
	// A'_x' = [L'(x',x'), L'((x'+3)%5, x')]
	VILVLV	V0, V3, V15	// [L'(0,0), L'(3,0)]
	VILVHV	V1, V4, V16	// [L'(1,1), L'(4,1)]
	VILVLV	V7, V5, V17	// [L'(2,2), L'(0,2)]
	VILVHV	V8, V6, V18	// [L'(3,3), L'(1,3)]
	VILVLV	V14, V12, V19	// [L'(4,4), L'(2,4)]

	// B'_x' = [L'((x'+1)%5, x'), L'((x'+4)%5, x')]
	VILVLV	V1, V4, V20	// [L'(1,0), L'(4,0)]
	VILVHV	V2, V0, V21	// [L'(2,1), L'(0,1)]
	VILVLV	V8, V6, V22	// [L'(3,2), L'(1,2)]
	VILVHV	V9, V7, V23	// [L'(4,3), L'(2,3)]
	VILVLV	V10, V13, V24	// [L'(0,4), L'(3,4)]

	// C'_x' = [L'((x'+2)%5, x'), 0]  (V31.e1 = 0 supplies the zero)
	VILVLV	V2, V31, V25	// [L'(2,0), 0]
	VILVHV	V3, V31, V26	// [L'(3,1), 0]
	VILVLV	V9, V31, V27	// [L'(4,2), 0]
	VILVHV	V5, V31, V28	// [L'(0,3), 0]
	VILVLV	V11, V31, V29	// [L'(1,4), 0]

	// Chi: out_x = in_x ^ (~in_{x+1} & in_{x+2}), per group, mod 5.
	// Group A: sources V15-V19 -> dests V0-V4.
	VANDNV	V17, V16, V30
	VXORV	V15, V30, V0
	VANDNV	V18, V17, V30
	VXORV	V16, V30, V1
	VANDNV	V19, V18, V30
	VXORV	V17, V30, V2
	VANDNV	V15, V19, V30
	VXORV	V18, V30, V3
	VANDNV	V16, V15, V30
	VXORV	V19, V30, V4

	// Group B: sources V20-V24 -> dests V5-V9.
	VANDNV	V22, V21, V30
	VXORV	V20, V30, V5
	VANDNV	V23, V22, V30
	VXORV	V21, V30, V6
	VANDNV	V24, V23, V30
	VXORV	V22, V30, V7
	VANDNV	V20, V24, V30
	VXORV	V23, V30, V8
	VANDNV	V21, V20, V30
	VXORV	V24, V30, V9

	// Group C: sources V25-V29 -> dests V10-V14.
	VANDNV	V27, V26, V30
	VXORV	V25, V30, V10
	VANDNV	V28, V27, V30
	VXORV	V26, V30, V11
	VANDNV	V29, V28, V30
	VXORV	V27, V30, V12
	VANDNV	V25, V29, V30
	VXORV	V28, V30, V13
	VANDNV	V26, V25, V30
	VXORV	V29, V30, V14

	// Iota: lane(0,0) = V0.e0 ^= RC[round] (mask to element 0).
	MOVV	(R6), R5
	VXORV	V30, V30, V30
	VMOVQ	R5, V30.V[0]
	VXORV	V30, V0, V0

	ADDV	$8, R6
	SUBV	$1, R7
	BNE	R7, R0, loop

	// Store lanes back.
	VMOVQ	V0.V[0], 0(R4)
	VMOVQ	V0.V[1], 40(R4)
	VMOVQ	V1.V[0], 8(R4)
	VMOVQ	V1.V[1], 48(R4)
	VMOVQ	V2.V[0], 16(R4)
	VMOVQ	V2.V[1], 56(R4)
	VMOVQ	V3.V[0], 24(R4)
	VMOVQ	V3.V[1], 64(R4)
	VMOVQ	V4.V[0], 32(R4)
	VMOVQ	V4.V[1], 72(R4)
	VMOVQ	V5.V[0], 80(R4)
	VMOVQ	V5.V[1], 120(R4)
	VMOVQ	V6.V[0], 88(R4)
	VMOVQ	V6.V[1], 128(R4)
	VMOVQ	V7.V[0], 96(R4)
	VMOVQ	V7.V[1], 136(R4)
	VMOVQ	V8.V[0], 104(R4)
	VMOVQ	V8.V[1], 144(R4)
	VMOVQ	V9.V[0], 112(R4)
	VMOVQ	V9.V[1], 152(R4)
	VMOVQ	V10.V[0], 160(R4)
	VMOVQ	V11.V[0], 168(R4)
	VMOVQ	V12.V[0], 176(R4)
	VMOVQ	V13.V[0], 184(R4)
	VMOVQ	V14.V[0], 192(R4)
	RET

// rho_pairs: per-register right-rotation amounts = 64 - rho[lane].
// Order: A_0..A_4, B_0..B_4, C_0..C_4.
DATA	rho_pairs<>+0x00(SB)/8, $0x0000000000000000	// r(0,0)=0
DATA	rho_pairs<>+0x08(SB)/8, $0x000000000000001c	// r(0,1)=36 -> 28
DATA	rho_pairs<>+0x10(SB)/8, $0x000000000000003f	// r(1,0)=1 -> 63
DATA	rho_pairs<>+0x18(SB)/8, $0x0000000000000014	// r(1,1)=44 -> 20
DATA	rho_pairs<>+0x20(SB)/8, $0x0000000000000002	// r(2,0)=62 -> 2
DATA	rho_pairs<>+0x28(SB)/8, $0x000000000000003a	// r(2,1)=6 -> 58
DATA	rho_pairs<>+0x30(SB)/8, $0x0000000000000024	// r(3,0)=28 -> 36
DATA	rho_pairs<>+0x38(SB)/8, $0x0000000000000009	// r(3,1)=55 -> 9
DATA	rho_pairs<>+0x40(SB)/8, $0x0000000000000025	// r(4,0)=27 -> 37
DATA	rho_pairs<>+0x48(SB)/8, $0x000000000000002c	// r(4,1)=20 -> 44

DATA	rho_pairs<>+0x50(SB)/8, $0x000000000000003d	// r(0,2)=3 -> 61
DATA	rho_pairs<>+0x58(SB)/8, $0x0000000000000017	// r(0,3)=41 -> 23
DATA	rho_pairs<>+0x60(SB)/8, $0x0000000000000036	// r(1,2)=10 -> 54
DATA	rho_pairs<>+0x68(SB)/8, $0x0000000000000013	// r(1,3)=45 -> 19
DATA	rho_pairs<>+0x70(SB)/8, $0x0000000000000015	// r(2,2)=43 -> 21
DATA	rho_pairs<>+0x78(SB)/8, $0x0000000000000031	// r(2,3)=15 -> 49
DATA	rho_pairs<>+0x80(SB)/8, $0x0000000000000027	// r(3,2)=25 -> 39
DATA	rho_pairs<>+0x88(SB)/8, $0x000000000000002b	// r(3,3)=21 -> 43
DATA	rho_pairs<>+0x90(SB)/8, $0x0000000000000019	// r(4,2)=39 -> 25
DATA	rho_pairs<>+0x98(SB)/8, $0x0000000000000038	// r(4,3)=8 -> 56

DATA	rho_pairs<>+0xa0(SB)/8, $0x000000000000002e	// r(0,4)=18 -> 46
DATA	rho_pairs<>+0xa8(SB)/8, $0
DATA	rho_pairs<>+0xb0(SB)/8, $0x000000000000003e	// r(1,4)=2 -> 62
DATA	rho_pairs<>+0xb8(SB)/8, $0
DATA	rho_pairs<>+0xc0(SB)/8, $0x0000000000000003	// r(2,4)=61 -> 3
DATA	rho_pairs<>+0xc8(SB)/8, $0
DATA	rho_pairs<>+0xd0(SB)/8, $0x0000000000000008	// r(3,4)=56 -> 8
DATA	rho_pairs<>+0xd8(SB)/8, $0
DATA	rho_pairs<>+0xe0(SB)/8, $0x0000000000000032	// r(4,4)=14 -> 50
DATA	rho_pairs<>+0xe8(SB)/8, $0
GLOBL	rho_pairs<>(SB), NOPTR|RODATA, $240

// Round constants
DATA	round_consts<>+0x00(SB)/8, $0x0000000000000001
DATA	round_consts<>+0x08(SB)/8, $0x0000000000008082
DATA	round_consts<>+0x10(SB)/8, $0x800000000000808a
DATA	round_consts<>+0x18(SB)/8, $0x8000000080008000
DATA	round_consts<>+0x20(SB)/8, $0x000000000000808b
DATA	round_consts<>+0x28(SB)/8, $0x0000000080000001
DATA	round_consts<>+0x30(SB)/8, $0x8000000080008081
DATA	round_consts<>+0x38(SB)/8, $0x8000000000008009
DATA	round_consts<>+0x40(SB)/8, $0x000000000000008a
DATA	round_consts<>+0x48(SB)/8, $0x0000000000000088
DATA	round_consts<>+0x50(SB)/8, $0x0000000080008009
DATA	round_consts<>+0x58(SB)/8, $0x000000008000000a
DATA	round_consts<>+0x60(SB)/8, $0x000000008000808b
DATA	round_consts<>+0x68(SB)/8, $0x800000000000008b
DATA	round_consts<>+0x70(SB)/8, $0x8000000000008089
DATA	round_consts<>+0x78(SB)/8, $0x8000000000008003
DATA	round_consts<>+0x80(SB)/8, $0x8000000000008002
DATA	round_consts<>+0x88(SB)/8, $0x8000000000000080
DATA	round_consts<>+0x90(SB)/8, $0x000000000000800a
DATA	round_consts<>+0x98(SB)/8, $0x800000008000000a
DATA	round_consts<>+0xA0(SB)/8, $0x8000000080008081
DATA	round_consts<>+0xA8(SB)/8, $0x8000000000008080
DATA	round_consts<>+0xB0(SB)/8, $0x0000000080000001
DATA	round_consts<>+0xB8(SB)/8, $0x8000000080008008
GLOBL	round_consts<>(SB), NOPTR|RODATA, $192
