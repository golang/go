// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file was originally from https://golang.org/cl/24717 by Vlad Krasnov of CloudFlare.

//go:build gc && !purego

#include "textflag.h"
// General register allocation
#define oup DI
#define inp SI
#define inl BX
#define adp CX // free to reuse, after we hash the additional data
#define keyp R8 // free to reuse, when we copy the key to stack
#define itr2 R9 // general iterator
#define itr1 CX // general iterator
#define acc0 R10
#define acc1 R11
#define acc2 R12
#define t0 R13
#define t1 R14
#define t2 R15
#define t3 R8
// Register and stack allocation for the SSE code
#define rStore (0*16)(BP)
#define sStore (1*16)(BP)
#define state1Store (2*16)(BP)
#define state2Store (3*16)(BP)
#define tmpStore (4*16)(BP)
#define ctr0Store (5*16)(BP)
#define ctr1Store (6*16)(BP)
#define ctr2Store (7*16)(BP)
#define ctr3Store (8*16)(BP)
#define A0 X0
#define A1 X1
#define A2 X2
#define B0 X3
#define B1 X4
#define B2 X5
#define C0 X6
#define C1 X7
#define C2 X8
#define D0 X9
#define D1 X10
#define D2 X11
#define T0 X12
#define T1 X13
#define T2 X14
#define T3 X15
#define A3 T0
#define B3 T1
#define C3 T2
#define D3 T3
// Register and stack allocation for the AVX2 code
#define rsStoreAVX2 (0*32)(BP)
#define state1StoreAVX2 (1*32)(BP)
#define state2StoreAVX2 (2*32)(BP)
#define ctr0StoreAVX2 (3*32)(BP)
#define ctr1StoreAVX2 (4*32)(BP)
#define ctr2StoreAVX2 (5*32)(BP)
#define ctr3StoreAVX2 (6*32)(BP)
#define tmpStoreAVX2 (7*32)(BP) // 256 bytes on stack
#define AA0 Y0
#define AA1 Y5
#define AA2 Y6
#define AA3 Y7
#define BB0 Y14
#define BB1 Y9
#define BB2 Y10
#define BB3 Y11
#define CC0 Y12
#define CC1 Y13
#define CC2 Y8
#define CC3 Y15
#define DD0 Y4
#define DD1 Y1
#define DD2 Y2
#define DD3 Y3
#define TT0 DD3
#define TT1 AA3
#define TT2 BB3
#define TT3 CC3
// ChaCha20 constants
DATA ·chacha20Constants<>+0x00(SB)/4, $0x61707865
DATA ·chacha20Constants<>+0x04(SB)/4, $0x3320646e
DATA ·chacha20Constants<>+0x08(SB)/4, $0x79622d32
DATA ·chacha20Constants<>+0x0c(SB)/4, $0x6b206574
DATA ·chacha20Constants<>+0x10(SB)/4, $0x61707865
DATA ·chacha20Constants<>+0x14(SB)/4, $0x3320646e
DATA ·chacha20Constants<>+0x18(SB)/4, $0x79622d32
DATA ·chacha20Constants<>+0x1c(SB)/4, $0x6b206574
// <<< 16 with PSHUFB
DATA ·rol16<>+0x00(SB)/8, $0x0504070601000302
DATA ·rol16<>+0x08(SB)/8, $0x0D0C0F0E09080B0A
DATA ·rol16<>+0x10(SB)/8, $0x0504070601000302
DATA ·rol16<>+0x18(SB)/8, $0x0D0C0F0E09080B0A
// <<< 8 with PSHUFB
DATA ·rol8<>+0x00(SB)/8, $0x0605040702010003
DATA ·rol8<>+0x08(SB)/8, $0x0E0D0C0F0A09080B
DATA ·rol8<>+0x10(SB)/8, $0x0605040702010003
DATA ·rol8<>+0x18(SB)/8, $0x0E0D0C0F0A09080B

DATA ·avx2InitMask<>+0x00(SB)/8, $0x0
DATA ·avx2InitMask<>+0x08(SB)/8, $0x0
DATA ·avx2InitMask<>+0x10(SB)/8, $0x1
DATA ·avx2InitMask<>+0x18(SB)/8, $0x0

DATA ·avx2IncMask<>+0x00(SB)/8, $0x2
DATA ·avx2IncMask<>+0x08(SB)/8, $0x0
DATA ·avx2IncMask<>+0x10(SB)/8, $0x2
DATA ·avx2IncMask<>+0x18(SB)/8, $0x0
// Poly1305 key clamp
DATA ·polyClampMask<>+0x00(SB)/8, $0x0FFFFFFC0FFFFFFF
DATA ·polyClampMask<>+0x08(SB)/8, $0x0FFFFFFC0FFFFFFC
DATA ·polyClampMask<>+0x10(SB)/8, $0xFFFFFFFFFFFFFFFF
DATA ·polyClampMask<>+0x18(SB)/8, $0xFFFFFFFFFFFFFFFF

DATA ·sseIncMask<>+0x00(SB)/8, $0x1
DATA ·sseIncMask<>+0x08(SB)/8, $0x0
// To load/store the last < 16 bytes in a buffer
DATA ·andMask<>+0x00(SB)/8, $0x00000000000000ff
DATA ·andMask<>+0x08(SB)/8, $0x0000000000000000
DATA ·andMask<>+0x10(SB)/8, $0x000000000000ffff
DATA ·andMask<>+0x18(SB)/8, $0x0000000000000000
DATA ·andMask<>+0x20(SB)/8, $0x0000000000ffffff
DATA ·andMask<>+0x28(SB)/8, $0x0000000000000000
DATA ·andMask<>+0x30(SB)/8, $0x00000000ffffffff
DATA ·andMask<>+0x38(SB)/8, $0x0000000000000000
DATA ·andMask<>+0x40(SB)/8, $0x000000ffffffffff
DATA ·andMask<>+0x48(SB)/8, $0x0000000000000000
DATA ·andMask<>+0x50(SB)/8, $0x0000ffffffffffff
DATA ·andMask<>+0x58(SB)/8, $0x0000000000000000
DATA ·andMask<>+0x60(SB)/8, $0x00ffffffffffffff
DATA ·andMask<>+0x68(SB)/8, $0x0000000000000000
DATA ·andMask<>+0x70(SB)/8, $0xffffffffffffffff
DATA ·andMask<>+0x78(SB)/8, $0x0000000000000000
DATA ·andMask<>+0x80(SB)/8, $0xffffffffffffffff
DATA ·andMask<>+0x88(SB)/8, $0x00000000000000ff
DATA ·andMask<>+0x90(SB)/8, $0xffffffffffffffff
DATA ·andMask<>+0x98(SB)/8, $0x000000000000ffff
DATA ·andMask<>+0xa0(SB)/8, $0xffffffffffffffff
DATA ·andMask<>+0xa8(SB)/8, $0x0000000000ffffff
DATA ·andMask<>+0xb0(SB)/8, $0xffffffffffffffff
DATA ·andMask<>+0xb8(SB)/8, $0x00000000ffffffff
DATA ·andMask<>+0xc0(SB)/8, $0xffffffffffffffff
DATA ·andMask<>+0xc8(SB)/8, $0x000000ffffffffff
DATA ·andMask<>+0xd0(SB)/8, $0xffffffffffffffff
DATA ·andMask<>+0xd8(SB)/8, $0x0000ffffffffffff
DATA ·andMask<>+0xe0(SB)/8, $0xffffffffffffffff
DATA ·andMask<>+0xe8(SB)/8, $0x00ffffffffffffff

GLOBL ·chacha20Constants<>(SB), (NOPTR+RODATA), $32
GLOBL ·rol16<>(SB), (NOPTR+RODATA), $32
GLOBL ·rol8<>(SB), (NOPTR+RODATA), $32
GLOBL ·sseIncMask<>(SB), (NOPTR+RODATA), $16
GLOBL ·avx2IncMask<>(SB), (NOPTR+RODATA), $32
GLOBL ·avx2InitMask<>(SB), (NOPTR+RODATA), $32
GLOBL ·polyClampMask<>(SB), (NOPTR+RODATA), $32
GLOBL ·andMask<>(SB), (NOPTR+RODATA), $240
// No PALIGNR in Go ASM yet (but VPALIGNR is present).
#define shiftB0Left BYTE $0x66; BYTE $0x0f; BYTE $0x3a; BYTE $0x0f; BYTE $0xdb; BYTE $0x04 // PALIGNR $4, X3, X3
#define shiftB1Left BYTE $0x66; BYTE $0x0f; BYTE $0x3a; BYTE $0x0f; BYTE $0xe4; BYTE $0x04 // PALIGNR $4, X4, X4
#define shiftB2Left BYTE $0x66; BYTE $0x0f; BYTE $0x3a; BYTE $0x0f; BYTE $0xed; BYTE $0x04 // PALIGNR $4, X5, X5
#define shiftB3Left BYTE $0x66; BYTE $0x45; BYTE $0x0f; BYTE $0x3a; BYTE $0x0f; BYTE $0xed; BYTE $0x04 // PALIGNR $4, X13, X13
#define shiftC0Left BYTE $0x66; BYTE $0x0f; BYTE $0x3a; BYTE $0x0f; BYTE $0xf6; BYTE $0x08 // PALIGNR $8, X6, X6
#define shiftC1Left BYTE $0x66; BYTE $0x0f; BYTE $0x3a; BYTE $0x0f; BYTE $0xff; BYTE $0x08 // PALIGNR $8, X7, X7
#define shiftC2Left BYTE $0x66; BYTE $0x45; BYTE $0x0f; BYTE $0x3a; BYTE $0x0f; BYTE $0xc0; BYTE $0x08 // PALIGNR $8, X8, X8
#define shiftC3Left BYTE $0x66; BYTE $0x45; BYTE $0x0f; BYTE $0x3a; BYTE $0x0f; BYTE $0xf6; BYTE $0x08 // PALIGNR $8, X14, X14
#define shiftD0Left BYTE $0x66; BYTE $0x45; BYTE $0x0f; BYTE $0x3a; BYTE $0x0f; BYTE $0xc9; BYTE $0x0c // PALIGNR $12, X9, X9
#define shiftD1Left BYTE $0x66; BYTE $0x45; BYTE $0x0f; BYTE $0x3a; BYTE $0x0f; BYTE $0xd2; BYTE $0x0c // PALIGNR $12, X10, X10
#define shiftD2Left BYTE $0x66; BYTE $0x45; BYTE $0x0f; BYTE $0x3a; BYTE $0x0f; BYTE $0xdb; BYTE $0x0c // PALIGNR $12, X11, X11
#define shiftD3Left BYTE $0x66; BYTE $0x45; BYTE $0x0f; BYTE $0x3a; BYTE $0x0f; BYTE $0xff; BYTE $0x0c // PALIGNR $12, X15, X15
#define shiftB0Right BYTE $0x66; BYTE $0x0f; BYTE $0x3a; BYTE $0x0f; BYTE $0xdb; BYTE $0x0c // PALIGNR $12, X3, X3
#define shiftB1Right BYTE $0x66; BYTE $0x0f; BYTE $0x3a; BYTE $0x0f; BYTE $0xe4; BYTE $0x0c // PALIGNR $12, X4, X4
#define shiftB2Right BYTE $0x66; BYTE $0x0f; BYTE $0x3a; BYTE $0x0f; BYTE $0xed; BYTE $0x0c // PALIGNR $12, X5, X5
#define shiftB3Right BYTE $0x66; BYTE $0x45; BYTE $0x0f; BYTE $0x3a; BYTE $0x0f; BYTE $0xed; BYTE $0x0c // PALIGNR $12, X13, X13
#define shiftC0Right shiftC0Left
#define shiftC1Right shiftC1Left
#define shiftC2Right shiftC2Left
#define shiftC3Right shiftC3Left
#define shiftD0Right BYTE $0x66; BYTE $0x45; BYTE $0x0f; BYTE $0x3a; BYTE $0x0f; BYTE $0xc9; BYTE $0x04 // PALIGNR $4, X9, X9
#define shiftD1Right BYTE $0x66; BYTE $0x45; BYTE $0x0f; BYTE $0x3a; BYTE $0x0f; BYTE $0xd2; BYTE $0x04 // PALIGNR $4, X10, X10
#define shiftD2Right BYTE $0x66; BYTE $0x45; BYTE $0x0f; BYTE $0x3a; BYTE $0x0f; BYTE $0xdb; BYTE $0x04 // PALIGNR $4, X11, X11
#define shiftD3Right BYTE $0x66; BYTE $0x45; BYTE $0x0f; BYTE $0x3a; BYTE $0x0f; BYTE $0xff; BYTE $0x04 // PALIGNR $4, X15, X15

// Some macros

// ROL rotates the uint32s in register R left by N bits, using temporary T.
#define ROL(N, R, T) \
	MOVO R, T; PSLLL $(N), T; PSRLL $(32-(N)), R; PXOR T, R

// ROL16 rotates the uint32s in register R left by 16, using temporary T if needed.
#ifdef GOAMD64_v2
#define ROL16(R, T) PSHUFB ·rol16<>(SB), R
#else
#define ROL16(R, T) ROL(16, R, T)
#endif

// ROL8 rotates the uint32s in register R left by 8, using temporary T if needed.
#ifdef GOAMD64_v2
#define ROL8(R, T) PSHUFB ·rol8<>(SB), R
#else
#define ROL8(R, T) ROL(8, R, T)
#endif

#define chachaQR(A, B, C, D, T) \
	PADDD B, A; PXOR A, D; ROL16(D, T) \
	PADDD D, C; PXOR C, B; MOVO B, T; PSLLL $12, T; PSRLL $20, B; PXOR T, B \
	PADDD B, A; PXOR A, D; ROL8(D, T) \
	PADDD D, C; PXOR C, B; MOVO B, T; PSLLL $7, T; PSRLL $25, B; PXOR T, B

#define chachaQR_AVX2(A, B, C, D, T) \
	VPADDD B, A, A; VPXOR A, D, D; VPSHUFB ·rol16<>(SB), D, D                         \
	VPADDD D, C, C; VPXOR C, B, B; VPSLLD $12, B, T; VPSRLD $20, B, B; VPXOR T, B, B \
	VPADDD B, A, A; VPXOR A, D, D; VPSHUFB ·rol8<>(SB), D, D                          \
	VPADDD D, C, C; VPXOR C, B, B; VPSLLD $7, B, T; VPSRLD $25, B, B; VPXOR T, B, B

#define polyAdd(S) ADDQ S, acc0; ADCQ 8+S, acc1; ADCQ $1, acc2
#define polyMulStage1 MOVQ (0*8)(BP), AX; MOVQ AX, t2; MULQ acc0; MOVQ AX, t0; MOVQ DX, t1; MOVQ (0*8)(BP), AX; MULQ acc1; IMULQ acc2, t2; ADDQ AX, t1; ADCQ DX, t2
#define polyMulStage2 MOVQ (1*8)(BP), AX; MOVQ AX, t3; MULQ acc0; ADDQ AX, t1; ADCQ $0, DX; MOVQ DX, acc0; MOVQ (1*8)(BP), AX; MULQ acc1; ADDQ AX, t2; ADCQ $0, DX
#define polyMulStage3 IMULQ acc2, t3; ADDQ acc0, t2; ADCQ DX, t3
#define polyMulReduceStage MOVQ t0, acc0; MOVQ t1, acc1; MOVQ t2, acc2; ANDQ $3, acc2; MOVQ t2, t0; ANDQ $-4, t0; MOVQ t3, t1; SHRQ $2, t3, t2; SHRQ $2, t3; ADDQ t0, acc0; ADCQ t1, acc1; ADCQ $0, acc2; ADDQ t2, acc0; ADCQ t3, acc1; ADCQ $0, acc2

#define polyMulStage1_AVX2 MOVQ (0*8)(BP), DX; MOVQ DX, t2; MULXQ acc0, t0, t1; IMULQ acc2, t2; MULXQ acc1, AX, DX; ADDQ AX, t1; ADCQ DX, t2
#define polyMulStage2_AVX2 MOVQ (1*8)(BP), DX; MULXQ acc0, acc0, AX; ADDQ acc0, t1; MULXQ acc1, acc1, t3; ADCQ acc1, t2; ADCQ $0, t3
#define polyMulStage3_AVX2 IMULQ acc2, DX; ADDQ AX, t2; ADCQ DX, t3

#define polyMul polyMulStage1; polyMulStage2; polyMulStage3; polyMulReduceStage
#define polyMulAVX2 polyMulStage1_AVX2; polyMulStage2_AVX2; polyMulStage3_AVX2; polyMulReduceStage
// ----------------------------------------------------------------------------
TEXT polyHashADInternal<>(SB), NOSPLIT, $0
	// adp points to beginning of additional data
	// itr2 holds ad length
	XORQ acc0, acc0
	XORQ acc1, acc1
	XORQ acc2, acc2
	CMPQ itr2, $13
	JNE  hashADLoop

openFastTLSAD:
	// Special treatment for the TLS case of 13 bytes
	MOVQ (adp), acc0
	MOVQ 5(adp), acc1
	SHRQ $24, acc1
	MOVQ $1, acc2
	polyMul
	RET

hashADLoop:
	// Hash in 16 byte chunks
	CMPQ itr2, $16
	JB   hashADTail
	polyAdd(0(adp))
	LEAQ (1*16)(adp), adp
	SUBQ $16, itr2
	polyMul
	JMP  hashADLoop

hashADTail:
	CMPQ itr2, $0
	JE   hashADDone

	// Hash last < 16 byte tail
	XORQ t0, t0
	XORQ t1, t1
	XORQ t2, t2
	ADDQ itr2, adp

hashADTailLoop:
	SHLQ $8, t0, t1
	SHLQ $8, t0
	MOVB -1(adp), t2
	XORQ t2, t0
	DECQ adp
	DECQ itr2
	JNE  hashADTailLoop

hashADTailFinish:
	ADDQ t0, acc0; ADCQ t1, acc1; ADCQ $1, acc2
	polyMul

	// Finished AD
hashADDone:
	RET

// ----------------------------------------------------------------------------
// func chacha20Poly1305Open(dst, key, src, ad []byte) bool
TEXT ·chacha20Poly1305Open(SB), 0, $288-97
	// For aligned stack access
	MOVQ SP, BP
	ADDQ $32, BP
	ANDQ $-32, BP
	MOVQ dst+0(FP), oup
	MOVQ key+24(FP), keyp
	MOVQ src+48(FP), inp
	MOVQ src_len+56(FP), inl
	MOVQ ad+72(FP), adp

	// Check for AVX2 support
	CMPB ·useAVX2(SB), $1
	JE   chacha20Poly1305Open_AVX2

	// Special optimization, for very short buffers
	CMPQ inl, $128
	JBE  openSSE128 // About 16% faster

	// For long buffers, prepare the poly key first
	MOVOU ·chacha20Constants<>(SB), A0
	MOVOU (1*16)(keyp), B0
	MOVOU (2*16)(keyp), C0
	MOVOU (3*16)(keyp), D0
	MOVO  D0, T1

	// Store state on stack for future use
	MOVO B0, state1Store
	MOVO C0, state2Store
	MOVO D0, ctr3Store
	MOVQ $10, itr2

openSSEPreparePolyKey:
	chachaQR(A0, B0, C0, D0, T0)
	shiftB0Left;  shiftC0Left; shiftD0Left
	chachaQR(A0, B0, C0, D0, T0)
	shiftB0Right; shiftC0Right; shiftD0Right
	DECQ          itr2
	JNE           openSSEPreparePolyKey

	// A0|B0 hold the Poly1305 32-byte key, C0,D0 can be discarded
	PADDL ·chacha20Constants<>(SB), A0; PADDL state1Store, B0

	// Clamp and store the key
	PAND ·polyClampMask<>(SB), A0
	MOVO A0, rStore; MOVO B0, sStore

	// Hash AAD
	MOVQ ad_len+80(FP), itr2
	CALL polyHashADInternal<>(SB)

openSSEMainLoop:
	CMPQ inl, $256
	JB   openSSEMainLoopDone

	// Load state, increment counter blocks
	MOVO ·chacha20Constants<>(SB), A0; MOVO state1Store, B0; MOVO state2Store, C0; MOVO ctr3Store, D0; PADDL ·sseIncMask<>(SB), D0
	MOVO A0, A1; MOVO B0, B1; MOVO C0, C1; MOVO D0, D1; PADDL ·sseIncMask<>(SB), D1
	MOVO A1, A2; MOVO B1, B2; MOVO C1, C2; MOVO D1, D2; PADDL ·sseIncMask<>(SB), D2
	MOVO A2, A3; MOVO B2, B3; MOVO C2, C3; MOVO D2, D3; PADDL ·sseIncMask<>(SB), D3

	// Store counters
	MOVO D0, ctr0Store; MOVO D1, ctr1Store; MOVO D2, ctr2Store; MOVO D3, ctr3Store

	// There are 10 ChaCha20 iterations of 2QR each, so for 6 iterations we hash 2 blocks, and for the remaining 4 only 1 block - for a total of 16
	MOVQ $4, itr1
	MOVQ inp, itr2

openSSEInternalLoop:
	MOVO          C3, tmpStore
	chachaQR(A0, B0, C0, D0, C3); chachaQR(A1, B1, C1, D1, C3); chachaQR(A2, B2, C2, D2, C3)
	MOVO          tmpStore, C3
	MOVO          C1, tmpStore
	chachaQR(A3, B3, C3, D3, C1)
	MOVO          tmpStore, C1
	polyAdd(0(itr2))
	shiftB0Left;  shiftB1Left; shiftB2Left; shiftB3Left
	shiftC0Left;  shiftC1Left; shiftC2Left; shiftC3Left
	shiftD0Left;  shiftD1Left; shiftD2Left; shiftD3Left
	polyMulStage1
	polyMulStage2
	LEAQ          (2*8)(itr2), itr2
	MOVO          C3, tmpStore
	chachaQR(A0, B0, C0, D0, C3); chachaQR(A1, B1, C1, D1, C3); chachaQR(A2, B2, C2, D2, C3)
	MOVO          tmpStore, C3
	MOVO          C1, tmpStore
	polyMulStage3
	chachaQR(A3, B3, C3, D3, C1)
	MOVO          tmpStore, C1
	polyMulReduceStage
	shiftB0Right; shiftB1Right; shiftB2Right; shiftB3Right
	shiftC0Right; shiftC1Right; shiftC2Right; shiftC3Right
	shiftD0Right; shiftD1Right; shiftD2Right; shiftD3Right
	DECQ          itr1
	JGE           openSSEInternalLoop

	polyAdd(0(itr2))
	polyMul
	LEAQ (2*8)(itr2), itr2

	CMPQ itr1, $-6
	JG   openSSEInternalLoop

	// Add in the state
	PADDD ·chacha20Constants<>(SB), A0; PADDD ·chacha20Constants<>(SB), A1; PADDD ·chacha20Constants<>(SB), A2; PADDD ·chacha20Constants<>(SB), A3
	PADDD state1Store, B0; PADDD state1Store, B1; PADDD state1Store, B2; PADDD state1Store, B3
	PADDD state2Store, C0; PADDD state2Store, C1; PADDD state2Store, C2; PADDD state2Store, C3
	PADDD ctr0Store, D0; PADDD ctr1Store, D1; PADDD ctr2Store, D2; PADDD ctr3Store, D3

	// Load - xor - store
	MOVO  D3, tmpStore
	MOVOU (0*16)(inp), D3; PXOR D3, A0; MOVOU A0, (0*16)(oup)
	MOVOU (1*16)(inp), D3; PXOR D3, B0; MOVOU B0, (1*16)(oup)
	MOVOU (2*16)(inp), D3; PXOR D3, C0; MOVOU C0, (2*16)(oup)
	MOVOU (3*16)(inp), D3; PXOR D3, D0; MOVOU D0, (3*16)(oup)
	MOVOU (4*16)(inp), D0; PXOR D0, A1; MOVOU A1, (4*16)(oup)
	MOVOU (5*16)(inp), D0; PXOR D0, B1; MOVOU B1, (5*16)(oup)
	MOVOU (6*16)(inp), D0; PXOR D0, C1; MOVOU C1, (6*16)(oup)
	MOVOU (7*16)(inp), D0; PXOR D0, D1; MOVOU D1, (7*16)(oup)
	MOVOU (8*16)(inp), D0; PXOR D0, A2; MOVOU A2, (8*16)(oup)
	MOVOU (9*16)(inp), D0; PXOR D0, B2; MOVOU B2, (9*16)(oup)
	MOVOU (10*16)(inp), D0; PXOR D0, C2; MOVOU C2, (10*16)(oup)
	MOVOU (11*16)(inp), D0; PXOR D0, D2; MOVOU D2, (11*16)(oup)
	MOVOU (12*16)(inp), D0; PXOR D0, A3; MOVOU A3, (12*16)(oup)
	MOVOU (13*16)(inp), D0; PXOR D0, B3; MOVOU B3, (13*16)(oup)
	MOVOU (14*16)(inp), D0; PXOR D0, C3; MOVOU C3, (14*16)(oup)
	MOVOU (15*16)(inp), D0; PXOR tmpStore, D0; MOVOU D0, (15*16)(oup)
	LEAQ  256(inp), inp
	LEAQ  256(oup), oup
	SUBQ  $256, inl
	JMP   openSSEMainLoop

openSSEMainLoopDone:
	// Handle the various tail sizes efficiently
	TESTQ inl, inl
	JE    openSSEFinalize
	CMPQ  inl, $64
	JBE   openSSETail64
	CMPQ  inl, $128
	JBE   openSSETail128
	CMPQ  inl, $192
	JBE   openSSETail192
	JMP   openSSETail256

openSSEFinalize:
	// Hash in the PT, AAD lengths
	ADDQ ad_len+80(FP), acc0; ADCQ src_len+56(FP), acc1; ADCQ $1, acc2
	polyMul

	// Final reduce
	MOVQ    acc0, t0
	MOVQ    acc1, t1
	MOVQ    acc2, t2
	SUBQ    $-5, acc0
	SBBQ    $-1, acc1
	SBBQ    $3, acc2
	CMOVQCS t0, acc0
	CMOVQCS t1, acc1
	CMOVQCS t2, acc2

	// Add in the "s" part of the key
	ADDQ 0+sStore, acc0
	ADCQ 8+sStore, acc1

	// Finally, constant time compare to the tag at the end of the message
	XORQ    AX, AX
	MOVQ    $1, DX
	XORQ    (0*8)(inp), acc0
	XORQ    (1*8)(inp), acc1
	ORQ     acc1, acc0
	CMOVQEQ DX, AX

	// Return true iff tags are equal
	MOVB AX, ret+96(FP)
	RET

// ----------------------------------------------------------------------------
// Special optimization for buffers smaller than 129 bytes
openSSE128:
	// For up to 128 bytes of ciphertext and 64 bytes for the poly key, we require to process three blocks
	MOVOU ·chacha20Constants<>(SB), A0; MOVOU (1*16)(keyp), B0; MOVOU (2*16)(keyp), C0; MOVOU (3*16)(keyp), D0
	MOVO  A0, A1; MOVO B0, B1; MOVO C0, C1; MOVO D0, D1; PADDL ·sseIncMask<>(SB), D1
	MOVO  A1, A2; MOVO B1, B2; MOVO C1, C2; MOVO D1, D2; PADDL ·sseIncMask<>(SB), D2
	MOVO  B0, T1; MOVO C0, T2; MOVO D1, T3
	MOVQ  $10, itr2

openSSE128InnerCipherLoop:
	chachaQR(A0, B0, C0, D0, T0); chachaQR(A1, B1, C1, D1, T0); chachaQR(A2, B2, C2, D2, T0)
	shiftB0Left;  shiftB1Left; shiftB2Left
	shiftC0Left;  shiftC1Left; shiftC2Left
	shiftD0Left;  shiftD1Left; shiftD2Left
	chachaQR(A0, B0, C0, D0, T0); chachaQR(A1, B1, C1, D1, T0); chachaQR(A2, B2, C2, D2, T0)
	shiftB0Right; shiftB1Right; shiftB2Right
	shiftC0Right; shiftC1Right; shiftC2Right
	shiftD0Right; shiftD1Right; shiftD2Right
	DECQ          itr2
	JNE           openSSE128InnerCipherLoop

	// A0|B0 hold the Poly1305 32-byte key, C0,D0 can be discarded
	PADDL ·chacha20Constants<>(SB), A0; PADDL ·chacha20Constants<>(SB), A1; PADDL ·chacha20Constants<>(SB), A2
	PADDL T1, B0; PADDL T1, B1; PADDL T1, B2
	PADDL T2, C1; PADDL T2, C2
	PADDL T3, D1; PADDL ·sseIncMask<>(SB), T3; PADDL T3, D2

	// Clamp and store the key
	PAND  ·polyClampMask<>(SB), A0
	MOVOU A0, rStore; MOVOU B0, sStore

	// Hash
	MOVQ ad_len+80(FP), itr2
	CALL polyHashADInternal<>(SB)

openSSE128Open:
	CMPQ inl, $16
	JB   openSSETail16
	SUBQ $16, inl

	// Load for hashing
	polyAdd(0(inp))

	// Load for decryption
	MOVOU (inp), T0; PXOR T0, A1; MOVOU A1, (oup)
	LEAQ  (1*16)(inp), inp
	LEAQ  (1*16)(oup), oup
	polyMul

	// Shift the stream "left"
	MOVO B1, A1
	MOVO C1, B1
	MOVO D1, C1
	MOVO A2, D1
	MOVO B2, A2
	MOVO C2, B2
	MOVO D2, C2
	JMP  openSSE128Open

openSSETail16:
	TESTQ inl, inl
	JE    openSSEFinalize

	// We can safely load the CT from the end, because it is padded with the MAC
	MOVQ   inl, itr2
	SHLQ   $4, itr2
	LEAQ   ·andMask<>(SB), t0
	MOVOU  (inp), T0
	ADDQ   inl, inp
	PAND   -16(t0)(itr2*1), T0
	MOVO   T0, 0+tmpStore
	MOVQ   T0, t0
	MOVQ   8+tmpStore, t1
	PXOR   A1, T0

	// We can only store one byte at a time, since plaintext can be shorter than 16 bytes
openSSETail16Store:
	MOVQ T0, t3
	MOVB t3, (oup)
	PSRLDQ $1, T0
	INCQ   oup
	DECQ   inl
	JNE    openSSETail16Store
	ADDQ   t0, acc0; ADCQ t1, acc1; ADCQ $1, acc2
	polyMul
	JMP    openSSEFinalize

// ----------------------------------------------------------------------------
// Special optimization for the last 64 bytes of ciphertext
openSSETail64:
	// Need to decrypt up to 64 bytes - prepare single block
	MOVO ·chacha20Constants<>(SB), A0; MOVO state1Store, B0; MOVO state2Store, C0; MOVO ctr3Store, D0; PADDL ·sseIncMask<>(SB), D0; MOVO D0, ctr0Store
	XORQ itr2, itr2
	MOVQ inl, itr1
	CMPQ itr1, $16
	JB   openSSETail64LoopB

openSSETail64LoopA:
	// Perform ChaCha rounds, while hashing the remaining input
	polyAdd(0(inp)(itr2*1))
	polyMul
	SUBQ $16, itr1

openSSETail64LoopB:
	ADDQ          $16, itr2
	chachaQR(A0, B0, C0, D0, T0)
	shiftB0Left;  shiftC0Left; shiftD0Left
	chachaQR(A0, B0, C0, D0, T0)
	shiftB0Right; shiftC0Right; shiftD0Right

	CMPQ itr1, $16
	JAE  openSSETail64LoopA

	CMPQ itr2, $160
	JNE  openSSETail64LoopB

	PADDL ·chacha20Constants<>(SB), A0; PADDL state1Store, B0; PADDL state2Store, C0; PADDL ctr0Store, D0

openSSETail64DecLoop:
	CMPQ  inl, $16
	JB    openSSETail64DecLoopDone
	SUBQ  $16, inl
	MOVOU (inp), T0
	PXOR  T0, A0
	MOVOU A0, (oup)
	LEAQ  16(inp), inp
	LEAQ  16(oup), oup
	MOVO  B0, A0
	MOVO  C0, B0
	MOVO  D0, C0
	JMP   openSSETail64DecLoop

openSSETail64DecLoopDone:
	MOVO A0, A1
	JMP  openSSETail16

// ----------------------------------------------------------------------------
// Special optimization for the last 128 bytes of ciphertext
openSSETail128:
	// Need to decrypt up to 128 bytes - prepare two blocks
	MOVO ·chacha20Constants<>(SB), A1; MOVO state1Store, B1; MOVO state2Store, C1; MOVO ctr3Store, D1; PADDL ·sseIncMask<>(SB), D1; MOVO D1, ctr0Store
	MOVO A1, A0; MOVO B1, B0; MOVO C1, C0; MOVO D1, D0; PADDL ·sseIncMask<>(SB), D0; MOVO D0, ctr1Store
	XORQ itr2, itr2
	MOVQ inl, itr1
	ANDQ $-16, itr1

openSSETail128LoopA:
	// Perform ChaCha rounds, while hashing the remaining input
	polyAdd(0(inp)(itr2*1))
	polyMul

openSSETail128LoopB:
	ADDQ          $16, itr2
	chachaQR(A0, B0, C0, D0, T0); chachaQR(A1, B1, C1, D1, T0)
	shiftB0Left;  shiftC0Left; shiftD0Left
	shiftB1Left;  shiftC1Left; shiftD1Left
	chachaQR(A0, B0, C0, D0, T0); chachaQR(A1, B1, C1, D1, T0)
	shiftB0Right; shiftC0Right; shiftD0Right
	shiftB1Right; shiftC1Right; shiftD1Right

	CMPQ itr2, itr1
	JB   openSSETail128LoopA

	CMPQ itr2, $160
	JNE  openSSETail128LoopB

	PADDL ·chacha20Constants<>(SB), A0; PADDL ·chacha20Constants<>(SB), A1
	PADDL state1Store, B0; PADDL state1Store, B1
	PADDL state2Store, C0; PADDL state2Store, C1
	PADDL ctr1Store, D0; PADDL ctr0Store, D1

	MOVOU (0*16)(inp), T0; MOVOU (1*16)(inp), T1; MOVOU (2*16)(inp), T2; MOVOU (3*16)(inp), T3
	PXOR  T0, A1; PXOR T1, B1; PXOR T2, C1; PXOR T3, D1
	MOVOU A1, (0*16)(oup); MOVOU B1, (1*16)(oup); MOVOU C1, (2*16)(oup); MOVOU D1, (3*16)(oup)

	SUBQ $64, inl
	LEAQ 64(inp), inp
	LEAQ 64(oup), oup
	JMP  openSSETail64DecLoop

// ----------------------------------------------------------------------------
// Special optimization for the last 192 bytes of ciphertext
openSSETail192:
	// Need to decrypt up to 192 bytes - prepare three blocks
	MOVO ·chacha20Constants<>(SB), A2; MOVO state1Store, B2; MOVO state2Store, C2; MOVO ctr3Store, D2; PADDL ·sseIncMask<>(SB), D2; MOVO D2, ctr0Store
	MOVO A2, A1; MOVO B2, B1; MOVO C2, C1; MOVO D2, D1; PADDL ·sseIncMask<>(SB), D1; MOVO D1, ctr1Store
	MOVO A1, A0; MOVO B1, B0; MOVO C1, C0; MOVO D1, D0; PADDL ·sseIncMask<>(SB), D0; MOVO D0, ctr2Store

	MOVQ    inl, itr1
	MOVQ    $160, itr2
	CMPQ    itr1, $160
	CMOVQGT itr2, itr1
	ANDQ    $-16, itr1
	XORQ    itr2, itr2

openSSLTail192LoopA:
	// Perform ChaCha rounds, while hashing the remaining input
	polyAdd(0(inp)(itr2*1))
	polyMul

openSSLTail192LoopB:
	ADDQ         $16, itr2
	chachaQR(A0, B0, C0, D0, T0); chachaQR(A1, B1, C1, D1, T0); chachaQR(A2, B2, C2, D2, T0)
	shiftB0Left; shiftC0Left; shiftD0Left
	shiftB1Left; shiftC1Left; shiftD1Left
	shiftB2Left; shiftC2Left; shiftD2Left

	chachaQR(A0, B0, C0, D0, T0); chachaQR(A1, B1, C1, D1, T0); chachaQR(A2, B2, C2, D2, T0)
	shiftB0Right; shiftC0Right; shiftD0Right
	shiftB1Right; shiftC1Right; shiftD1Right
	shiftB2Right; shiftC2Right; shiftD2Right

	CMPQ itr2, itr1
	JB   openSSLTail192LoopA

	CMPQ itr2, $160
	JNE  openSSLTail192LoopB

	CMPQ inl, $176
	JB   openSSLTail192Store

	polyAdd(160(inp))
	polyMul

	CMPQ inl, $192
	JB   openSSLTail192Store

	polyAdd(176(inp))
	polyMul

openSSLTail192Store:
	PADDL ·chacha20Constants<>(SB), A0; PADDL ·chacha20Constants<>(SB), A1; PADDL ·chacha20Constants<>(SB), A2
	PADDL state1Store, B0; PADDL state1Store, B1; PADDL state1Store, B2
	PADDL state2Store, C0; PADDL state2Store, C1; PADDL state2Store, C2
	PADDL ctr2Store, D0; PADDL ctr1Store, D1; PADDL ctr0Store, D2

	MOVOU (0*16)(inp), T0; MOVOU (1*16)(inp), T1; MOVOU (2*16)(inp), T2; MOVOU (3*16)(inp), T3
	PXOR  T0, A2; PXOR T1, B2; PXOR T2, C2; PXOR T3, D2
	MOVOU A2, (0*16)(oup); MOVOU B2, (1*16)(oup); MOVOU C2, (2*16)(oup); MOVOU D2, (3*16)(oup)

	MOVOU (4*16)(inp), T0; MOVOU (5*16)(inp), T1; MOVOU (6*16)(inp), T2; MOVOU (7*16)(inp), T3
	PXOR  T0, A1; PXOR T1, B1; PXOR T2, C1; PXOR T3, D1
	MOVOU A1, (4*16)(oup); MOVOU B1, (5*16)(oup); MOVOU C1, (6*16)(oup); MOVOU D1, (7*16)(oup)

	SUBQ $128, inl
	LEAQ 128(inp), inp
	LEAQ 128(oup), oup
	JMP  openSSETail64DecLoop

// ----------------------------------------------------------------------------
// Special optimization for the last 256 bytes of ciphertext
openSSETail256:
	// Need to decrypt up to 256 bytes - prepare four blocks
	MOVO ·chacha20Constants<>(SB), A0; MOVO state1Store, B0; MOVO state2Store, C0; MOVO ctr3Store, D0; PADDL ·sseIncMask<>(SB), D0
	MOVO A0, A1; MOVO B0, B1; MOVO C0, C1; MOVO D0, D1; PADDL ·sseIncMask<>(SB), D1
	MOVO A1, A2; MOVO B1, B2; MOVO C1, C2; MOVO D1, D2; PADDL ·sseIncMask<>(SB), D2
	MOVO A2, A3; MOVO B2, B3; MOVO C2, C3; MOVO D2, D3; PADDL ·sseIncMask<>(SB), D3

	// Store counters
	MOVO D0, ctr0Store; MOVO D1, ctr1Store; MOVO D2, ctr2Store; MOVO D3, ctr3Store
	XORQ itr2, itr2

openSSETail256Loop:
	// This loop inteleaves 8 ChaCha quarter rounds with 1 poly multiplication
	polyAdd(0(inp)(itr2*1))
	MOVO          C3, tmpStore
	chachaQR(A0, B0, C0, D0, C3); chachaQR(A1, B1, C1, D1, C3); chachaQR(A2, B2, C2, D2, C3)
	MOVO          tmpStore, C3
	MOVO          C1, tmpStore
	chachaQR(A3, B3, C3, D3, C1)
	MOVO          tmpStore, C1
	shiftB0Left;  shiftB1Left; shiftB2Left; shiftB3Left
	shiftC0Left;  shiftC1Left; shiftC2Left; shiftC3Left
	shiftD0Left;  shiftD1Left; shiftD2Left; shiftD3Left
	polyMulStage1
	polyMulStage2
	MOVO          C3, tmpStore
	chachaQR(A0, B0, C0, D0, C3); chachaQR(A1, B1, C1, D1, C3); chachaQR(A2, B2, C2, D2, C3)
	MOVO          tmpStore, C3
	MOVO          C1, tmpStore
	chachaQR(A3, B3, C3, D3, C1)
	MOVO          tmpStore, C1
	polyMulStage3
	polyMulReduceStage
	shiftB0Right; shiftB1Right; shiftB2Right; shiftB3Right
	shiftC0Right; shiftC1Right; shiftC2Right; shiftC3Right
	shiftD0Right; shiftD1Right; shiftD2Right; shiftD3Right
	ADDQ          $2*8, itr2
	CMPQ          itr2, $160
	JB            openSSETail256Loop
	MOVQ          inl, itr1
	ANDQ          $-16, itr1

openSSETail256HashLoop:
	polyAdd(0(inp)(itr2*1))
	polyMul
	ADDQ $2*8, itr2
	CMPQ itr2, itr1
	JB   openSSETail256HashLoop

	// Add in the state
	PADDD ·chacha20Constants<>(SB), A0; PADDD ·chacha20Constants<>(SB), A1; PADDD ·chacha20Constants<>(SB), A2; PADDD ·chacha20Constants<>(SB), A3
	PADDD state1Store, B0; PADDD state1Store, B1; PADDD state1Store, B2; PADDD state1Store, B3
	PADDD state2Store, C0; PADDD state2Store, C1; PADDD state2Store, C2; PADDD state2Store, C3
	PADDD ctr0Store, D0; PADDD ctr1Store, D1; PADDD ctr2Store, D2; PADDD ctr3Store, D3
	MOVO  D3, tmpStore

	// Load - xor - store
	MOVOU (0*16)(inp), D3; PXOR D3, A0
	MOVOU (1*16)(inp), D3; PXOR D3, B0
	MOVOU (2*16)(inp), D3; PXOR D3, C0
	MOVOU (3*16)(inp), D3; PXOR D3, D0
	MOVOU A0, (0*16)(oup)
	MOVOU B0, (1*16)(oup)
	MOVOU C0, (2*16)(oup)
	MOVOU D0, (3*16)(oup)
	MOVOU (4*16)(inp), A0; MOVOU (5*16)(inp), B0; MOVOU (6*16)(inp), C0; MOVOU (7*16)(inp), D0
	PXOR  A0, A1; PXOR B0, B1; PXOR C0, C1; PXOR D0, D1
	MOVOU A1, (4*16)(oup); MOVOU B1, (5*16)(oup); MOVOU C1, (6*16)(oup); MOVOU D1, (7*16)(oup)
	MOVOU (8*16)(inp), A0; MOVOU (9*16)(inp), B0; MOVOU (10*16)(inp), C0; MOVOU (11*16)(inp), D0
	PXOR  A0, A2; PXOR B0, B2; PXOR C0, C2; PXOR D0, D2
	MOVOU A2, (8*16)(oup); MOVOU B2, (9*16)(oup); MOVOU C2, (10*16)(oup); MOVOU D2, (11*16)(oup)
	LEAQ  192(inp), inp
	LEAQ  192(oup), oup
	SUBQ  $192, inl
	MOVO  A3, A0
	MOVO  B3, B0
	MOVO  C3, C0
	MOVO  tmpStore, D0

	JMP openSSETail64DecLoop

// ----------------------------------------------------------------------------
// ------------------------- AVX2 Code ----------------------------------------
chacha20Poly1305Open_AVX2:
	VZEROUPPER
	VMOVDQU ·chacha20Constants<>(SB), AA0
	BYTE    $0xc4; BYTE $0x42; BYTE $0x7d; BYTE $0x5a; BYTE $0x70; BYTE $0x10 // broadcasti128 16(r8), ymm14
	BYTE    $0xc4; BYTE $0x42; BYTE $0x7d; BYTE $0x5a; BYTE $0x60; BYTE $0x20 // broadcasti128 32(r8), ymm12
	BYTE    $0xc4; BYTE $0xc2; BYTE $0x7d; BYTE $0x5a; BYTE $0x60; BYTE $0x30 // broadcasti128 48(r8), ymm4
	VPADDD  ·avx2InitMask<>(SB), DD0, DD0

	// Special optimization, for very short buffers
	CMPQ inl, $192
	JBE  openAVX2192
	CMPQ inl, $320
	JBE  openAVX2320

	// For the general key prepare the key first - as a byproduct we have 64 bytes of cipher stream
	VMOVDQA BB0, state1StoreAVX2
	VMOVDQA CC0, state2StoreAVX2
	VMOVDQA DD0, ctr3StoreAVX2
	MOVQ    $10, itr2

openAVX2PreparePolyKey:
	chachaQR_AVX2(AA0, BB0, CC0, DD0, TT0)
	VPALIGNR $4, BB0, BB0, BB0; VPALIGNR $8, CC0, CC0, CC0; VPALIGNR $12, DD0, DD0, DD0
	chachaQR_AVX2(AA0, BB0, CC0, DD0, TT0)
	VPALIGNR $12, BB0, BB0, BB0; VPALIGNR $8, CC0, CC0, CC0; VPALIGNR $4, DD0, DD0, DD0
	DECQ     itr2
	JNE      openAVX2PreparePolyKey

	VPADDD ·chacha20Constants<>(SB), AA0, AA0
	VPADDD state1StoreAVX2, BB0, BB0
	VPADDD state2StoreAVX2, CC0, CC0
	VPADDD ctr3StoreAVX2, DD0, DD0

	VPERM2I128 $0x02, AA0, BB0, TT0

	// Clamp and store poly key
	VPAND   ·polyClampMask<>(SB), TT0, TT0
	VMOVDQA TT0, rsStoreAVX2

	// Stream for the first 64 bytes
	VPERM2I128 $0x13, AA0, BB0, AA0
	VPERM2I128 $0x13, CC0, DD0, BB0

	// Hash AD + first 64 bytes
	MOVQ ad_len+80(FP), itr2
	CALL polyHashADInternal<>(SB)
	XORQ itr1, itr1

openAVX2InitialHash64:
	polyAdd(0(inp)(itr1*1))
	polyMulAVX2
	ADDQ $16, itr1
	CMPQ itr1, $64
	JNE  openAVX2InitialHash64

	// Decrypt the first 64 bytes
	VPXOR   (0*32)(inp), AA0, AA0
	VPXOR   (1*32)(inp), BB0, BB0
	VMOVDQU AA0, (0*32)(oup)
	VMOVDQU BB0, (1*32)(oup)
	LEAQ    (2*32)(inp), inp
	LEAQ    (2*32)(oup), oup
	SUBQ    $64, inl

openAVX2MainLoop:
	CMPQ inl, $512
	JB   openAVX2MainLoopDone

	// Load state, increment counter blocks, store the incremented counters
	VMOVDQU ·chacha20Constants<>(SB), AA0; VMOVDQA AA0, AA1; VMOVDQA AA0, AA2; VMOVDQA AA0, AA3
	VMOVDQA state1StoreAVX2, BB0; VMOVDQA BB0, BB1; VMOVDQA BB0, BB2; VMOVDQA BB0, BB3
	VMOVDQA state2StoreAVX2, CC0; VMOVDQA CC0, CC1; VMOVDQA CC0, CC2; VMOVDQA CC0, CC3
	VMOVDQA ctr3StoreAVX2, DD0; VPADDD ·avx2IncMask<>(SB), DD0, DD0; VPADDD ·avx2IncMask<>(SB), DD0, DD1; VPADDD ·avx2IncMask<>(SB), DD1, DD2; VPADDD ·avx2IncMask<>(SB), DD2, DD3
	VMOVDQA DD0, ctr0StoreAVX2; VMOVDQA DD1, ctr1StoreAVX2; VMOVDQA DD2, ctr2StoreAVX2; VMOVDQA DD3, ctr3StoreAVX2
	XORQ    itr1, itr1

openAVX2InternalLoop:
	// Lets just say this spaghetti loop interleaves 2 quarter rounds with 3 poly multiplications
	// Effectively per 512 bytes of stream we hash 480 bytes of ciphertext
	polyAdd(0*8(inp)(itr1*1))
	VPADDD   BB0, AA0, AA0; VPADDD BB1, AA1, AA1; VPADDD BB2, AA2, AA2; VPADDD BB3, AA3, AA3
	polyMulStage1_AVX2
	VPXOR    AA0, DD0, DD0; VPXOR AA1, DD1, DD1; VPXOR AA2, DD2, DD2; VPXOR AA3, DD3, DD3
	VPSHUFB  ·rol16<>(SB), DD0, DD0; VPSHUFB ·rol16<>(SB), DD1, DD1; VPSHUFB ·rol16<>(SB), DD2, DD2; VPSHUFB ·rol16<>(SB), DD3, DD3
	polyMulStage2_AVX2
	VPADDD   DD0, CC0, CC0; VPADDD DD1, CC1, CC1; VPADDD DD2, CC2, CC2; VPADDD DD3, CC3, CC3
	VPXOR    CC0, BB0, BB0; VPXOR CC1, BB1, BB1; VPXOR CC2, BB2, BB2; VPXOR CC3, BB3, BB3
	polyMulStage3_AVX2
	VMOVDQA  CC3, tmpStoreAVX2
	VPSLLD   $12, BB0, CC3; VPSRLD $20, BB0, BB0; VPXOR CC3, BB0, BB0
	VPSLLD   $12, BB1, CC3; VPSRLD $20, BB1, BB1; VPXOR CC3, BB1, BB1
	VPSLLD   $12, BB2, CC3; VPSRLD $20, BB2, BB2; VPXOR CC3, BB2, BB2
	VPSLLD   $12, BB3, CC3; VPSRLD $20, BB3, BB3; VPXOR CC3, BB3, BB3
	VMOVDQA  tmpStoreAVX2, CC3
	polyMulReduceStage
	VPADDD   BB0, AA0, AA0; VPADDD BB1, AA1, AA1; VPADDD BB2, AA2, AA2; VPADDD BB3, AA3, AA3
	VPXOR    AA0, DD0, DD0; VPXOR AA1, DD1, DD1; VPXOR AA2, DD2, DD2; VPXOR AA3, DD3, DD3
	VPSHUFB  ·rol8<>(SB), DD0, DD0; VPSHUFB ·rol8<>(SB), DD1, DD1; VPSHUFB ·rol8<>(SB), DD2, DD2; VPSHUFB ·rol8<>(SB), DD3, DD3
	polyAdd(2*8(inp)(itr1*1))
	VPADDD   DD0, CC0, CC0; VPADDD DD1, CC1, CC1; VPADDD DD2, CC2, CC2; VPADDD DD3, CC3, CC3
	polyMulStage1_AVX2
	VPXOR    CC0, BB0, BB0; VPXOR CC1, BB1, BB1; VPXOR CC2, BB2, BB2; VPXOR CC3, BB3, BB3
	VMOVDQA  CC3, tmpStoreAVX2
	VPSLLD   $7, BB0, CC3; VPSRLD $25, BB0, BB0; VPXOR CC3, BB0, BB0
	VPSLLD   $7, BB1, CC3; VPSRLD $25, BB1, BB1; VPXOR CC3, BB1, BB1
	VPSLLD   $7, BB2, CC3; VPSRLD $25, BB2, BB2; VPXOR CC3, BB2, BB2
	VPSLLD   $7, BB3, CC3; VPSRLD $25, BB3, BB3; VPXOR CC3, BB3, BB3
	VMOVDQA  tmpStoreAVX2, CC3
	polyMulStage2_AVX2
	VPALIGNR $4, BB0, BB0, BB0; VPALIGNR $4, BB1, BB1, BB1; VPALIGNR $4, BB2, BB2, BB2; VPALIGNR $4, BB3, BB3, BB3
	VPALIGNR $8, CC0, CC0, CC0; VPALIGNR $8, CC1, CC1, CC1; VPALIGNR $8, CC2, CC2, CC2; VPALIGNR $8, CC3, CC3, CC3
	VPALIGNR $12, DD0, DD0, DD0; VPALIGNR $12, DD1, DD1, DD1; VPALIGNR $12, DD2, DD2, DD2; VPALIGNR $12, DD3, DD3, DD3
	VPADDD   BB0, AA0, AA0; VPADDD BB1, AA1, AA1; VPADDD BB2, AA2, AA2; VPADDD BB3, AA3, AA3
	polyMulStage3_AVX2
	VPXOR    AA0, DD0, DD0; VPXOR AA1, DD1, DD1; VPXOR AA2, DD2, DD2; VPXOR AA3, DD3, DD3
	VPSHUFB  ·rol16<>(SB), DD0, DD0; VPSHUFB ·rol16<>(SB), DD1, DD1; VPSHUFB ·rol16<>(SB), DD2, DD2; VPSHUFB ·rol16<>(SB), DD3, DD3
	polyMulReduceStage
	VPADDD   DD0, CC0, CC0; VPADDD DD1, CC1, CC1; VPADDD DD2, CC2, CC2; VPADDD DD3, CC3, CC3
	VPXOR    CC0, BB0, BB0; VPXOR CC1, BB1, BB1; VPXOR CC2, BB2, BB2; VPXOR CC3, BB3, BB3
	polyAdd(4*8(inp)(itr1*1))
	LEAQ     (6*8)(itr1), itr1
	VMOVDQA  CC3, tmpStoreAVX2
	VPSLLD   $12, BB0, CC3; VPSRLD $20, BB0, BB0; VPXOR CC3, BB0, BB0
	VPSLLD   $12, BB1, CC3; VPSRLD $20, BB1, BB1; VPXOR CC3, BB1, BB1
	VPSLLD   $12, BB2, CC3; VPSRLD $20, BB2, BB2; VPXOR CC3, BB2, BB2
	VPSLLD   $12, BB3, CC3; VPSRLD $20, BB3, BB3; VPXOR CC3, BB3, BB3
	VMOVDQA  tmpStoreAVX2, CC3
	polyMulStage1_AVX2
	VPADDD   BB0, AA0, AA0; VPADDD BB1, AA1, AA1; VPADDD BB2, AA2, AA2; VPADDD BB3, AA3, AA3
	VPXOR    AA0, DD0, DD0; VPXOR AA1, DD1, DD1; VPXOR AA2, DD2, DD2; VPXOR AA3, DD3, DD3
	polyMulStage2_AVX2
	VPSHUFB  ·rol8<>(SB), DD0, DD0; VPSHUFB ·rol8<>(SB), DD1, DD1; VPSHUFB ·rol8<>(SB), DD2, DD2; VPSHUFB ·rol8<>(SB), DD3, DD3
	VPADDD   DD0, CC0, CC0; VPADDD DD1, CC1, CC1; VPADDD DD2, CC2, CC2; VPADDD DD3, CC3, CC3
	polyMulStage3_AVX2
	VPXOR    CC0, BB0, BB0; VPXOR CC1, BB1, BB1; VPXOR CC2, BB2, BB2; VPXOR CC3, BB3, BB3
	VMOVDQA  CC3, tmpStoreAVX2
	VPSLLD   $7, BB0, CC3; VPSRLD $25, BB0, BB0; VPXOR CC3, BB0, BB0
	VPSLLD   $7, BB1, CC3; VPSRLD $25, BB1, BB1; VPXOR CC3, BB1, BB1
	VPSLLD   $7, BB2, CC3; VPSRLD $25, BB2, BB2; VPXOR CC3, BB2, BB2
	VPSLLD   $7, BB3, CC3; VPSRLD $25, BB3, BB3; VPXOR CC3, BB3, BB3
	VMOVDQA  tmpStoreAVX2, CC3
	polyMulReduceStage
	VPALIGNR $12, BB0, BB0, BB0; VPALIGNR $12, BB1, BB1, BB1; VPALIGNR $12, BB2, BB2, BB2; VPALIGNR $12, BB3, BB3, BB3
	VPALIGNR $8, CC0, CC0, CC0; VPALIGNR $8, CC1, CC1, CC1; VPALIGNR $8, CC2, CC2, CC2; VPALIGNR $8, CC3, CC3, CC3
	VPALIGNR $4, DD0, DD0, DD0; VPALIGNR $4, DD1, DD1, DD1; VPALIGNR $4, DD2, DD2, DD2; VPALIGNR $4, DD3, DD3, DD3
	CMPQ     itr1, $480
	JNE      openAVX2InternalLoop

	VPADDD  ·chacha20Constants<>(SB), AA0, AA0; VPADDD ·chacha20Constants<>(SB), AA1, AA1; VPADDD ·chacha20Constants<>(SB), AA2, AA2; VPADDD ·chacha20Constants<>(SB), AA3, AA3
	VPADDD  state1StoreAVX2, BB0, BB0; VPADDD state1StoreAVX2, BB1, BB1; VPADDD state1StoreAVX2, BB2, BB2; VPADDD state1StoreAVX2, BB3, BB3
	VPADDD  state2StoreAVX2, CC0, CC0; VPADDD state2StoreAVX2, CC1, CC1; VPADDD state2StoreAVX2, CC2, CC2; VPADDD state2StoreAVX2, CC3, CC3
	VPADDD  ctr0StoreAVX2, DD0, DD0; VPADDD ctr1StoreAVX2, DD1, DD1; VPADDD ctr2StoreAVX2, DD2, DD2; VPADDD ctr3StoreAVX2, DD3, DD3
	VMOVDQA CC3, tmpStoreAVX2

	// We only hashed 480 of the 512 bytes available - hash the remaining 32 here
	polyAdd(480(inp))
	polyMulAVX2
	VPERM2I128 $0x02, AA0, BB0, CC3; VPERM2I128 $0x13, AA0, BB0, BB0; VPERM2I128 $0x02, CC0, DD0, AA0; VPERM2I128 $0x13, CC0, DD0, CC0
	VPXOR      (0*32)(inp), CC3, CC3; VPXOR (1*32)(inp), AA0, AA0; VPXOR (2*32)(inp), BB0, BB0; VPXOR (3*32)(inp), CC0, CC0
	VMOVDQU    CC3, (0*32)(oup); VMOVDQU AA0, (1*32)(oup); VMOVDQU BB0, (2*32)(oup); VMOVDQU CC0, (3*32)(oup)
	VPERM2I128 $0x02, AA1, BB1, AA0; VPERM2I128 $0x02, CC1, DD1, BB0; VPERM2I128 $0x13, AA1, BB1, CC0; VPERM2I128 $0x13, CC1, DD1, DD0
	VPXOR      (4*32)(inp), AA0, AA0; VPXOR (5*32)(inp), BB0, BB0; VPXOR (6*32)(inp), CC0, CC0; VPXOR (7*32)(inp), DD0, DD0
	VMOVDQU    AA0, (4*32)(oup); VMOVDQU BB0, (5*32)(oup); VMOVDQU CC0, (6*32)(oup); VMOVDQU DD0, (7*32)(oup)

	// and here
	polyAdd(496(inp))
	polyMulAVX2
	VPERM2I128 $0x02, AA2, BB2, AA0; VPERM2I128 $0x02, CC2, DD2, BB0; VPERM2I128 $0x13, AA2, BB2, CC0; VPERM2I128 $0x13, CC2, DD2, DD0
	VPXOR      (8*32)(inp), AA0, AA0; VPXOR (9*32)(inp), BB0, BB0; VPXOR (10*32)(inp), CC0, CC0; VPXOR (11*32)(inp), DD0, DD0
	VMOVDQU    AA0, (8*32)(oup); VMOVDQU BB0, (9*32)(oup); VMOVDQU CC0, (10*32)(oup); VMOVDQU DD0, (11*32)(oup)
	VPERM2I128 $0x02, AA3, BB3, AA0; VPERM2I128 $0x02, tmpStoreAVX2, DD3, BB0; VPERM2I128 $0x13, AA3, BB3, CC0; VPERM2I128 $0x13, tmpStoreAVX2, DD3, DD0
	VPXOR      (12*32)(inp), AA0, AA0; VPXOR (13*32)(inp), BB0, BB0; VPXOR (14*32)(inp), CC0, CC0; VPXOR (15*32)(inp), DD0, DD0
	VMOVDQU    AA0, (12*32)(oup); VMOVDQU BB0, (13*32)(oup); VMOVDQU CC0, (14*32)(oup); VMOVDQU DD0, (15*32)(oup)
	LEAQ       (32*16)(inp), inp
	LEAQ       (32*16)(oup), oup
	SUBQ       $(32*16), inl
	JMP        openAVX2MainLoop

openAVX2MainLoopDone:
	// Handle the various tail sizes efficiently
	TESTQ inl, inl
	JE    openSSEFinalize
	CMPQ  inl, $128
	JBE   openAVX2Tail128
	CMPQ  inl, $256
	JBE   openAVX2Tail256
	CMPQ  inl, $384
	JBE   openAVX2Tail384
	JMP   openAVX2Tail512

// ----------------------------------------------------------------------------
// Special optimization for buffers smaller than 193 bytes
openAVX2192:
	// For up to 192 bytes of ciphertext and 64 bytes for the poly key, we process four blocks
	VMOVDQA AA0, AA1
	VMOVDQA BB0, BB1
	VMOVDQA CC0, CC1
	VPADDD  ·avx2IncMask<>(SB), DD0, DD1
	VMOVDQA AA0, AA2
	VMOVDQA BB0, BB2
	VMOVDQA CC0, CC2
	VMOVDQA DD0, DD2
	VMOVDQA DD1, TT3
	MOVQ    $10, itr2

openAVX2192InnerCipherLoop:
	chachaQR_AVX2(AA0, BB0, CC0, DD0, TT0); chachaQR_AVX2(AA1, BB1, CC1, DD1, TT0)
	VPALIGNR   $4, BB0, BB0, BB0; VPALIGNR $4, BB1, BB1, BB1
	VPALIGNR   $8, CC0, CC0, CC0; VPALIGNR $8, CC1, CC1, CC1
	VPALIGNR   $12, DD0, DD0, DD0; VPALIGNR $12, DD1, DD1, DD1
	chachaQR_AVX2(AA0, BB0, CC0, DD0, TT0); chachaQR_AVX2(AA1, BB1, CC1, DD1, TT0)
	VPALIGNR   $12, BB0, BB0, BB0; VPALIGNR $12, BB1, BB1, BB1
	VPALIGNR   $8, CC0, CC0, CC0; VPALIGNR $8, CC1, CC1, CC1
	VPALIGNR   $4, DD0, DD0, DD0; VPALIGNR $4, DD1, DD1, DD1
	DECQ       itr2
	JNE        openAVX2192InnerCipherLoop
	VPADDD     AA2, AA0, AA0; VPADDD AA2, AA1, AA1
	VPADDD     BB2, BB0, BB0; VPADDD BB2, BB1, BB1
	VPADDD     CC2, CC0, CC0; VPADDD CC2, CC1, CC1
	VPADDD     DD2, DD0, DD0; VPADDD TT3, DD1, DD1
	VPERM2I128 $0x02, AA0, BB0, TT0

	// Clamp and store poly key
	VPAND   ·polyClampMask<>(SB), TT0, TT0
	VMOVDQA TT0, rsStoreAVX2

	// Stream for up to 192 bytes
	VPERM2I128 $0x13, AA0, BB0, AA0
	VPERM2I128 $0x13, CC0, DD0, BB0
	VPERM2I128 $0x02, AA1, BB1, CC0
	VPERM2I128 $0x02, CC1, DD1, DD0
	VPERM2I128 $0x13, AA1, BB1, AA1
	VPERM2I128 $0x13, CC1, DD1, BB1

openAVX2ShortOpen:
	// Hash
	MOVQ ad_len+80(FP), itr2
	CALL polyHashADInternal<>(SB)

openAVX2ShortOpenLoop:
	CMPQ inl, $32
	JB   openAVX2ShortTail32
	SUBQ $32, inl

	// Load for hashing
	polyAdd(0*8(inp))
	polyMulAVX2
	polyAdd(2*8(inp))
	polyMulAVX2

	// Load for decryption
	VPXOR   (inp), AA0, AA0
	VMOVDQU AA0, (oup)
	LEAQ    (1*32)(inp), inp
	LEAQ    (1*32)(oup), oup

	// Shift stream left
	VMOVDQA BB0, AA0
	VMOVDQA CC0, BB0
	VMOVDQA DD0, CC0
	VMOVDQA AA1, DD0
	VMOVDQA BB1, AA1
	VMOVDQA CC1, BB1
	VMOVDQA DD1, CC1
	VMOVDQA AA2, DD1
	VMOVDQA BB2, AA2
	JMP     openAVX2ShortOpenLoop

openAVX2ShortTail32:
	CMPQ    inl, $16
	VMOVDQA A0, A1
	JB      openAVX2ShortDone

	SUBQ $16, inl

	// Load for hashing
	polyAdd(0*8(inp))
	polyMulAVX2

	// Load for decryption
	VPXOR      (inp), A0, T0
	VMOVDQU    T0, (oup)
	LEAQ       (1*16)(inp), inp
	LEAQ       (1*16)(oup), oup
	VPERM2I128 $0x11, AA0, AA0, AA0
	VMOVDQA    A0, A1

openAVX2ShortDone:
	VZEROUPPER
	JMP openSSETail16

// ----------------------------------------------------------------------------
// Special optimization for buffers smaller than 321 bytes
openAVX2320:
	// For up to 320 bytes of ciphertext and 64 bytes for the poly key, we process six blocks
	VMOVDQA AA0, AA1; VMOVDQA BB0, BB1; VMOVDQA CC0, CC1; VPADDD ·avx2IncMask<>(SB), DD0, DD1
	VMOVDQA AA0, AA2; VMOVDQA BB0, BB2; VMOVDQA CC0, CC2; VPADDD ·avx2IncMask<>(SB), DD1, DD2
	VMOVDQA BB0, TT1; VMOVDQA CC0, TT2; VMOVDQA DD0, TT3
	MOVQ    $10, itr2

openAVX2320InnerCipherLoop:
	chachaQR_AVX2(AA0, BB0, CC0, DD0, TT0); chachaQR_AVX2(AA1, BB1, CC1, DD1, TT0); chachaQR_AVX2(AA2, BB2, CC2, DD2, TT0)
	VPALIGNR $4, BB0, BB0, BB0; VPALIGNR $4, BB1, BB1, BB1; VPALIGNR $4, BB2, BB2, BB2
	VPALIGNR $8, CC0, CC0, CC0; VPALIGNR $8, CC1, CC1, CC1; VPALIGNR $8, CC2, CC2, CC2
	VPALIGNR $12, DD0, DD0, DD0; VPALIGNR $12, DD1, DD1, DD1; VPALIGNR $12, DD2, DD2, DD2
	chachaQR_AVX2(AA0, BB0, CC0, DD0, TT0); chachaQR_AVX2(AA1, BB1, CC1, DD1, TT0); chachaQR_AVX2(AA2, BB2, CC2, DD2, TT0)
	VPALIGNR $12, BB0, BB0, BB0; VPALIGNR $12, BB1, BB1, BB1; VPALIGNR $12, BB2, BB2, BB2
	VPALIGNR $8, CC0, CC0, CC0; VPALIGNR $8, CC1, CC1, CC1; VPALIGNR $8, CC2, CC2, CC2
	VPALIGNR $4, DD0, DD0, DD0; VPALIGNR $4, DD1, DD1, DD1; VPALIGNR $4, DD2, DD2, DD2
	DECQ     itr2
	JNE      openAVX2320InnerCipherLoop

	VMOVDQA ·chacha20Constants<>(SB), TT0
	VPADDD  TT0, AA0, AA0; VPADDD TT0, AA1, AA1; VPADDD TT0, AA2, AA2
	VPADDD  TT1, BB0, BB0; VPADDD TT1, BB1, BB1; VPADDD TT1, BB2, BB2
	VPADDD  TT2, CC0, CC0; VPADDD TT2, CC1, CC1; VPADDD TT2, CC2, CC2
	VMOVDQA ·avx2IncMask<>(SB), TT0
	VPADDD  TT3, DD0, DD0; VPADDD TT0, TT3, TT3
	VPADDD  TT3, DD1, DD1; VPADDD TT0, TT3, TT3
	VPADDD  TT3, DD2, DD2

	// Clamp and store poly key
	VPERM2I128 $0x02, AA0, BB0, TT0
	VPAND      ·polyClampMask<>(SB), TT0, TT0
	VMOVDQA    TT0, rsStoreAVX2

	// Stream for up to 320 bytes
	VPERM2I128 $0x13, AA0, BB0, AA0
	VPERM2I128 $0x13, CC0, DD0, BB0
	VPERM2I128 $0x02, AA1, BB1, CC0
	VPERM2I128 $0x02, CC1, DD1, DD0
	VPERM2I128 $0x13, AA1, BB1, AA1
	VPERM2I128 $0x13, CC1, DD1, BB1
	VPERM2I128 $0x02, AA2, BB2, CC1
	VPERM2I128 $0x02, CC2, DD2, DD1
	VPERM2I128 $0x13, AA2, BB2, AA2
	VPERM2I128 $0x13, CC2, DD2, BB2
	JMP        openAVX2ShortOpen

// ----------------------------------------------------------------------------
// Special optimization for the last 128 bytes of ciphertext
openAVX2Tail128:
	// Need to decrypt up to 128 bytes - prepare two blocks
	VMOVDQA ·chacha20Constants<>(SB), AA1
	VMOVDQA state1StoreAVX2, BB1
	VMOVDQA state2StoreAVX2, CC1
	VMOVDQA ctr3StoreAVX2, DD1
	VPADDD  ·avx2IncMask<>(SB), DD1, DD1
	VMOVDQA DD1, DD0

	XORQ  itr2, itr2
	MOVQ  inl, itr1
	ANDQ  $-16, itr1
	TESTQ itr1, itr1
	JE    openAVX2Tail128LoopB

openAVX2Tail128LoopA:
	// Perform ChaCha rounds, while hashing the remaining input
	polyAdd(0(inp)(itr2*1))
	polyMulAVX2

openAVX2Tail128LoopB:
	ADDQ     $16, itr2
	chachaQR_AVX2(AA1, BB1, CC1, DD1, TT0)
	VPALIGNR $4, BB1, BB1, BB1
	VPALIGNR $8, CC1, CC1, CC1
	VPALIGNR $12, DD1, DD1, DD1
	chachaQR_AVX2(AA1, BB1, CC1, DD1, TT0)
	VPALIGNR $12, BB1, BB1, BB1
	VPALIGNR $8, CC1, CC1, CC1
	VPALIGNR $4, DD1, DD1, DD1
	CMPQ     itr2, itr1
	JB       openAVX2Tail128LoopA
	CMPQ     itr2, $160
	JNE      openAVX2Tail128LoopB

	VPADDD     ·chacha20Constants<>(SB), AA1, AA1
	VPADDD     state1StoreAVX2, BB1, BB1
	VPADDD     state2StoreAVX2, CC1, CC1
	VPADDD     DD0, DD1, DD1
	VPERM2I128 $0x02, AA1, BB1, AA0; VPERM2I128 $0x02, CC1, DD1, BB0; VPERM2I128 $0x13, AA1, BB1, CC0; VPERM2I128 $0x13, CC1, DD1, DD0

openAVX2TailLoop:
	CMPQ inl, $32
	JB   openAVX2Tail
	SUBQ $32, inl

	// Load for decryption
	VPXOR   (inp), AA0, AA0
	VMOVDQU AA0, (oup)
	LEAQ    (1*32)(inp), inp
	LEAQ    (1*32)(oup), oup
	VMOVDQA BB0, AA0
	VMOVDQA CC0, BB0
	VMOVDQA DD0, CC0
	JMP     openAVX2TailLoop

openAVX2Tail:
	CMPQ    inl, $16
	VMOVDQA A0, A1
	JB      openAVX2TailDone
	SUBQ    $16, inl

	// Load for decryption
	VPXOR      (inp), A0, T0
	VMOVDQU    T0, (oup)
	LEAQ       (1*16)(inp), inp
	LEAQ       (1*16)(oup), oup
	VPERM2I128 $0x11, AA0, AA0, AA0
	VMOVDQA    A0, A1

openAVX2TailDone:
	VZEROUPPER
	JMP openSSETail16

// ----------------------------------------------------------------------------
// Special optimization for the last 256 bytes of ciphertext
openAVX2Tail256:
	// Need to decrypt up to 256 bytes - prepare four blocks
	VMOVDQA ·chacha20Constants<>(SB), AA0; VMOVDQA AA0, AA1
	VMOVDQA state1StoreAVX2, BB0; VMOVDQA BB0, BB1
	VMOVDQA state2StoreAVX2, CC0; VMOVDQA CC0, CC1
	VMOVDQA ctr3StoreAVX2, DD0
	VPADDD  ·avx2IncMask<>(SB), DD0, DD0
	VPADDD  ·avx2IncMask<>(SB), DD0, DD1
	VMOVDQA DD0, TT1
	VMOVDQA DD1, TT2

	// Compute the number of iterations that will hash data
	MOVQ    inl, tmpStoreAVX2
	MOVQ    inl, itr1
	SUBQ    $128, itr1
	SHRQ    $4, itr1
	MOVQ    $10, itr2
	CMPQ    itr1, $10
	CMOVQGT itr2, itr1
	MOVQ    inp, inl
	XORQ    itr2, itr2

openAVX2Tail256LoopA:
	polyAdd(0(inl))
	polyMulAVX2
	LEAQ 16(inl), inl

	// Perform ChaCha rounds, while hashing the remaining input
openAVX2Tail256LoopB:
	chachaQR_AVX2(AA0, BB0, CC0, DD0, TT0); chachaQR_AVX2(AA1, BB1, CC1, DD1, TT0)
	VPALIGNR $4, BB0, BB0, BB0; VPALIGNR $4, BB1, BB1, BB1
	VPALIGNR $8, CC0, CC0, CC0; VPALIGNR $8, CC1, CC1, CC1
	VPALIGNR $12, DD0, DD0, DD0; VPALIGNR $12, DD1, DD1, DD1
	INCQ     itr2
	chachaQR_AVX2(AA0, BB0, CC0, DD0, TT0); chachaQR_AVX2(AA1, BB1, CC1, DD1, TT0)
	VPALIGNR $12, BB0, BB0, BB0; VPALIGNR $12, BB1, BB1, BB1
	VPALIGNR $8, CC0, CC0, CC0; VPALIGNR $8, CC1, CC1, CC1
	VPALIGNR $4, DD0, DD0, DD0; VPALIGNR $4, DD1, DD1, DD1
	CMPQ     itr2, itr1
	JB       openAVX2Tail256LoopA

	CMPQ itr2, $10
	JNE  openAVX2Tail256LoopB

	MOVQ inl, itr2
	SUBQ inp, inl
	MOVQ inl, itr1
	MOVQ tmpStoreAVX2, inl

	// Hash the remainder of data (if any)
openAVX2Tail256Hash:
	ADDQ $16, itr1
	CMPQ itr1, inl
	JGT  openAVX2Tail256HashEnd
	polyAdd (0(itr2))
	polyMulAVX2
	LEAQ 16(itr2), itr2
	JMP  openAVX2Tail256Hash

// Store 128 bytes safely, then go to store loop
openAVX2Tail256HashEnd:
	VPADDD     ·chacha20Constants<>(SB), AA0, AA0; VPADDD ·chacha20Constants<>(SB), AA1, AA1
	VPADDD     state1StoreAVX2, BB0, BB0; VPADDD state1StoreAVX2, BB1, BB1
	VPADDD     state2StoreAVX2, CC0, CC0; VPADDD state2StoreAVX2, CC1, CC1
	VPADDD     TT1, DD0, DD0; VPADDD TT2, DD1, DD1
	VPERM2I128 $0x02, AA0, BB0, AA2; VPERM2I128 $0x02, CC0, DD0, BB2; VPERM2I128 $0x13, AA0, BB0, CC2; VPERM2I128 $0x13, CC0, DD0, DD2
	VPERM2I128 $0x02, AA1, BB1, AA0; VPERM2I128 $0x02, CC1, DD1, BB0; VPERM2I128 $0x13, AA1, BB1, CC0; VPERM2I128 $0x13, CC1, DD1, DD0

	VPXOR   (0*32)(inp), AA2, AA2; VPXOR (1*32)(inp), BB2, BB2; VPXOR (2*32)(inp), CC2, CC2; VPXOR (3*32)(inp), DD2, DD2
	VMOVDQU AA2, (0*32)(oup); VMOVDQU BB2, (1*32)(oup); VMOVDQU CC2, (2*32)(oup); VMOVDQU DD2, (3*32)(oup)
	LEAQ    (4*32)(inp), inp
	LEAQ    (4*32)(oup), oup
	SUBQ    $4*32, inl

	JMP openAVX2TailLoop

// ----------------------------------------------------------------------------
// Special optimization for the last 384 bytes of ciphertext
openAVX2Tail384:
	// Need to decrypt up to 384 bytes - prepare six blocks
	VMOVDQA ·chacha20Constants<>(SB), AA0; VMOVDQA AA0, AA1; VMOVDQA AA0, AA2
	VMOVDQA state1StoreAVX2, BB0; VMOVDQA BB0, BB1; VMOVDQA BB0, BB2
	VMOVDQA state2StoreAVX2, CC0; VMOVDQA CC0, CC1; VMOVDQA CC0, CC2
	VMOVDQA ctr3StoreAVX2, DD0
	VPADDD  ·avx2IncMask<>(SB), DD0, DD0
	VPADDD  ·avx2IncMask<>(SB), DD0, DD1
	VPADDD  ·avx2IncMask<>(SB), DD1, DD2
	VMOVDQA DD0, ctr0StoreAVX2
	VMOVDQA DD1, ctr1StoreAVX2
	VMOVDQA DD2, ctr2StoreAVX2

	// Compute the number of iterations that will hash two blocks of data
	MOVQ    inl, tmpStoreAVX2
	MOVQ    inl, itr1
	SUBQ    $256, itr1
	SHRQ    $4, itr1
	ADDQ    $6, itr1
	MOVQ    $10, itr2
	CMPQ    itr1, $10
	CMOVQGT itr2, itr1
	MOVQ    inp, inl
	XORQ    itr2, itr2

	// Perform ChaCha rounds, while hashing the remaining input
openAVX2Tail384LoopB:
	polyAdd(0(inl))
	polyMulAVX2
	LEAQ 16(inl), inl

openAVX2Tail384LoopA:
	chachaQR_AVX2(AA0, BB0, CC0, DD0, TT0); chachaQR_AVX2(AA1, BB1, CC1, DD1, TT0); chachaQR_AVX2(AA2, BB2, CC2, DD2, TT0)
	VPALIGNR $4, BB0, BB0, BB0; VPALIGNR $4, BB1, BB1, BB1; VPALIGNR $4, BB2, BB2, BB2
	VPALIGNR $8, CC0, CC0, CC0; VPALIGNR $8, CC1, CC1, CC1; VPALIGNR $8, CC2, CC2, CC2
	VPALIGNR $12, DD0, DD0, DD0; VPALIGNR $12, DD1, DD1, DD1; VPALIGNR $12, DD2, DD2, DD2
	polyAdd(0(inl))
	polyMulAVX2
	LEAQ     16(inl), inl
	INCQ     itr2
	chachaQR_AVX2(AA0, BB0, CC0, DD0, TT0); chachaQR_AVX2(AA1, BB1, CC1, DD1, TT0); chachaQR_AVX2(AA2, BB2, CC2, DD2, TT0)
	VPALIGNR $12, BB0, BB0, BB0; VPALIGNR $12, BB1, BB1, BB1; VPALIGNR $12, BB2, BB2, BB2
	VPALIGNR $8, CC0, CC0, CC0; VPALIGNR $8, CC1, CC1, CC1; VPALIGNR $8, CC2, CC2, CC2
	VPALIGNR $4, DD0, DD0, DD0; VPALIGNR $4, DD1, DD1, DD1; VPALIGNR $4, DD2, DD2, DD2

	CMPQ itr2, itr1
	JB   openAVX2Tail384LoopB

	CMPQ itr2, $10
	JNE  openAVX2Tail384LoopA

	MOVQ inl, itr2
	SUBQ inp, inl
	MOVQ inl, itr1
	MOVQ tmpStoreAVX2, inl

openAVX2Tail384Hash:
	ADDQ $16, itr1
	CMPQ itr1, inl
	JGT  openAVX2Tail384HashEnd
	polyAdd(0(itr2))
	polyMulAVX2
	LEAQ 16(itr2), itr2
	JMP  openAVX2Tail384Hash

// Store 256 bytes safely, then go to store loop
openAVX2Tail384HashEnd:
	VPADDD     ·chacha20Constants<>(SB), AA0, AA0; VPADDD ·chacha20Constants<>(SB), AA1, AA1; VPADDD ·chacha20Constants<>(SB), AA2, AA2
	VPADDD     state1StoreAVX2, BB0, BB0; VPADDD state1StoreAVX2, BB1, BB1; VPADDD state1StoreAVX2, BB2, BB2
	VPADDD     state2StoreAVX2, CC0, CC0; VPADDD state2StoreAVX2, CC1, CC1; VPADDD state2StoreAVX2, CC2, CC2
	VPADDD     ctr0StoreAVX2, DD0, DD0; VPADDD ctr1StoreAVX2, DD1, DD1; VPADDD ctr2StoreAVX2, DD2, DD2
	VPERM2I128 $0x02, AA0, BB0, TT0; VPERM2I128 $0x02, CC0, DD0, TT1; VPERM2I128 $0x13, AA0, BB0, TT2; VPERM2I128 $0x13, CC0, DD0, TT3
	VPXOR      (0*32)(inp), TT0, TT0; VPXOR (1*32)(inp), TT1, TT1; VPXOR (2*32)(inp), TT2, TT2; VPXOR (3*32)(inp), TT3, TT3
	VMOVDQU    TT0, (0*32)(oup); VMOVDQU TT1, (1*32)(oup); VMOVDQU TT2, (2*32)(oup); VMOVDQU TT3, (3*32)(oup)
	VPERM2I128 $0x02, AA1, BB1, TT0; VPERM2I128 $0x02, CC1, DD1, TT1; VPERM2I128 $0x13, AA1, BB1, TT2; VPERM2I128 $0x13, CC1, DD1, TT3
	VPXOR      (4*32)(inp), TT0, TT0; VPXOR (5*32)(inp), TT1, TT1; VPXOR (6*32)(inp), TT2, TT2; VPXOR (7*32)(inp), TT3, TT3
	VMOVDQU    TT0, (4*32)(oup); VMOVDQU TT1, (5*32)(oup); VMOVDQU TT2, (6*32)(oup); VMOVDQU TT3, (7*32)(oup)
	VPERM2I128 $0x02, AA2, BB2, AA0; VPERM2I128 $0x02, CC2, DD2, BB0; VPERM2I128 $0x13, AA2, BB2, CC0; VPERM2I128 $0x13, CC2, DD2, DD0
	LEAQ       (8*32)(inp), inp
	LEAQ       (8*32)(oup), oup
	SUBQ       $8*32, inl
	JMP        openAVX2TailLoop

// ----------------------------------------------------------------------------
// Special optimization for the last 512 bytes of ciphertext
openAVX2Tail512:
	VMOVDQU ·chacha20Constants<>(SB), AA0; VMOVDQA AA0, AA1; VMOVDQA AA0, AA2; VMOVDQA AA0, AA3
	VMOVDQA state1StoreAVX2, BB0; VMOVDQA BB0, BB1; VMOVDQA BB0, BB2; VMOVDQA BB0, BB3
	VMOVDQA state2StoreAVX2, CC0; VMOVDQA CC0, CC1; VMOVDQA CC0, CC2; VMOVDQA CC0, CC3
	VMOVDQA ctr3StoreAVX2, DD0; VPADDD ·avx2IncMask<>(SB), DD0, DD0; VPADDD ·avx2IncMask<>(SB), DD0, DD1; VPADDD ·avx2IncMask<>(SB), DD1, DD2; VPADDD ·avx2IncMask<>(SB), DD2, DD3
	VMOVDQA DD0, ctr0StoreAVX2; VMOVDQA DD1, ctr1StoreAVX2; VMOVDQA DD2, ctr2StoreAVX2; VMOVDQA DD3, ctr3StoreAVX2
	XORQ    itr1, itr1
	MOVQ    inp, itr2

openAVX2Tail512LoopB:
	polyAdd(0(itr2))
	polyMulAVX2
	LEAQ (2*8)(itr2), itr2

openAVX2Tail512LoopA:
	VPADDD   BB0, AA0, AA0; VPADDD BB1, AA1, AA1; VPADDD BB2, AA2, AA2; VPADDD BB3, AA3, AA3
	VPXOR    AA0, DD0, DD0; VPXOR AA1, DD1, DD1; VPXOR AA2, DD2, DD2; VPXOR AA3, DD3, DD3
	VPSHUFB  ·rol16<>(SB), DD0, DD0; VPSHUFB ·rol16<>(SB), DD1, DD1; VPSHUFB ·rol16<>(SB), DD2, DD2; VPSHUFB ·rol16<>(SB), DD3, DD3
	VPADDD   DD0, CC0, CC0; VPADDD DD1, CC1, CC1; VPADDD DD2, CC2, CC2; VPADDD DD3, CC3, CC3
	VPXOR    CC0, BB0, BB0; VPXOR CC1, BB1, BB1; VPXOR CC2, BB2, BB2; VPXOR CC3, BB3, BB3
	VMOVDQA  CC3, tmpStoreAVX2
	VPSLLD   $12, BB0, CC3; VPSRLD $20, BB0, BB0; VPXOR CC3, BB0, BB0
	VPSLLD   $12, BB1, CC3; VPSRLD $20, BB1, BB1; VPXOR CC3, BB1, BB1
	VPSLLD   $12, BB2, CC3; VPSRLD $20, BB2, BB2; VPXOR CC3, BB2, BB2
	VPSLLD   $12, BB3, CC3; VPSRLD $20, BB3, BB3; VPXOR CC3, BB3, BB3
	VMOVDQA  tmpStoreAVX2, CC3
	polyAdd(0*8(itr2))
	polyMulAVX2
	VPADDD   BB0, AA0, AA0; VPADDD BB1, AA1, AA1; VPADDD BB2, AA2, AA2; VPADDD BB3, AA3, AA3
	VPXOR    AA0, DD0, DD0; VPXOR AA1, DD1, DD1; VPXOR AA2, DD2, DD2; VPXOR AA3, DD3, DD3
	VPSHUFB  ·rol8<>(SB), DD0, DD0; VPSHUFB ·rol8<>(SB), DD1, DD1; VPSHUFB ·rol8<>(SB), DD2, DD2; VPSHUFB ·rol8<>(SB), DD3, DD3
	VPADDD   DD0, CC0, CC0; VPADDD DD1, CC1, CC1; VPADDD DD2, CC2, CC2; VPADDD DD3, CC3, CC3
	VPXOR    CC0, BB0, BB0; VPXOR CC1, BB1, BB1; VPXOR CC2, BB2, BB2; VPXOR CC3, BB3, BB3
	VMOVDQA  CC3, tmpStoreAVX2
	VPSLLD   $7, BB0, CC3; VPSRLD $25, BB0, BB0; VPXOR CC3, BB0, BB0
	VPSLLD   $7, BB1, CC3; VPSRLD $25, BB1, BB1; VPXOR CC3, BB1, BB1
	VPSLLD   $7, BB2, CC3; VPSRLD $25, BB2, BB2; VPXOR CC3, BB2, BB2
	VPSLLD   $7, BB3, CC3; VPSRLD $25, BB3, BB3; VPXOR CC3, BB3, BB3
	VMOVDQA  tmpStoreAVX2, CC3
	VPALIGNR $4, BB0, BB0, BB0; VPALIGNR $4, BB1, BB1, BB1; VPALIGNR $4, BB2, BB2, BB2; VPALIGNR $4, BB3, BB3, BB3
	VPALIGNR $8, CC0, CC0, CC0; VPALIGNR $8, CC1, CC1, CC1; VPALIGNR $8, CC2, CC2, CC2; VPALIGNR $8, CC3, CC3, CC3
	VPALIGNR $12, DD0, DD0, DD0; VPALIGNR $12, DD1, DD1, DD1; VPALIGNR $12, DD2, DD2, DD2; VPALIGNR $12, DD3, DD3, DD3
	VPADDD   BB0, AA0, AA0; VPADDD BB1, AA1, AA1; VPADDD BB2, AA2, AA2; VPADDD BB3, AA3, AA3
	VPXOR    AA0, DD0, DD0; VPXOR AA1, DD1, DD1; VPXOR AA2, DD2, DD2; VPXOR AA3, DD3, DD3
	VPSHUFB  ·rol16<>(SB), DD0, DD0; VPSHUFB ·rol16<>(SB), DD1, DD1; VPSHUFB ·rol16<>(SB), DD2, DD2; VPSHUFB ·rol16<>(SB), DD3, DD3
	VPADDD   DD0, CC0, CC0; VPADDD DD1, CC1, CC1; VPADDD DD2, CC2, CC2; VPADDD DD3, CC3, CC3
	VPXOR    CC0, BB0, BB0; VPXOR CC1, BB1, BB1; VPXOR CC2, BB2, BB2; VPXOR CC3, BB3, BB3
	polyAdd(2*8(itr2))
	polyMulAVX2
	LEAQ     (4*8)(itr2), itr2
	VMOVDQA  CC3, tmpStoreAVX2
	VPSLLD   $12, BB0, CC3; VPSRLD $20, BB0, BB0; VPXOR CC3, BB0, BB0
	VPSLLD   $12, BB1, CC3; VPSRLD $20, BB1, BB1; VPXOR CC3, BB1, BB1
	VPSLLD   $12, BB2, CC3; VPSRLD $20, BB2, BB2; VPXOR CC3, BB2, BB2
	VPSLLD   $12, BB3, CC3; VPSRLD $20, BB3, BB3; VPXOR CC3, BB3, BB3
	VMOVDQA  tmpStoreAVX2, CC3
	VPADDD   BB0, AA0, AA0; VPADDD BB1, AA1, AA1; VPADDD BB2, AA2, AA2; VPADDD BB3, AA3, AA3
	VPXOR    AA0, DD0, DD0; VPXOR AA1, DD1, DD1; VPXOR AA2, DD2, DD2; VPXOR AA3, DD3, DD3
	VPSHUFB  ·rol8<>(SB), DD0, DD0; VPSHUFB ·rol8<>(SB), DD1, DD1; VPSHUFB ·rol8<>(SB), DD2, DD2; VPSHUFB ·rol8<>(SB), DD3, DD3
	VPADDD   DD0, CC0, CC0; VPADDD DD1, CC1, CC1; VPADDD DD2, CC2, CC2; VPADDD DD3, CC3, CC3
	VPXOR    CC0, BB0, BB0; VPXOR CC1, BB1, BB1; VPXOR CC2, BB2, BB2; VPXOR CC3, BB3, BB3
	VMOVDQA  CC3, tmpStoreAVX2
	VPSLLD   $7, BB0, CC3; VPSRLD $25, BB0, BB0; VPXOR CC3, BB0, BB0
	VPSLLD   $7, BB1, CC3; VPSRLD $25, BB1, BB1; VPXOR CC3, BB1, BB1
	VPSLLD   $7, BB2, CC3; VPSRLD $25, BB2, BB2; VPXOR CC3, BB2, BB2
	VPSLLD   $7, BB3, CC3; VPSRLD $25, BB3, BB3; VPXOR CC3, BB3, BB3
	VMOVDQA  tmpStoreAVX2, CC3
	VPALIGNR $12, BB0, BB0, BB0; VPALIGNR $12, BB1, BB1, BB1; VPALIGNR $12, BB2, BB2, BB2; VPALIGNR $12, BB3, BB3, BB3
	VPALIGNR $8, CC0, CC0, CC0; VPALIGNR $8, CC1, CC1, CC1; VPALIGNR $8, CC2, CC2, CC2; VPALIGNR $8, CC3, CC3, CC3
	VPALIGNR $4, DD0, DD0, DD0; VPALIGNR $4, DD1, DD1, DD1; VPALIGNR $4, DD2, DD2, DD2; VPALIGNR $4, DD3, DD3, DD3
	INCQ     itr1
	CMPQ     itr1, $4
	JLT      openAVX2Tail512LoopB

	CMPQ itr1, $10
	JNE  openAVX2Tail512LoopA

	MOVQ inl, itr1
	SUBQ $384, itr1
	ANDQ $-16, itr1

openAVX2Tail512HashLoop:
	TESTQ itr1, itr1
	JE    openAVX2Tail512HashEnd
	polyAdd(0(itr2))
	polyMulAVX2
	LEAQ  16(itr2), itr2
	SUBQ  $16, itr1
	JMP   openAVX2Tail512HashLoop

openAVX2Tail512HashEnd:
	VPADDD     ·chacha20Constants<>(SB), AA0, AA0; VPADDD ·chacha20Constants<>(SB), AA1, AA1; VPADDD ·chacha20Constants<>(SB), AA2, AA2; VPADDD ·chacha20Constants<>(SB), AA3, AA3
	VPADDD     state1StoreAVX2, BB0, BB0; VPADDD state1StoreAVX2, BB1, BB1; VPADDD state1StoreAVX2, BB2, BB2; VPADDD state1StoreAVX2, BB3, BB3
	VPADDD     state2StoreAVX2, CC0, CC0; VPADDD state2StoreAVX2, CC1, CC1; VPADDD state2StoreAVX2, CC2, CC2; VPADDD state2StoreAVX2, CC3, CC3
	VPADDD     ctr0StoreAVX2, DD0, DD0; VPADDD ctr1StoreAVX2, DD1, DD1; VPADDD ctr2StoreAVX2, DD2, DD2; VPADDD ctr3StoreAVX2, DD3, DD3
	VMOVDQA    CC3, tmpStoreAVX2
	VPERM2I128 $0x02, AA0, BB0, CC3; VPERM2I128 $0x13, AA0, BB0, BB0; VPERM2I128 $0x02, CC0, DD0, AA0; VPERM2I128 $0x13, CC0, DD0, CC0
	VPXOR      (0*32)(inp), CC3, CC3; VPXOR (1*32)(inp), AA0, AA0; VPXOR (2*32)(inp), BB0, BB0; VPXOR (3*32)(inp), CC0, CC0
	VMOVDQU    CC3, (0*32)(oup); VMOVDQU AA0, (1*32)(oup); VMOVDQU BB0, (2*32)(oup); VMOVDQU CC0, (3*32)(oup)
	VPERM2I128 $0x02, AA1, BB1, AA0; VPERM2I128 $0x02, CC1, DD1, BB0; VPERM2I128 $0x13, AA1, BB1, CC0; VPERM2I128 $0x13, CC1, DD1, DD0
	VPXOR      (4*32)(inp), AA0, AA0; VPXOR (5*32)(inp), BB0, BB0; VPXOR (6*32)(inp), CC0, CC0; VPXOR (7*32)(inp), DD0, DD0
	VMOVDQU    AA0, (4*32)(oup); VMOVDQU BB0, (5*32)(oup); VMOVDQU CC0, (6*32)(oup); VMOVDQU DD0, (7*32)(oup)
	VPERM2I128 $0x02, AA2, BB2, AA0; VPERM2I128 $0x02, CC2, DD2, BB0; VPERM2I128 $0x13, AA2, BB2, CC0; VPERM2I128 $0x13, CC2, DD2, DD0
	VPXOR      (8*32)(inp), AA0, AA0; VPXOR (9*32)(inp), BB0, BB0; VPXOR (10*32)(inp), CC0, CC0; VPXOR (11*32)(inp), DD0, DD0
	VMOVDQU    AA0, (8*32)(oup); VMOVDQU BB0, (9*32)(oup); VMOVDQU CC0, (10*32)(oup); VMOVDQU DD0, (11*32)(oup)
	VPERM2I128 $0x02, AA3, BB3, AA0; VPERM2I128 $0x02, tmpStoreAVX2, DD3, BB0; VPERM2I128 $0x13, AA3, BB3, CC0; VPERM2I128 $0x13, tmpStoreAVX2, DD3, DD0

	LEAQ (12*32)(inp), inp
	LEAQ (12*32)(oup), oup
	SUBQ $12*32, inl

	JMP openAVX2TailLoop

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// func chacha20Poly1305Seal(dst, key, src, ad []byte)
TEXT ·chacha20Poly1305Seal(SB), 0, $288-96
	// For aligned stack access
	MOVQ SP, BP
	ADDQ $32, BP
	ANDQ $-32, BP
	MOVQ dst+0(FP), oup
	MOVQ key+24(FP), keyp
	MOVQ src+48(FP), inp
	MOVQ src_len+56(FP), inl
	MOVQ ad+72(FP), adp

	CMPB ·useAVX2(SB), $1
	JE   chacha20Poly1305Seal_AVX2

	// Special optimization, for very short buffers
	CMPQ inl, $128
	JBE  sealSSE128 // About 15% faster

	// In the seal case - prepare the poly key + 3 blocks of stream in the first iteration
	MOVOU ·chacha20Constants<>(SB), A0
	MOVOU (1*16)(keyp), B0
	MOVOU (2*16)(keyp), C0
	MOVOU (3*16)(keyp), D0

	// Store state on stack for future use
	MOVO B0, state1Store
	MOVO C0, state2Store

	// Load state, increment counter blocks
	MOVO A0, A1; MOVO B0, B1; MOVO C0, C1; MOVO D0, D1; PADDL ·sseIncMask<>(SB), D1
	MOVO A1, A2; MOVO B1, B2; MOVO C1, C2; MOVO D1, D2; PADDL ·sseIncMask<>(SB), D2
	MOVO A2, A3; MOVO B2, B3; MOVO C2, C3; MOVO D2, D3; PADDL ·sseIncMask<>(SB), D3

	// Store counters
	MOVO D0, ctr0Store; MOVO D1, ctr1Store; MOVO D2, ctr2Store; MOVO D3, ctr3Store
	MOVQ $10, itr2

sealSSEIntroLoop:
	MOVO         C3, tmpStore
	chachaQR(A0, B0, C0, D0, C3); chachaQR(A1, B1, C1, D1, C3); chachaQR(A2, B2, C2, D2, C3)
	MOVO         tmpStore, C3
	MOVO         C1, tmpStore
	chachaQR(A3, B3, C3, D3, C1)
	MOVO         tmpStore, C1
	shiftB0Left; shiftB1Left; shiftB2Left; shiftB3Left
	shiftC0Left; shiftC1Left; shiftC2Left; shiftC3Left
	shiftD0Left; shiftD1Left; shiftD2Left; shiftD3Left

	MOVO          C3, tmpStore
	chachaQR(A0, B0, C0, D0, C3); chachaQR(A1, B1, C1, D1, C3); chachaQR(A2, B2, C2, D2, C3)
	MOVO          tmpStore, C3
	MOVO          C1, tmpStore
	chachaQR(A3, B3, C3, D3, C1)
	MOVO          tmpStore, C1
	shiftB0Right; shiftB1Right; shiftB2Right; shiftB3Right
	shiftC0Right; shiftC1Right; shiftC2Right; shiftC3Right
	shiftD0Right; shiftD1Right; shiftD2Right; shiftD3Right
	DECQ          itr2
	JNE           sealSSEIntroLoop

	// Add in the state
	PADDD ·chacha20Constants<>(SB), A0; PADDD ·chacha20Constants<>(SB), A1; PADDD ·chacha20Constants<>(SB), A2; PADDD ·chacha20Constants<>(SB), A3
	PADDD state1Store, B0; PADDD state1Store, B1; PADDD state1Store, B2; PADDD state1Store, B3
	PADDD state2Store, C1; PADDD state2Store, C2; PADDD state2Store, C3
	PADDD ctr1Store, D1; PADDD ctr2Store, D2; PADDD ctr3Store, D3

	// Clamp and store the key
	PAND ·polyClampMask<>(SB), A0
	MOVO A0, rStore
	MOVO B0, sStore

	// Hash AAD
	MOVQ ad_len+80(FP), itr2
	CALL polyHashADInternal<>(SB)

	MOVOU (0*16)(inp), A0; MOVOU (1*16)(inp), B0; MOVOU (2*16)(inp), C0; MOVOU (3*16)(inp), D0
	PXOR  A0, A1; PXOR B0, B1; PXOR C0, C1; PXOR D0, D1
	MOVOU A1, (0*16)(oup); MOVOU B1, (1*16)(oup); MOVOU C1, (2*16)(oup); MOVOU D1, (3*16)(oup)
	MOVOU (4*16)(inp), A0; MOVOU (5*16)(inp), B0; MOVOU (6*16)(inp), C0; MOVOU (7*16)(inp), D0
	PXOR  A0, A2; PXOR B0, B2; PXOR C0, C2; PXOR D0, D2
	MOVOU A2, (4*16)(oup); MOVOU B2, (5*16)(oup); MOVOU C2, (6*16)(oup); MOVOU D2, (7*16)(oup)

	MOVQ $128, itr1
	SUBQ $128, inl
	LEAQ 128(inp), inp

	MOVO A3, A1; MOVO B3, B1; MOVO C3, C1; MOVO D3, D1

	CMPQ inl, $64
	JBE  sealSSE128SealHash

	MOVOU (0*16)(inp), A0; MOVOU (1*16)(inp), B0; MOVOU (2*16)(inp), C0; MOVOU (3*16)(inp), D0
	PXOR  A0, A3; PXOR B0, B3; PXOR C0, C3; PXOR D0, D3
	MOVOU A3, (8*16)(oup); MOVOU B3, (9*16)(oup); MOVOU C3, (10*16)(oup); MOVOU D3, (11*16)(oup)

	ADDQ $64, itr1
	SUBQ $64, inl
	LEAQ 64(inp), inp

	MOVQ $2, itr1
	MOVQ $8, itr2

	CMPQ inl, $64
	JBE  sealSSETail64
	CMPQ inl, $128
	JBE  sealSSETail128
	CMPQ inl, $192
	JBE  sealSSETail192

sealSSEMainLoop:
	// Load state, increment counter blocks
	MOVO ·chacha20Constants<>(SB), A0; MOVO state1Store, B0; MOVO state2Store, C0; MOVO ctr3Store, D0; PADDL ·sseIncMask<>(SB), D0
	MOVO A0, A1; MOVO B0, B1; MOVO C0, C1; MOVO D0, D1; PADDL ·sseIncMask<>(SB), D1
	MOVO A1, A2; MOVO B1, B2; MOVO C1, C2; MOVO D1, D2; PADDL ·sseIncMask<>(SB), D2
	MOVO A2, A3; MOVO B2, B3; MOVO C2, C3; MOVO D2, D3; PADDL ·sseIncMask<>(SB), D3

	// Store counters
	MOVO D0, ctr0Store; MOVO D1, ctr1Store; MOVO D2, ctr2Store; MOVO D3, ctr3Store

sealSSEInnerLoop:
	MOVO          C3, tmpStore
	chachaQR(A0, B0, C0, D0, C3); chachaQR(A1, B1, C1, D1, C3); chachaQR(A2, B2, C2, D2, C3)
	MOVO          tmpStore, C3
	MOVO          C1, tmpStore
	chachaQR(A3, B3, C3, D3, C1)
	MOVO          tmpStore, C1
	polyAdd(0(oup))
	shiftB0Left;  shiftB1Left; shiftB2Left; shiftB3Left
	shiftC0Left;  shiftC1Left; shiftC2Left; shiftC3Left
	shiftD0Left;  shiftD1Left; shiftD2Left; shiftD3Left
	polyMulStage1
	polyMulStage2
	LEAQ          (2*8)(oup), oup
	MOVO          C3, tmpStore
	chachaQR(A0, B0, C0, D0, C3); chachaQR(A1, B1, C1, D1, C3); chachaQR(A2, B2, C2, D2, C3)
	MOVO          tmpStore, C3
	MOVO          C1, tmpStore
	polyMulStage3
	chachaQR(A3, B3, C3, D3, C1)
	MOVO          tmpStore, C1
	polyMulReduceStage
	shiftB0Right; shiftB1Right; shiftB2Right; shiftB3Right
	shiftC0Right; shiftC1Right; shiftC2Right; shiftC3Right
	shiftD0Right; shiftD1Right; shiftD2Right; shiftD3Right
	DECQ          itr2
	JGE           sealSSEInnerLoop
	polyAdd(0(oup))
	polyMul
	LEAQ          (2*8)(oup), oup
	DECQ          itr1
	JG            sealSSEInnerLoop

	// Add in the state
	PADDD ·chacha20Constants<>(SB), A0; PADDD ·chacha20Constants<>(SB), A1; PADDD ·chacha20Constants<>(SB), A2; PADDD ·chacha20Constants<>(SB), A3
	PADDD state1Store, B0; PADDD state1Store, B1; PADDD state1Store, B2; PADDD state1Store, B3
	PADDD state2Store, C0; PADDD state2Store, C1; PADDD state2Store, C2; PADDD state2Store, C3
	PADDD ctr0Store, D0; PADDD ctr1Store, D1; PADDD ctr2Store, D2; PADDD ctr3Store, D3
	MOVO  D3, tmpStore

	// Load - xor - store
	MOVOU (0*16)(inp), D3; PXOR D3, A0
	MOVOU (1*16)(inp), D3; PXOR D3, B0
	MOVOU (2*16)(inp), D3; PXOR D3, C0
	MOVOU (3*16)(inp), D3; PXOR D3, D0
	MOVOU A0, (0*16)(oup)
	MOVOU B0, (1*16)(oup)
	MOVOU C0, (2*16)(oup)
	MOVOU D0, (3*16)(oup)
	MOVO  tmpStore, D3

	MOVOU (4*16)(inp), A0; MOVOU (5*16)(inp), B0; MOVOU (6*16)(inp), C0; MOVOU (7*16)(inp), D0
	PXOR  A0, A1; PXOR B0, B1; PXOR C0, C1; PXOR D0, D1
	MOVOU A1, (4*16)(oup); MOVOU B1, (5*16)(oup); MOVOU C1, (6*16)(oup); MOVOU D1, (7*16)(oup)
	MOVOU (8*16)(inp), A0; MOVOU (9*16)(inp), B0; MOVOU (10*16)(inp), C0; MOVOU (11*16)(inp), D0
	PXOR  A0, A2; PXOR B0, B2; PXOR C0, C2; PXOR D0, D2
	MOVOU A2, (8*16)(oup); MOVOU B2, (9*16)(oup); MOVOU C2, (10*16)(oup); MOVOU D2, (11*16)(oup)
	ADDQ  $192, inp
	MOVQ  $192, itr1
	SUBQ  $192, inl
	MOVO  A3, A1
	MOVO  B3, B1
	MOVO  C3, C1
	MOVO  D3, D1
	CMPQ  inl, $64
	JBE   sealSSE128SealHash
	MOVOU (0*16)(inp), A0; MOVOU (1*16)(inp), B0; MOVOU (2*16)(inp), C0; MOVOU (3*16)(inp), D0
	PXOR  A0, A3; PXOR B0, B3; PXOR C0, C3; PXOR D0, D3
	MOVOU A3, (12*16)(oup); MOVOU B3, (13*16)(oup); MOVOU C3, (14*16)(oup); MOVOU D3, (15*16)(oup)
	LEAQ  64(inp), inp
	SUBQ  $64, inl
	MOVQ  $6, itr1
	MOVQ  $4, itr2
	CMPQ  inl, $192
	JG    sealSSEMainLoop

	MOVQ  inl, itr1
	TESTQ inl, inl
	JE    sealSSE128SealHash
	MOVQ  $6, itr1
	CMPQ  inl, $64
	JBE   sealSSETail64
	CMPQ  inl, $128
	JBE   sealSSETail128
	JMP   sealSSETail192

// ----------------------------------------------------------------------------
// Special optimization for the last 64 bytes of plaintext
sealSSETail64:
	// Need to encrypt up to 64 bytes - prepare single block, hash 192 or 256 bytes
	MOVO  ·chacha20Constants<>(SB), A1
	MOVO  state1Store, B1
	MOVO  state2Store, C1
	MOVO  ctr3Store, D1
	PADDL ·sseIncMask<>(SB), D1
	MOVO  D1, ctr0Store

sealSSETail64LoopA:
	// Perform ChaCha rounds, while hashing the previously encrypted ciphertext
	polyAdd(0(oup))
	polyMul
	LEAQ 16(oup), oup

sealSSETail64LoopB:
	chachaQR(A1, B1, C1, D1, T1)
	shiftB1Left;  shiftC1Left; shiftD1Left
	chachaQR(A1, B1, C1, D1, T1)
	shiftB1Right; shiftC1Right; shiftD1Right
	polyAdd(0(oup))
	polyMul
	LEAQ          16(oup), oup

	DECQ itr1
	JG   sealSSETail64LoopA

	DECQ  itr2
	JGE   sealSSETail64LoopB
	PADDL ·chacha20Constants<>(SB), A1
	PADDL state1Store, B1
	PADDL state2Store, C1
	PADDL ctr0Store, D1

	JMP sealSSE128Seal

// ----------------------------------------------------------------------------
// Special optimization for the last 128 bytes of plaintext
sealSSETail128:
	// Need to encrypt up to 128 bytes - prepare two blocks, hash 192 or 256 bytes
	MOVO ·chacha20Constants<>(SB), A0; MOVO state1Store, B0; MOVO state2Store, C0; MOVO ctr3Store, D0; PADDL ·sseIncMask<>(SB), D0; MOVO D0, ctr0Store
	MOVO A0, A1; MOVO B0, B1; MOVO C0, C1; MOVO D0, D1; PADDL ·sseIncMask<>(SB), D1; MOVO D1, ctr1Store

sealSSETail128LoopA:
	// Perform ChaCha rounds, while hashing the previously encrypted ciphertext
	polyAdd(0(oup))
	polyMul
	LEAQ 16(oup), oup

sealSSETail128LoopB:
	chachaQR(A0, B0, C0, D0, T0); chachaQR(A1, B1, C1, D1, T0)
	shiftB0Left;  shiftC0Left; shiftD0Left
	shiftB1Left;  shiftC1Left; shiftD1Left
	polyAdd(0(oup))
	polyMul
	LEAQ          16(oup), oup
	chachaQR(A0, B0, C0, D0, T0); chachaQR(A1, B1, C1, D1, T0)
	shiftB0Right; shiftC0Right; shiftD0Right
	shiftB1Right; shiftC1Right; shiftD1Right

	DECQ itr1
	JG   sealSSETail128LoopA

	DECQ itr2
	JGE  sealSSETail128LoopB

	PADDL ·chacha20Constants<>(SB), A0; PADDL ·chacha20Constants<>(SB), A1
	PADDL state1Store, B0; PADDL state1Store, B1
	PADDL state2Store, C0; PADDL state2Store, C1
	PADDL ctr0Store, D0; PADDL ctr1Store, D1

	MOVOU (0*16)(inp), T0; MOVOU (1*16)(inp), T1; MOVOU (2*16)(inp), T2; MOVOU (3*16)(inp), T3
	PXOR  T0, A0; PXOR T1, B0; PXOR T2, C0; PXOR T3, D0
	MOVOU A0, (0*16)(oup); MOVOU B0, (1*16)(oup); MOVOU C0, (2*16)(oup); MOVOU D0, (3*16)(oup)

	MOVQ $64, itr1
	LEAQ 64(inp), inp
	SUBQ $64, inl

	JMP sealSSE128SealHash

// ----------------------------------------------------------------------------
// Special optimization for the last 192 bytes of plaintext
sealSSETail192:
	// Need to encrypt up to 192 bytes - prepare three blocks, hash 192 or 256 bytes
	MOVO ·chacha20Constants<>(SB), A0; MOVO state1Store, B0; MOVO state2Store, C0; MOVO ctr3Store, D0; PADDL ·sseIncMask<>(SB), D0; MOVO D0, ctr0Store
	MOVO A0, A1; MOVO B0, B1; MOVO C0, C1; MOVO D0, D1; PADDL ·sseIncMask<>(SB), D1; MOVO D1, ctr1Store
	MOVO A1, A2; MOVO B1, B2; MOVO C1, C2; MOVO D1, D2; PADDL ·sseIncMask<>(SB), D2; MOVO D2, ctr2Store

sealSSETail192LoopA:
	// Perform ChaCha rounds, while hashing the previously encrypted ciphertext
	polyAdd(0(oup))
	polyMul
	LEAQ 16(oup), oup

sealSSETail192LoopB:
	chachaQR(A0, B0, C0, D0, T0); chachaQR(A1, B1, C1, D1, T0); chachaQR(A2, B2, C2, D2, T0)
	shiftB0Left; shiftC0Left; shiftD0Left
	shiftB1Left; shiftC1Left; shiftD1Left
	shiftB2Left; shiftC2Left; shiftD2Left

	polyAdd(0(oup))
	polyMul
	LEAQ 16(oup), oup

	chachaQR(A0, B0, C0, D0, T0); chachaQR(A1, B1, C1, D1, T0); chachaQR(A2, B2, C2, D2, T0)
	shiftB0Right; shiftC0Right; shiftD0Right
	shiftB1Right; shiftC1Right; shiftD1Right
	shiftB2Right; shiftC2Right; shiftD2Right

	DECQ itr1
	JG   sealSSETail192LoopA

	DECQ itr2
	JGE  sealSSETail192LoopB

	PADDL ·chacha20Constants<>(SB), A0; PADDL ·chacha20Constants<>(SB), A1; PADDL ·chacha20Constants<>(SB), A2
	PADDL state1Store, B0; PADDL state1Store, B1; PADDL state1Store, B2
	PADDL state2Store, C0; PADDL state2Store, C1; PADDL state2Store, C2
	PADDL ctr0Store, D0; PADDL ctr1Store, D1; PADDL ctr2Store, D2

	MOVOU (0*16)(inp), T0; MOVOU (1*16)(inp), T1; MOVOU (2*16)(inp), T2; MOVOU (3*16)(inp), T3
	PXOR  T0, A0; PXOR T1, B0; PXOR T2, C0; PXOR T3, D0
	MOVOU A0, (0*16)(oup); MOVOU B0, (1*16)(oup); MOVOU C0, (2*16)(oup); MOVOU D0, (3*16)(oup)
	MOVOU (4*16)(inp), T0; MOVOU (5*16)(inp), T1; MOVOU (6*16)(inp), T2; MOVOU (7*16)(inp), T3
	PXOR  T0, A1; PXOR T1, B1; PXOR T2, C1; PXOR T3, D1
	MOVOU A1, (4*16)(oup); MOVOU B1, (5*16)(oup); MOVOU C1, (6*16)(oup); MOVOU D1, (7*16)(oup)

	MOVO A2, A1
	MOVO B2, B1
	MOVO C2, C1
	MOVO D2, D1
	MOVQ $128, itr1
	LEAQ 128(inp), inp
	SUBQ $128, inl

	JMP sealSSE128SealHash

// ----------------------------------------------------------------------------
// Special seal optimization for buffers smaller than 129 bytes
sealSSE128:
	// For up to 128 bytes of ciphertext and 64 bytes for the poly key, we require to process three blocks
	MOVOU ·chacha20Constants<>(SB), A0; MOVOU (1*16)(keyp), B0; MOVOU (2*16)(keyp), C0; MOVOU (3*16)(keyp), D0
	MOVO  A0, A1; MOVO B0, B1; MOVO C0, C1; MOVO D0, D1; PADDL ·sseIncMask<>(SB), D1
	MOVO  A1, A2; MOVO B1, B2; MOVO C1, C2; MOVO D1, D2; PADDL ·sseIncMask<>(SB), D2
	MOVO  B0, T1; MOVO C0, T2; MOVO D1, T3
	MOVQ  $10, itr2

sealSSE128InnerCipherLoop:
	chachaQR(A0, B0, C0, D0, T0); chachaQR(A1, B1, C1, D1, T0); chachaQR(A2, B2, C2, D2, T0)
	shiftB0Left;  shiftB1Left; shiftB2Left
	shiftC0Left;  shiftC1Left; shiftC2Left
	shiftD0Left;  shiftD1Left; shiftD2Left
	chachaQR(A0, B0, C0, D0, T0); chachaQR(A1, B1, C1, D1, T0); chachaQR(A2, B2, C2, D2, T0)
	shiftB0Right; shiftB1Right; shiftB2Right
	shiftC0Right; shiftC1Right; shiftC2Right
	shiftD0Right; shiftD1Right; shiftD2Right
	DECQ          itr2
	JNE           sealSSE128InnerCipherLoop

	// A0|B0 hold the Poly1305 32-byte key, C0,D0 can be discarded
	PADDL ·chacha20Constants<>(SB), A0; PADDL ·chacha20Constants<>(SB), A1; PADDL ·chacha20Constants<>(SB), A2
	PADDL T1, B0; PADDL T1, B1; PADDL T1, B2
	PADDL T2, C1; PADDL T2, C2
	PADDL T3, D1; PADDL ·sseIncMask<>(SB), T3; PADDL T3, D2
	PAND  ·polyClampMask<>(SB), A0
	MOVOU A0, rStore
	MOVOU B0, sStore

	// Hash
	MOVQ ad_len+80(FP), itr2
	CALL polyHashADInternal<>(SB)
	XORQ itr1, itr1

sealSSE128SealHash:
	// itr1 holds the number of bytes encrypted but not yet hashed
	CMPQ itr1, $16
	JB   sealSSE128Seal
	polyAdd(0(oup))
	polyMul

	SUBQ $16, itr1
	ADDQ $16, oup

	JMP sealSSE128SealHash

sealSSE128Seal:
	CMPQ inl, $16
	JB   sealSSETail
	SUBQ $16, inl

	// Load for decryption
	MOVOU (inp), T0
	PXOR  T0, A1
	MOVOU A1, (oup)
	LEAQ  (1*16)(inp), inp
	LEAQ  (1*16)(oup), oup

	// Extract for hashing
	MOVQ   A1, t0
	PSRLDQ $8, A1
	MOVQ A1, t1
	ADDQ   t0, acc0; ADCQ t1, acc1; ADCQ $1, acc2
	polyMul

	// Shift the stream "left"
	MOVO B1, A1
	MOVO C1, B1
	MOVO D1, C1
	MOVO A2, D1
	MOVO B2, A2
	MOVO C2, B2
	MOVO D2, C2
	JMP  sealSSE128Seal

sealSSETail:
	TESTQ inl, inl
	JE    sealSSEFinalize

	// We can only load the PT one byte at a time to avoid read after end of buffer
	MOVQ inl, itr2
	SHLQ $4, itr2
	LEAQ ·andMask<>(SB), t0
	MOVQ inl, itr1
	LEAQ -1(inp)(inl*1), inp
	XORQ t2, t2
	XORQ t3, t3
	XORQ AX, AX

sealSSETailLoadLoop:
	SHLQ $8, t2, t3
	SHLQ $8, t2
	MOVB (inp), AX
	XORQ AX, t2
	LEAQ   -1(inp), inp
	DECQ   itr1
	JNE    sealSSETailLoadLoop
	MOVQ t2, 0+tmpStore
	MOVQ t3, 8+tmpStore
	PXOR 0+tmpStore, A1
	MOVOU  A1, (oup)
	MOVOU  -16(t0)(itr2*1), T0
	PAND   T0, A1
	MOVQ   A1, t0
	PSRLDQ $8, A1
	MOVQ   A1, t1
	ADDQ   t0, acc0; ADCQ t1, acc1; ADCQ $1, acc2
	polyMul

	ADDQ inl, oup

sealSSEFinalize:
	// Hash in the buffer lengths
	ADDQ ad_len+80(FP), acc0
	ADCQ src_len+56(FP), acc1
	ADCQ $1, acc2
	polyMul

	// Final reduce
	MOVQ    acc0, t0
	MOVQ    acc1, t1
	MOVQ    acc2, t2
	SUBQ    $-5, acc0
	SBBQ    $-1, acc1
	SBBQ    $3, acc2
	CMOVQCS t0, acc0
	CMOVQCS t1, acc1
	CMOVQCS t2, acc2

	// Add in the "s" part of the key
	ADDQ 0+sStore, acc0
	ADCQ 8+sStore, acc1

	// Finally store the tag at the end of the message
	MOVQ acc0, (0*8)(oup)
	MOVQ acc1, (1*8)(oup)
	RET

// ----------------------------------------------------------------------------
// ------------------------- AVX2 Code ----------------------------------------
chacha20Poly1305Seal_AVX2:
	VZEROUPPER
	VMOVDQU ·chacha20Constants<>(SB), AA0
	BYTE    $0xc4; BYTE $0x42; BYTE $0x7d; BYTE $0x5a; BYTE $0x70; BYTE $0x10 // broadcasti128 16(r8), ymm14
	BYTE    $0xc4; BYTE $0x42; BYTE $0x7d; BYTE $0x5a; BYTE $0x60; BYTE $0x20 // broadcasti128 32(r8), ymm12
	BYTE    $0xc4; BYTE $0xc2; BYTE $0x7d; BYTE $0x5a; BYTE $0x60; BYTE $0x30 // broadcasti128 48(r8), ymm4
	VPADDD  ·avx2InitMask<>(SB), DD0, DD0

	// Special optimizations, for very short buffers
	CMPQ inl, $192
	JBE  seal192AVX2 // 33% faster
	CMPQ inl, $320
	JBE  seal320AVX2 // 17% faster

	// For the general key prepare the key first - as a byproduct we have 64 bytes of cipher stream
	VMOVDQA AA0, AA1; VMOVDQA AA0, AA2; VMOVDQA AA0, AA3
	VMOVDQA BB0, BB1; VMOVDQA BB0, BB2; VMOVDQA BB0, BB3; VMOVDQA BB0, state1StoreAVX2
	VMOVDQA CC0, CC1; VMOVDQA CC0, CC2; VMOVDQA CC0, CC3; VMOVDQA CC0, state2StoreAVX2
	VPADDD  ·avx2IncMask<>(SB), DD0, DD1; VMOVDQA DD0, ctr0StoreAVX2
	VPADDD  ·avx2IncMask<>(SB), DD1, DD2; VMOVDQA DD1, ctr1StoreAVX2
	VPADDD  ·avx2IncMask<>(SB), DD2, DD3; VMOVDQA DD2, ctr2StoreAVX2
	VMOVDQA DD3, ctr3StoreAVX2
	MOVQ    $10, itr2

sealAVX2IntroLoop:
	VMOVDQA CC3, tmpStoreAVX2
	chachaQR_AVX2(AA0, BB0, CC0, DD0, CC3); chachaQR_AVX2(AA1, BB1, CC1, DD1, CC3); chachaQR_AVX2(AA2, BB2, CC2, DD2, CC3)
	VMOVDQA tmpStoreAVX2, CC3
	VMOVDQA CC1, tmpStoreAVX2
	chachaQR_AVX2(AA3, BB3, CC3, DD3, CC1)
	VMOVDQA tmpStoreAVX2, CC1

	VPALIGNR $4, BB0, BB0, BB0; VPALIGNR $8, CC0, CC0, CC0; VPALIGNR $12, DD0, DD0, DD0
	VPALIGNR $4, BB1, BB1, BB1; VPALIGNR $8, CC1, CC1, CC1; VPALIGNR $12, DD1, DD1, DD1
	VPALIGNR $4, BB2, BB2, BB2; VPALIGNR $8, CC2, CC2, CC2; VPALIGNR $12, DD2, DD2, DD2
	VPALIGNR $4, BB3, BB3, BB3; VPALIGNR $8, CC3, CC3, CC3; VPALIGNR $12, DD3, DD3, DD3

	VMOVDQA CC3, tmpStoreAVX2
	chachaQR_AVX2(AA0, BB0, CC0, DD0, CC3); chachaQR_AVX2(AA1, BB1, CC1, DD1, CC3); chachaQR_AVX2(AA2, BB2, CC2, DD2, CC3)
	VMOVDQA tmpStoreAVX2, CC3
	VMOVDQA CC1, tmpStoreAVX2
	chachaQR_AVX2(AA3, BB3, CC3, DD3, CC1)
	VMOVDQA tmpStoreAVX2, CC1

	VPALIGNR $12, BB0, BB0, BB0; VPALIGNR $8, CC0, CC0, CC0; VPALIGNR $4, DD0, DD0, DD0
	VPALIGNR $12, BB1, BB1, BB1; VPALIGNR $8, CC1, CC1, CC1; VPALIGNR $4, DD1, DD1, DD1
	VPALIGNR $12, BB2, BB2, BB2; VPALIGNR $8, CC2, CC2, CC2; VPALIGNR $4, DD2, DD2, DD2
	VPALIGNR $12, BB3, BB3, BB3; VPALIGNR $8, CC3, CC3, CC3; VPALIGNR $4, DD3, DD3, DD3
	DECQ     itr2
	JNE      sealAVX2IntroLoop

	VPADDD ·chacha20Constants<>(SB), AA0, AA0; VPADDD ·chacha20Constants<>(SB), AA1, AA1; VPADDD ·chacha20Constants<>(SB), AA2, AA2; VPADDD ·chacha20Constants<>(SB), AA3, AA3
	VPADDD state1StoreAVX2, BB0, BB0; VPADDD state1StoreAVX2, BB1, BB1; VPADDD state1StoreAVX2, BB2, BB2; VPADDD state1StoreAVX2, BB3, BB3
	VPADDD state2StoreAVX2, CC0, CC0; VPADDD state2StoreAVX2, CC1, CC1; VPADDD state2StoreAVX2, CC2, CC2; VPADDD state2StoreAVX2, CC3, CC3
	VPADDD ctr0StoreAVX2, DD0, DD0; VPADDD ctr1StoreAVX2, DD1, DD1; VPADDD ctr2StoreAVX2, DD2, DD2; VPADDD ctr3StoreAVX2, DD3, DD3

	VPERM2I128 $0x13, CC0, DD0, CC0 // Stream bytes 96 - 127
	VPERM2I128 $0x02, AA0, BB0, DD0 // The Poly1305 key
	VPERM2I128 $0x13, AA0, BB0, AA0 // Stream bytes 64 - 95

	// Clamp and store poly key
	VPAND   ·polyClampMask<>(SB), DD0, DD0
	VMOVDQA DD0, rsStoreAVX2

	// Hash AD
	MOVQ ad_len+80(FP), itr2
	CALL polyHashADInternal<>(SB)

	// Can store at least 320 bytes
	VPXOR   (0*32)(inp), AA0, AA0
	VPXOR   (1*32)(inp), CC0, CC0
	VMOVDQU AA0, (0*32)(oup)
	VMOVDQU CC0, (1*32)(oup)

	VPERM2I128 $0x02, AA1, BB1, AA0; VPERM2I128 $0x02, CC1, DD1, BB0; VPERM2I128 $0x13, AA1, BB1, CC0; VPERM2I128 $0x13, CC1, DD1, DD0
	VPXOR      (2*32)(inp), AA0, AA0; VPXOR (3*32)(inp), BB0, BB0; VPXOR (4*32)(inp), CC0, CC0; VPXOR (5*32)(inp), DD0, DD0
	VMOVDQU    AA0, (2*32)(oup); VMOVDQU BB0, (3*32)(oup); VMOVDQU CC0, (4*32)(oup); VMOVDQU DD0, (5*32)(oup)
	VPERM2I128 $0x02, AA2, BB2, AA0; VPERM2I128 $0x02, CC2, DD2, BB0; VPERM2I128 $0x13, AA2, BB2, CC0; VPERM2I128 $0x13, CC2, DD2, DD0
	VPXOR      (6*32)(inp), AA0, AA0; VPXOR (7*32)(inp), BB0, BB0; VPXOR (8*32)(inp), CC0, CC0; VPXOR (9*32)(inp), DD0, DD0
	VMOVDQU    AA0, (6*32)(oup); VMOVDQU BB0, (7*32)(oup); VMOVDQU CC0, (8*32)(oup); VMOVDQU DD0, (9*32)(oup)

	MOVQ $320, itr1
	SUBQ $320, inl
	LEAQ 320(inp), inp

	VPERM2I128 $0x02, AA3, BB3, AA0; VPERM2I128 $0x02, CC3, DD3, BB0; VPERM2I128 $0x13, AA3, BB3, CC0; VPERM2I128 $0x13, CC3, DD3, DD0
	CMPQ       inl, $128
	JBE        sealAVX2SealHash

	VPXOR   (0*32)(inp), AA0, AA0; VPXOR (1*32)(inp), BB0, BB0; VPXOR (2*32)(inp), CC0, CC0; VPXOR (3*32)(inp), DD0, DD0
	VMOVDQU AA0, (10*32)(oup); VMOVDQU BB0, (11*32)(oup); VMOVDQU CC0, (12*32)(oup); VMOVDQU DD0, (13*32)(oup)
	SUBQ    $128, inl
	LEAQ    128(inp), inp

	MOVQ $8, itr1
	MOVQ $2, itr2

	CMPQ inl, $128
	JBE  sealAVX2Tail128
	CMPQ inl, $256
	JBE  sealAVX2Tail256
	CMPQ inl, $384
	JBE  sealAVX2Tail384
	CMPQ inl, $512
	JBE  sealAVX2Tail512

	// We have 448 bytes to hash, but main loop hashes 512 bytes at a time - perform some rounds, before the main loop
	VMOVDQA ·chacha20Constants<>(SB), AA0; VMOVDQA AA0, AA1; VMOVDQA AA0, AA2; VMOVDQA AA0, AA3
	VMOVDQA state1StoreAVX2, BB0; VMOVDQA BB0, BB1; VMOVDQA BB0, BB2; VMOVDQA BB0, BB3
	VMOVDQA state2StoreAVX2, CC0; VMOVDQA CC0, CC1; VMOVDQA CC0, CC2; VMOVDQA CC0, CC3
	VMOVDQA ctr3StoreAVX2, DD0
	VPADDD  ·avx2IncMask<>(SB), DD0, DD0; VPADDD ·avx2IncMask<>(SB), DD0, DD1; VPADDD ·avx2IncMask<>(SB), DD1, DD2; VPADDD ·avx2IncMask<>(SB), DD2, DD3
	VMOVDQA DD0, ctr0StoreAVX2; VMOVDQA DD1, ctr1StoreAVX2; VMOVDQA DD2, ctr2StoreAVX2; VMOVDQA DD3, ctr3StoreAVX2

	VMOVDQA CC3, tmpStoreAVX2
	chachaQR_AVX2(AA0, BB0, CC0, DD0, CC3); chachaQR_AVX2(AA1, BB1, CC1, DD1, CC3); chachaQR_AVX2(AA2, BB2, CC2, DD2, CC3)
	VMOVDQA tmpStoreAVX2, CC3
	VMOVDQA CC1, tmpStoreAVX2
	chachaQR_AVX2(AA3, BB3, CC3, DD3, CC1)
	VMOVDQA tmpStoreAVX2, CC1

	VPALIGNR $4, BB0, BB0, BB0; VPALIGNR $8, CC0, CC0, CC0; VPALIGNR $12, DD0, DD0, DD0
	VPALIGNR $4, BB1, BB1, BB1; VPALIGNR $8, CC1, CC1, CC1; VPALIGNR $12, DD1, DD1, DD1
	VPALIGNR $4, BB2, BB2, BB2; VPALIGNR $8, CC2, CC2, CC2; VPALIGNR $12, DD2, DD2, DD2
	VPALIGNR $4, BB3, BB3, BB3; VPALIGNR $8, CC3, CC3, CC3; VPALIGNR $12, DD3, DD3, DD3

	VMOVDQA CC3, tmpStoreAVX2
	chachaQR_AVX2(AA0, BB0, CC0, DD0, CC3); chachaQR_AVX2(AA1, BB1, CC1, DD1, CC3); chachaQR_AVX2(AA2, BB2, CC2, DD2, CC3)
	VMOVDQA tmpStoreAVX2, CC3
	VMOVDQA CC1, tmpStoreAVX2
	chachaQR_AVX2(AA3, BB3, CC3, DD3, CC1)
	VMOVDQA tmpStoreAVX2, CC1

	VPALIGNR $12, BB0, BB0, BB0; VPALIGNR $8, CC0, CC0, CC0; VPALIGNR $4, DD0, DD0, DD0
	VPALIGNR $12, BB1, BB1, BB1; VPALIGNR $8, CC1, CC1, CC1; VPALIGNR $4, DD1, DD1, DD1
	VPALIGNR $12, BB2, BB2, BB2; VPALIGNR $8, CC2, CC2, CC2; VPALIGNR $4, DD2, DD2, DD2
	VPALIGNR $12, BB3, BB3, BB3; VPALIGNR $8, CC3, CC3, CC3; VPALIGNR $4, DD3, DD3, DD3
	VPADDD   BB0, AA0, AA0; VPADDD BB1, AA1, AA1; VPADDD BB2, AA2, AA2; VPADDD BB3, AA3, AA3
	VPXOR    AA0, DD0, DD0; VPXOR AA1, DD1, DD1; VPXOR AA2, DD2, DD2; VPXOR AA3, DD3, DD3
	VPSHUFB  ·rol16<>(SB), DD0, DD0; VPSHUFB ·rol16<>(SB), DD1, DD1; VPSHUFB ·rol16<>(SB), DD2, DD2; VPSHUFB ·rol16<>(SB), DD3, DD3
	VPADDD   DD0, CC0, CC0; VPADDD DD1, CC1, CC1; VPADDD DD2, CC2, CC2; VPADDD DD3, CC3, CC3
	VPXOR    CC0, BB0, BB0; VPXOR CC1, BB1, BB1; VPXOR CC2, BB2, BB2; VPXOR CC3, BB3, BB3
	VMOVDQA  CC3, tmpStoreAVX2
	VPSLLD   $12, BB0, CC3; VPSRLD $20, BB0, BB0; VPXOR CC3, BB0, BB0
	VPSLLD   $12, BB1, CC3; VPSRLD $20, BB1, BB1; VPXOR CC3, BB1, BB1
	VPSLLD   $12, BB2, CC3; VPSRLD $20, BB2, BB2; VPXOR CC3, BB2, BB2
	VPSLLD   $12, BB3, CC3; VPSRLD $20, BB3, BB3; VPXOR CC3, BB3, BB3
	VMOVDQA  tmpStoreAVX2, CC3

	SUBQ $16, oup                  // Adjust the pointer
	MOVQ $9, itr1
	JMP  sealAVX2InternalLoopStart

sealAVX2MainLoop:
	// Load state, increment counter blocks, store the incremented counters
	VMOVDQU ·chacha20Constants<>(SB), AA0; VMOVDQA AA0, AA1; VMOVDQA AA0, AA2; VMOVDQA AA0, AA3
	VMOVDQA state1StoreAVX2, BB0; VMOVDQA BB0, BB1; VMOVDQA BB0, BB2; VMOVDQA BB0, BB3
	VMOVDQA state2StoreAVX2, CC0; VMOVDQA CC0, CC1; VMOVDQA CC0, CC2; VMOVDQA CC0, CC3
	VMOVDQA ctr3StoreAVX2, DD0; VPADDD ·avx2IncMask<>(SB), DD0, DD0; VPADDD ·avx2IncMask<>(SB), DD0, DD1; VPADDD ·avx2IncMask<>(SB), DD1, DD2; VPADDD ·avx2IncMask<>(SB), DD2, DD3
	VMOVDQA DD0, ctr0StoreAVX2; VMOVDQA DD1, ctr1StoreAVX2; VMOVDQA DD2, ctr2StoreAVX2; VMOVDQA DD3, ctr3StoreAVX2
	MOVQ    $10, itr1

sealAVX2InternalLoop:
	polyAdd(0*8(oup))
	VPADDD  BB0, AA0, AA0; VPADDD BB1, AA1, AA1; VPADDD BB2, AA2, AA2; VPADDD BB3, AA3, AA3
	polyMulStage1_AVX2
	VPXOR   AA0, DD0, DD0; VPXOR AA1, DD1, DD1; VPXOR AA2, DD2, DD2; VPXOR AA3, DD3, DD3
	VPSHUFB ·rol16<>(SB), DD0, DD0; VPSHUFB ·rol16<>(SB), DD1, DD1; VPSHUFB ·rol16<>(SB), DD2, DD2; VPSHUFB ·rol16<>(SB), DD3, DD3
	polyMulStage2_AVX2
	VPADDD  DD0, CC0, CC0; VPADDD DD1, CC1, CC1; VPADDD DD2, CC2, CC2; VPADDD DD3, CC3, CC3
	VPXOR   CC0, BB0, BB0; VPXOR CC1, BB1, BB1; VPXOR CC2, BB2, BB2; VPXOR CC3, BB3, BB3
	polyMulStage3_AVX2
	VMOVDQA CC3, tmpStoreAVX2
	VPSLLD  $12, BB0, CC3; VPSRLD $20, BB0, BB0; VPXOR CC3, BB0, BB0
	VPSLLD  $12, BB1, CC3; VPSRLD $20, BB1, BB1; VPXOR CC3, BB1, BB1
	VPSLLD  $12, BB2, CC3; VPSRLD $20, BB2, BB2; VPXOR CC3, BB2, BB2
	VPSLLD  $12, BB3, CC3; VPSRLD $20, BB3, BB3; VPXOR CC3, BB3, BB3
	VMOVDQA tmpStoreAVX2, CC3
	polyMulReduceStage

sealAVX2InternalLoopStart:
	VPADDD   BB0, AA0, AA0; VPADDD BB1, AA1, AA1; VPADDD BB2, AA2, AA2; VPADDD BB3, AA3, AA3
	VPXOR    AA0, DD0, DD0; VPXOR AA1, DD1, DD1; VPXOR AA2, DD2, DD2; VPXOR AA3, DD3, DD3
	VPSHUFB  ·rol8<>(SB), DD0, DD0; VPSHUFB ·rol8<>(SB), DD1, DD1; VPSHUFB ·rol8<>(SB), DD2, DD2; VPSHUFB ·rol8<>(SB), DD3, DD3
	polyAdd(2*8(oup))
	VPADDD   DD0, CC0, CC0; VPADDD DD1, CC1, CC1; VPADDD DD2, CC2, CC2; VPADDD DD3, CC3, CC3
	polyMulStage1_AVX2
	VPXOR    CC0, BB0, BB0; VPXOR CC1, BB1, BB1; VPXOR CC2, BB2, BB2; VPXOR CC3, BB3, BB3
	VMOVDQA  CC3, tmpStoreAVX2
	VPSLLD   $7, BB0, CC3; VPSRLD $25, BB0, BB0; VPXOR CC3, BB0, BB0
	VPSLLD   $7, BB1, CC3; VPSRLD $25, BB1, BB1; VPXOR CC3, BB1, BB1
	VPSLLD   $7, BB2, CC3; VPSRLD $25, BB2, BB2; VPXOR CC3, BB2, BB2
	VPSLLD   $7, BB3, CC3; VPSRLD $25, BB3, BB3; VPXOR CC3, BB3, BB3
	VMOVDQA  tmpStoreAVX2, CC3
	polyMulStage2_AVX2
	VPALIGNR $4, BB0, BB0, BB0; VPALIGNR $4, BB1, BB1, BB1; VPALIGNR $4, BB2, BB2, BB2; VPALIGNR $4, BB3, BB3, BB3
	VPALIGNR $8, CC0, CC0, CC0; VPALIGNR $8, CC1, CC1, CC1; VPALIGNR $8, CC2, CC2, CC2; VPALIGNR $8, CC3, CC3, CC3
	VPALIGNR $12, DD0, DD0, DD0; VPALIGNR $12, DD1, DD1, DD1; VPALIGNR $12, DD2, DD2, DD2; VPALIGNR $12, DD3, DD3, DD3
	VPADDD   BB0, AA0, AA0; VPADDD BB1, AA1, AA1; VPADDD BB2, AA2, AA2; VPADDD BB3, AA3, AA3
	polyMulStage3_AVX2
	VPXOR    AA0, DD0, DD0; VPXOR AA1, DD1, DD1; VPXOR AA2, DD2, DD2; VPXOR AA3, DD3, DD3
	VPSHUFB  ·rol16<>(SB), DD0, DD0; VPSHUFB ·rol16<>(SB), DD1, DD1; VPSHUFB ·rol16<>(SB), DD2, DD2; VPSHUFB ·rol16<>(SB), DD3, DD3
	polyMulReduceStage
	VPADDD   DD0, CC0, CC0; VPADDD DD1, CC1, CC1; VPADDD DD2, CC2, CC2; VPADDD DD3, CC3, CC3
	VPXOR    CC0, BB0, BB0; VPXOR CC1, BB1, BB1; VPXOR CC2, BB2, BB2; VPXOR CC3, BB3, BB3
	polyAdd(4*8(oup))
	LEAQ     (6*8)(oup), oup
	VMOVDQA  CC3, tmpStoreAVX2
	VPSLLD   $12, BB0, CC3; VPSRLD $20, BB0, BB0; VPXOR CC3, BB0, BB0
	VPSLLD   $12, BB1, CC3; VPSRLD $20, BB1, BB1; VPXOR CC3, BB1, BB1
	VPSLLD   $12, BB2, CC3; VPSRLD $20, BB2, BB2; VPXOR CC3, BB2, BB2
	VPSLLD   $12, BB3, CC3; VPSRLD $20, BB3, BB3; VPXOR CC3, BB3, BB3
	VMOVDQA  tmpStoreAVX2, CC3
	polyMulStage1_AVX2
	VPADDD   BB0, AA0, AA0; VPADDD BB1, AA1, AA1; VPADDD BB2, AA2, AA2; VPADDD BB3, AA3, AA3
	VPXOR    AA0, DD0, DD0; VPXOR AA1, DD1, DD1; VPXOR AA2, DD2, DD2; VPXOR AA3, DD3, DD3
	polyMulStage2_AVX2
	VPSHUFB  ·rol8<>(SB), DD0, DD0; VPSHUFB ·rol8<>(SB), DD1, DD1; VPSHUFB ·rol8<>(SB), DD2, DD2; VPSHUFB ·rol8<>(SB), DD3, DD3
	VPADDD   DD0, CC0, CC0; VPADDD DD1, CC1, CC1; VPADDD DD2, CC2, CC2; VPADDD DD3, CC3, CC3
	polyMulStage3_AVX2
	VPXOR    CC0, BB0, BB0; VPXOR CC1, BB1, BB1; VPXOR CC2, BB2, BB2; VPXOR CC3, BB3, BB3
	VMOVDQA  CC3, tmpStoreAVX2
	VPSLLD   $7, BB0, CC3; VPSRLD $25, BB0, BB0; VPXOR CC3, BB0, BB0
	VPSLLD   $7, BB1, CC3; VPSRLD $25, BB1, BB1; VPXOR CC3, BB1, BB1
	VPSLLD   $7, BB2, CC3; VPSRLD $25, BB2, BB2; VPXOR CC3, BB2, BB2
	VPSLLD   $7, BB3, CC3; VPSRLD $25, BB3, BB3; VPXOR CC3, BB3, BB3
	VMOVDQA  tmpStoreAVX2, CC3
	polyMulReduceStage
	VPALIGNR $12, BB0, BB0, BB0; VPALIGNR $12, BB1, BB1, BB1; VPALIGNR $12, BB2, BB2, BB2; VPALIGNR $12, BB3, BB3, BB3
	VPALIGNR $8, CC0, CC0, CC0; VPALIGNR $8, CC1, CC1, CC1; VPALIGNR $8, CC2, CC2, CC2; VPALIGNR $8, CC3, CC3, CC3
	VPALIGNR $4, DD0, DD0, DD0; VPALIGNR $4, DD1, DD1, DD1; VPALIGNR $4, DD2, DD2, DD2; VPALIGNR $4, DD3, DD3, DD3
	DECQ     itr1
	JNE      sealAVX2InternalLoop

	VPADDD  ·chacha20Constants<>(SB), AA0, AA0; VPADDD ·chacha20Constants<>(SB), AA1, AA1; VPADDD ·chacha20Constants<>(SB), AA2, AA2; VPADDD ·chacha20Constants<>(SB), AA3, AA3
	VPADDD  state1StoreAVX2, BB0, BB0; VPADDD state1StoreAVX2, BB1, BB1; VPADDD state1StoreAVX2, BB2, BB2; VPADDD state1StoreAVX2, BB3, BB3
	VPADDD  state2StoreAVX2, CC0, CC0; VPADDD state2StoreAVX2, CC1, CC1; VPADDD state2StoreAVX2, CC2, CC2; VPADDD state2StoreAVX2, CC3, CC3
	VPADDD  ctr0StoreAVX2, DD0, DD0; VPADDD ctr1StoreAVX2, DD1, DD1; VPADDD ctr2StoreAVX2, DD2, DD2; VPADDD ctr3StoreAVX2, DD3, DD3
	VMOVDQA CC3, tmpStoreAVX2

	// We only hashed 480 of the 512 bytes available - hash the remaining 32 here
	polyAdd(0*8(oup))
	polyMulAVX2
	LEAQ       (4*8)(oup), oup
	VPERM2I128 $0x02, AA0, BB0, CC3; VPERM2I128 $0x13, AA0, BB0, BB0; VPERM2I128 $0x02, CC0, DD0, AA0; VPERM2I128 $0x13, CC0, DD0, CC0
	VPXOR      (0*32)(inp), CC3, CC3; VPXOR (1*32)(inp), AA0, AA0; VPXOR (2*32)(inp), BB0, BB0; VPXOR (3*32)(inp), CC0, CC0
	VMOVDQU    CC3, (0*32)(oup); VMOVDQU AA0, (1*32)(oup); VMOVDQU BB0, (2*32)(oup); VMOVDQU CC0, (3*32)(oup)
	VPERM2I128 $0x02, AA1, BB1, AA0; VPERM2I128 $0x02, CC1, DD1, BB0; VPERM2I128 $0x13, AA1, BB1, CC0; VPERM2I128 $0x13, CC1, DD1, DD0
	VPXOR      (4*32)(inp), AA0, AA0; VPXOR (5*32)(inp), BB0, BB0; VPXOR (6*32)(inp), CC0, CC0; VPXOR (7*32)(inp), DD0, DD0
	VMOVDQU    AA0, (4*32)(oup); VMOVDQU BB0, (5*32)(oup); VMOVDQU CC0, (6*32)(oup); VMOVDQU DD0, (7*32)(oup)

	// and here
	polyAdd(-2*8(oup))
	polyMulAVX2
	VPERM2I128 $0x02, AA2, BB2, AA0; VPERM2I128 $0x02, CC2, DD2, BB0; VPERM2I128 $0x13, AA2, BB2, CC0; VPERM2I128 $0x13, CC2, DD2, DD0
	VPXOR      (8*32)(inp), AA0, AA0; VPXOR (9*32)(inp), BB0, BB0; VPXOR (10*32)(inp), CC0, CC0; VPXOR (11*32)(inp), DD0, DD0
	VMOVDQU    AA0, (8*32)(oup); VMOVDQU BB0, (9*32)(oup); VMOVDQU CC0, (10*32)(oup); VMOVDQU DD0, (11*32)(oup)
	VPERM2I128 $0x02, AA3, BB3, AA0; VPERM2I128 $0x02, tmpStoreAVX2, DD3, BB0; VPERM2I128 $0x13, AA3, BB3, CC0; VPERM2I128 $0x13, tmpStoreAVX2, DD3, DD0
	VPXOR      (12*32)(inp), AA0, AA0; VPXOR (13*32)(inp), BB0, BB0; VPXOR (14*32)(inp), CC0, CC0; VPXOR (15*32)(inp), DD0, DD0
	VMOVDQU    AA0, (12*32)(oup); VMOVDQU BB0, (13*32)(oup); VMOVDQU CC0, (14*32)(oup); VMOVDQU DD0, (15*32)(oup)
	LEAQ       (32*16)(inp), inp
	SUBQ       $(32*16), inl
	CMPQ       inl, $512
	JG         sealAVX2MainLoop

	// Tail can only hash 480 bytes
	polyAdd(0*8(oup))
	polyMulAVX2
	polyAdd(2*8(oup))
	polyMulAVX2
	LEAQ 32(oup), oup

	MOVQ $10, itr1
	MOVQ $0, itr2
	CMPQ inl, $128
	JBE  sealAVX2Tail128
	CMPQ inl, $256
	JBE  sealAVX2Tail256
	CMPQ inl, $384
	JBE  sealAVX2Tail384
	JMP  sealAVX2Tail512

// ----------------------------------------------------------------------------
// Special optimization for buffers smaller than 193 bytes
seal192AVX2:
	// For up to 192 bytes of ciphertext and 64 bytes for the poly key, we process four blocks
	VMOVDQA AA0, AA1
	VMOVDQA BB0, BB1
	VMOVDQA CC0, CC1
	VPADDD  ·avx2IncMask<>(SB), DD0, DD1
	VMOVDQA AA0, AA2
	VMOVDQA BB0, BB2
	VMOVDQA CC0, CC2
	VMOVDQA DD0, DD2
	VMOVDQA DD1, TT3
	MOVQ    $10, itr2

sealAVX2192InnerCipherLoop:
	chachaQR_AVX2(AA0, BB0, CC0, DD0, TT0); chachaQR_AVX2(AA1, BB1, CC1, DD1, TT0)
	VPALIGNR   $4, BB0, BB0, BB0; VPALIGNR $4, BB1, BB1, BB1
	VPALIGNR   $8, CC0, CC0, CC0; VPALIGNR $8, CC1, CC1, CC1
	VPALIGNR   $12, DD0, DD0, DD0; VPALIGNR $12, DD1, DD1, DD1
	chachaQR_AVX2(AA0, BB0, CC0, DD0, TT0); chachaQR_AVX2(AA1, BB1, CC1, DD1, TT0)
	VPALIGNR   $12, BB0, BB0, BB0; VPALIGNR $12, BB1, BB1, BB1
	VPALIGNR   $8, CC0, CC0, CC0; VPALIGNR $8, CC1, CC1, CC1
	VPALIGNR   $4, DD0, DD0, DD0; VPALIGNR $4, DD1, DD1, DD1
	DECQ       itr2
	JNE        sealAVX2192InnerCipherLoop
	VPADDD     AA2, AA0, AA0; VPADDD AA2, AA1, AA1
	VPADDD     BB2, BB0, BB0; VPADDD BB2, BB1, BB1
	VPADDD     CC2, CC0, CC0; VPADDD CC2, CC1, CC1
	VPADDD     DD2, DD0, DD0; VPADDD TT3, DD1, DD1
	VPERM2I128 $0x02, AA0, BB0, TT0

	// Clamp and store poly key
	VPAND   ·polyClampMask<>(SB), TT0, TT0
	VMOVDQA TT0, rsStoreAVX2

	// Stream for up to 192 bytes
	VPERM2I128 $0x13, AA0, BB0, AA0
	VPERM2I128 $0x13, CC0, DD0, BB0
	VPERM2I128 $0x02, AA1, BB1, CC0
	VPERM2I128 $0x02, CC1, DD1, DD0
	VPERM2I128 $0x13, AA1, BB1, AA1
	VPERM2I128 $0x13, CC1, DD1, BB1

sealAVX2ShortSeal:
	// Hash aad
	MOVQ ad_len+80(FP), itr2
	CALL polyHashADInternal<>(SB)
	XORQ itr1, itr1

sealAVX2SealHash:
	// itr1 holds the number of bytes encrypted but not yet hashed
	CMPQ itr1, $16
	JB   sealAVX2ShortSealLoop
	polyAdd(0(oup))
	polyMul
	SUBQ $16, itr1
	ADDQ $16, oup
	JMP  sealAVX2SealHash

sealAVX2ShortSealLoop:
	CMPQ inl, $32
	JB   sealAVX2ShortTail32
	SUBQ $32, inl

	// Load for encryption
	VPXOR   (inp), AA0, AA0
	VMOVDQU AA0, (oup)
	LEAQ    (1*32)(inp), inp

	// Now can hash
	polyAdd(0*8(oup))
	polyMulAVX2
	polyAdd(2*8(oup))
	polyMulAVX2
	LEAQ (1*32)(oup), oup

	// Shift stream left
	VMOVDQA BB0, AA0
	VMOVDQA CC0, BB0
	VMOVDQA DD0, CC0
	VMOVDQA AA1, DD0
	VMOVDQA BB1, AA1
	VMOVDQA CC1, BB1
	VMOVDQA DD1, CC1
	VMOVDQA AA2, DD1
	VMOVDQA BB2, AA2
	JMP     sealAVX2ShortSealLoop

sealAVX2ShortTail32:
	CMPQ    inl, $16
	VMOVDQA A0, A1
	JB      sealAVX2ShortDone

	SUBQ $16, inl

	// Load for encryption
	VPXOR   (inp), A0, T0
	VMOVDQU T0, (oup)
	LEAQ    (1*16)(inp), inp

	// Hash
	polyAdd(0*8(oup))
	polyMulAVX2
	LEAQ       (1*16)(oup), oup
	VPERM2I128 $0x11, AA0, AA0, AA0
	VMOVDQA    A0, A1

sealAVX2ShortDone:
	VZEROUPPER
	JMP sealSSETail

// ----------------------------------------------------------------------------
// Special optimization for buffers smaller than 321 bytes
seal320AVX2:
	// For up to 320 bytes of ciphertext and 64 bytes for the poly key, we process six blocks
	VMOVDQA AA0, AA1; VMOVDQA BB0, BB1; VMOVDQA CC0, CC1; VPADDD ·avx2IncMask<>(SB), DD0, DD1
	VMOVDQA AA0, AA2; VMOVDQA BB0, BB2; VMOVDQA CC0, CC2; VPADDD ·avx2IncMask<>(SB), DD1, DD2
	VMOVDQA BB0, TT1; VMOVDQA CC0, TT2; VMOVDQA DD0, TT3
	MOVQ    $10, itr2

sealAVX2320InnerCipherLoop:
	chachaQR_AVX2(AA0, BB0, CC0, DD0, TT0); chachaQR_AVX2(AA1, BB1, CC1, DD1, TT0); chachaQR_AVX2(AA2, BB2, CC2, DD2, TT0)
	VPALIGNR $4, BB0, BB0, BB0; VPALIGNR $4, BB1, BB1, BB1; VPALIGNR $4, BB2, BB2, BB2
	VPALIGNR $8, CC0, CC0, CC0; VPALIGNR $8, CC1, CC1, CC1; VPALIGNR $8, CC2, CC2, CC2
	VPALIGNR $12, DD0, DD0, DD0; VPALIGNR $12, DD1, DD1, DD1; VPALIGNR $12, DD2, DD2, DD2
	chachaQR_AVX2(AA0, BB0, CC0, DD0, TT0); chachaQR_AVX2(AA1, BB1, CC1, DD1, TT0); chachaQR_AVX2(AA2, BB2, CC2, DD2, TT0)
	VPALIGNR $12, BB0, BB0, BB0; VPALIGNR $12, BB1, BB1, BB1; VPALIGNR $12, BB2, BB2, BB2
	VPALIGNR $8, CC0, CC0, CC0; VPALIGNR $8, CC1, CC1, CC1; VPALIGNR $8, CC2, CC2, CC2
	VPALIGNR $4, DD0, DD0, DD0; VPALIGNR $4, DD1, DD1, DD1; VPALIGNR $4, DD2, DD2, DD2
	DECQ     itr2
	JNE      sealAVX2320InnerCipherLoop

	VMOVDQA ·chacha20Constants<>(SB), TT0
	VPADDD  TT0, AA0, AA0; VPADDD TT0, AA1, AA1; VPADDD TT0, AA2, AA2
	VPADDD  TT1, BB0, BB0; VPADDD TT1, BB1, BB1; VPADDD TT1, BB2, BB2
	VPADDD  TT2, CC0, CC0; VPADDD TT2, CC1, CC1; VPADDD TT2, CC2, CC2
	VMOVDQA ·avx2IncMask<>(SB), TT0
	VPADDD  TT3, DD0, DD0; VPADDD TT0, TT3, TT3
	VPADDD  TT3, DD1, DD1; VPADDD TT0, TT3, TT3
	VPADDD  TT3, DD2, DD2

	// Clamp and store poly key
	VPERM2I128 $0x02, AA0, BB0, TT0
	VPAND      ·polyClampMask<>(SB), TT0, TT0
	VMOVDQA    TT0, rsStoreAVX2

	// Stream for up to 320 bytes
	VPERM2I128 $0x13, AA0, BB0, AA0
	VPERM2I128 $0x13, CC0, DD0, BB0
	VPERM2I128 $0x02, AA1, BB1, CC0
	VPERM2I128 $0x02, CC1, DD1, DD0
	VPERM2I128 $0x13, AA1, BB1, AA1
	VPERM2I128 $0x13, CC1, DD1, BB1
	VPERM2I128 $0x02, AA2, BB2, CC1
	VPERM2I128 $0x02, CC2, DD2, DD1
	VPERM2I128 $0x13, AA2, BB2, AA2
	VPERM2I128 $0x13, CC2, DD2, BB2
	JMP        sealAVX2ShortSeal

// ----------------------------------------------------------------------------
// Special optimization for the last 128 bytes of ciphertext
sealAVX2Tail128:
	// Need to decrypt up to 128 bytes - prepare two blocks
	// If we got here after the main loop - there are 512 encrypted bytes waiting to be hashed
	// If we got here before the main loop - there are 448 encrpyred bytes waiting to be hashed
	VMOVDQA ·chacha20Constants<>(SB), AA0
	VMOVDQA state1StoreAVX2, BB0
	VMOVDQA state2StoreAVX2, CC0
	VMOVDQA ctr3StoreAVX2, DD0
	VPADDD  ·avx2IncMask<>(SB), DD0, DD0
	VMOVDQA DD0, DD1

sealAVX2Tail128LoopA:
	polyAdd(0(oup))
	polyMul
	LEAQ 16(oup), oup

sealAVX2Tail128LoopB:
	chachaQR_AVX2(AA0, BB0, CC0, DD0, TT0)
	polyAdd(0(oup))
	polyMul
	VPALIGNR $4, BB0, BB0, BB0
	VPALIGNR $8, CC0, CC0, CC0
	VPALIGNR $12, DD0, DD0, DD0
	chachaQR_AVX2(AA0, BB0, CC0, DD0, TT0)
	polyAdd(16(oup))
	polyMul
	LEAQ     32(oup), oup
	VPALIGNR $12, BB0, BB0, BB0
	VPALIGNR $8, CC0, CC0, CC0
	VPALIGNR $4, DD0, DD0, DD0
	DECQ     itr1
	JG       sealAVX2Tail128LoopA
	DECQ     itr2
	JGE      sealAVX2Tail128LoopB

	VPADDD ·chacha20Constants<>(SB), AA0, AA1
	VPADDD state1StoreAVX2, BB0, BB1
	VPADDD state2StoreAVX2, CC0, CC1
	VPADDD DD1, DD0, DD1

	VPERM2I128 $0x02, AA1, BB1, AA0
	VPERM2I128 $0x02, CC1, DD1, BB0
	VPERM2I128 $0x13, AA1, BB1, CC0
	VPERM2I128 $0x13, CC1, DD1, DD0
	JMP        sealAVX2ShortSealLoop

// ----------------------------------------------------------------------------
// Special optimization for the last 256 bytes of ciphertext
sealAVX2Tail256:
	// Need to decrypt up to 256 bytes - prepare two blocks
	// If we got here after the main loop - there are 512 encrypted bytes waiting to be hashed
	// If we got here before the main loop - there are 448 encrpyred bytes waiting to be hashed
	VMOVDQA ·chacha20Constants<>(SB), AA0; VMOVDQA ·chacha20Constants<>(SB), AA1
	VMOVDQA state1StoreAVX2, BB0; VMOVDQA state1StoreAVX2, BB1
	VMOVDQA state2StoreAVX2, CC0; VMOVDQA state2StoreAVX2, CC1
	VMOVDQA ctr3StoreAVX2, DD0
	VPADDD  ·avx2IncMask<>(SB), DD0, DD0
	VPADDD  ·avx2IncMask<>(SB), DD0, DD1
	VMOVDQA DD0, TT1
	VMOVDQA DD1, TT2

sealAVX2Tail256LoopA:
	polyAdd(0(oup))
	polyMul
	LEAQ 16(oup), oup

sealAVX2Tail256LoopB:
	chachaQR_AVX2(AA0, BB0, CC0, DD0, TT0); chachaQR_AVX2(AA1, BB1, CC1, DD1, TT0)
	polyAdd(0(oup))
	polyMul
	VPALIGNR $4, BB0, BB0, BB0; VPALIGNR $4, BB1, BB1, BB1
	VPALIGNR $8, CC0, CC0, CC0; VPALIGNR $8, CC1, CC1, CC1
	VPALIGNR $12, DD0, DD0, DD0; VPALIGNR $12, DD1, DD1, DD1
	chachaQR_AVX2(AA0, BB0, CC0, DD0, TT0); chachaQR_AVX2(AA1, BB1, CC1, DD1, TT0)
	polyAdd(16(oup))
	polyMul
	LEAQ     32(oup), oup
	VPALIGNR $12, BB0, BB0, BB0; VPALIGNR $12, BB1, BB1, BB1
	VPALIGNR $8, CC0, CC0, CC0; VPALIGNR $8, CC1, CC1, CC1
	VPALIGNR $4, DD0, DD0, DD0; VPALIGNR $4, DD1, DD1, DD1
	DECQ     itr1
	JG       sealAVX2Tail256LoopA
	DECQ     itr2
	JGE      sealAVX2Tail256LoopB

	VPADDD     ·chacha20Constants<>(SB), AA0, AA0; VPADDD ·chacha20Constants<>(SB), AA1, AA1
	VPADDD     state1StoreAVX2, BB0, BB0; VPADDD state1StoreAVX2, BB1, BB1
	VPADDD     state2StoreAVX2, CC0, CC0; VPADDD state2StoreAVX2, CC1, CC1
	VPADDD     TT1, DD0, DD0; VPADDD TT2, DD1, DD1
	VPERM2I128 $0x02, AA0, BB0, TT0
	VPERM2I128 $0x02, CC0, DD0, TT1
	VPERM2I128 $0x13, AA0, BB0, TT2
	VPERM2I128 $0x13, CC0, DD0, TT3
	VPXOR      (0*32)(inp), TT0, TT0; VPXOR (1*32)(inp), TT1, TT1; VPXOR (2*32)(inp), TT2, TT2; VPXOR (3*32)(inp), TT3, TT3
	VMOVDQU    TT0, (0*32)(oup); VMOVDQU TT1, (1*32)(oup); VMOVDQU TT2, (2*32)(oup); VMOVDQU TT3, (3*32)(oup)
	MOVQ       $128, itr1
	LEAQ       128(inp), inp
	SUBQ       $128, inl
	VPERM2I128 $0x02, AA1, BB1, AA0
	VPERM2I128 $0x02, CC1, DD1, BB0
	VPERM2I128 $0x13, AA1, BB1, CC0
	VPERM2I128 $0x13, CC1, DD1, DD0

	JMP sealAVX2SealHash

// ----------------------------------------------------------------------------
// Special optimization for the last 384 bytes of ciphertext
sealAVX2Tail384:
	// Need to decrypt up to 384 bytes - prepare two blocks
	// If we got here after the main loop - there are 512 encrypted bytes waiting to be hashed
	// If we got here before the main loop - there are 448 encrpyred bytes waiting to be hashed
	VMOVDQA ·chacha20Constants<>(SB), AA0; VMOVDQA AA0, AA1; VMOVDQA AA0, AA2
	VMOVDQA state1StoreAVX2, BB0; VMOVDQA BB0, BB1; VMOVDQA BB0, BB2
	VMOVDQA state2StoreAVX2, CC0; VMOVDQA CC0, CC1; VMOVDQA CC0, CC2
	VMOVDQA ctr3StoreAVX2, DD0
	VPADDD  ·avx2IncMask<>(SB), DD0, DD0; VPADDD ·avx2IncMask<>(SB), DD0, DD1; VPADDD ·avx2IncMask<>(SB), DD1, DD2
	VMOVDQA DD0, TT1; VMOVDQA DD1, TT2; VMOVDQA DD2, TT3

sealAVX2Tail384LoopA:
	polyAdd(0(oup))
	polyMul
	LEAQ 16(oup), oup

sealAVX2Tail384LoopB:
	chachaQR_AVX2(AA0, BB0, CC0, DD0, TT0); chachaQR_AVX2(AA1, BB1, CC1, DD1, TT0); chachaQR_AVX2(AA2, BB2, CC2, DD2, TT0)
	polyAdd(0(oup))
	polyMul
	VPALIGNR $4, BB0, BB0, BB0; VPALIGNR $4, BB1, BB1, BB1; VPALIGNR $4, BB2, BB2, BB2
	VPALIGNR $8, CC0, CC0, CC0; VPALIGNR $8, CC1, CC1, CC1; VPALIGNR $8, CC2, CC2, CC2
	VPALIGNR $12, DD0, DD0, DD0; VPALIGNR $12, DD1, DD1, DD1; VPALIGNR $12, DD2, DD2, DD2
	chachaQR_AVX2(AA0, BB0, CC0, DD0, TT0); chachaQR_AVX2(AA1, BB1, CC1, DD1, TT0); chachaQR_AVX2(AA2, BB2, CC2, DD2, TT0)
	polyAdd(16(oup))
	polyMul
	LEAQ     32(oup), oup
	VPALIGNR $12, BB0, BB0, BB0; VPALIGNR $12, BB1, BB1, BB1; VPALIGNR $12, BB2, BB2, BB2
	VPALIGNR $8, CC0, CC0, CC0; VPALIGNR $8, CC1, CC1, CC1; VPALIGNR $8, CC2, CC2, CC2
	VPALIGNR $4, DD0, DD0, DD0; VPALIGNR $4, DD1, DD1, DD1; VPALIGNR $4, DD2, DD2, DD2
	DECQ     itr1
	JG       sealAVX2Tail384LoopA
	DECQ     itr2
	JGE      sealAVX2Tail384LoopB

	VPADDD     ·chacha20Constants<>(SB), AA0, AA0; VPADDD ·chacha20Constants<>(SB), AA1, AA1; VPADDD ·chacha20Constants<>(SB), AA2, AA2
	VPADDD     state1StoreAVX2, BB0, BB0; VPADDD state1StoreAVX2, BB1, BB1; VPADDD state1StoreAVX2, BB2, BB2
	VPADDD     state2StoreAVX2, CC0, CC0; VPADDD state2StoreAVX2, CC1, CC1; VPADDD state2StoreAVX2, CC2, CC2
	VPADDD     TT1, DD0, DD0; VPADDD TT2, DD1, DD1; VPADDD TT3, DD2, DD2
	VPERM2I128 $0x02, AA0, BB0, TT0
	VPERM2I128 $0x02, CC0, DD0, TT1
	VPERM2I128 $0x13, AA0, BB0, TT2
	VPERM2I128 $0x13, CC0, DD0, TT3
	VPXOR      (0*32)(inp), TT0, TT0; VPXOR (1*32)(inp), TT1, TT1; VPXOR (2*32)(inp), TT2, TT2; VPXOR (3*32)(inp), TT3, TT3
	VMOVDQU    TT0, (0*32)(oup); VMOVDQU TT1, (1*32)(oup); VMOVDQU TT2, (2*32)(oup); VMOVDQU TT3, (3*32)(oup)
	VPERM2I128 $0x02, AA1, BB1, TT0
	VPERM2I128 $0x02, CC1, DD1, TT1
	VPERM2I128 $0x13, AA1, BB1, TT2
	VPERM2I128 $0x13, CC1, DD1, TT3
	VPXOR      (4*32)(inp), TT0, TT0; VPXOR (5*32)(inp), TT1, TT1; VPXOR (6*32)(inp), TT2, TT2; VPXOR (7*32)(inp), TT3, TT3
	VMOVDQU    TT0, (4*32)(oup); VMOVDQU TT1, (5*32)(oup); VMOVDQU TT2, (6*32)(oup); VMOVDQU TT3, (7*32)(oup)
	MOVQ       $256, itr1
	LEAQ       256(inp), inp
	SUBQ       $256, inl
	VPERM2I128 $0x02, AA2, BB2, AA0
	VPERM2I128 $0x02, CC2, DD2, BB0
	VPERM2I128 $0x13, AA2, BB2, CC0
	VPERM2I128 $0x13, CC2, DD2, DD0

	JMP sealAVX2SealHash

// ----------------------------------------------------------------------------
// Special optimization for the last 512 bytes of ciphertext
sealAVX2Tail512:
	// Need to decrypt up to 512 bytes - prepare two blocks
	// If we got here after the main loop - there are 512 encrypted bytes waiting to be hashed
	// If we got here before the main loop - there are 448 encrpyred bytes waiting to be hashed
	VMOVDQA ·chacha20Constants<>(SB), AA0; VMOVDQA AA0, AA1; VMOVDQA AA0, AA2; VMOVDQA AA0, AA3
	VMOVDQA state1StoreAVX2, BB0; VMOVDQA BB0, BB1; VMOVDQA BB0, BB2; VMOVDQA BB0, BB3
	VMOVDQA state2StoreAVX2, CC0; VMOVDQA CC0, CC1; VMOVDQA CC0, CC2; VMOVDQA CC0, CC3
	VMOVDQA ctr3StoreAVX2, DD0
	VPADDD  ·avx2IncMask<>(SB), DD0, DD0; VPADDD ·avx2IncMask<>(SB), DD0, DD1; VPADDD ·avx2IncMask<>(SB), DD1, DD2; VPADDD ·avx2IncMask<>(SB), DD2, DD3
	VMOVDQA DD0, ctr0StoreAVX2; VMOVDQA DD1, ctr1StoreAVX2; VMOVDQA DD2, ctr2StoreAVX2; VMOVDQA DD3, ctr3StoreAVX2

sealAVX2Tail512LoopA:
	polyAdd(0(oup))
	polyMul
	LEAQ 16(oup), oup

sealAVX2Tail512LoopB:
	VPADDD   BB0, AA0, AA0; VPADDD BB1, AA1, AA1; VPADDD BB2, AA2, AA2; VPADDD BB3, AA3, AA3
	VPXOR    AA0, DD0, DD0; VPXOR AA1, DD1, DD1; VPXOR AA2, DD2, DD2; VPXOR AA3, DD3, DD3
	VPSHUFB  ·rol16<>(SB), DD0, DD0; VPSHUFB ·rol16<>(SB), DD1, DD1; VPSHUFB ·rol16<>(SB), DD2, DD2; VPSHUFB ·rol16<>(SB), DD3, DD3
	VPADDD   DD0, CC0, CC0; VPADDD DD1, CC1, CC1; VPADDD DD2, CC2, CC2; VPADDD DD3, CC3, CC3
	VPXOR    CC0, BB0, BB0; VPXOR CC1, BB1, BB1; VPXOR CC2, BB2, BB2; VPXOR CC3, BB3, BB3
	VMOVDQA  CC3, tmpStoreAVX2
	VPSLLD   $12, BB0, CC3; VPSRLD $20, BB0, BB0; VPXOR CC3, BB0, BB0
	VPSLLD   $12, BB1, CC3; VPSRLD $20, BB1, BB1; VPXOR CC3, BB1, BB1
	VPSLLD   $12, BB2, CC3; VPSRLD $20, BB2, BB2; VPXOR CC3, BB2, BB2
	VPSLLD   $12, BB3, CC3; VPSRLD $20, BB3, BB3; VPXOR CC3, BB3, BB3
	VMOVDQA  tmpStoreAVX2, CC3
	polyAdd(0*8(oup))
	polyMulAVX2
	VPADDD   BB0, AA0, AA0; VPADDD BB1, AA1, AA1; VPADDD BB2, AA2, AA2; VPADDD BB3, AA3, AA3
	VPXOR    AA0, DD0, DD0; VPXOR AA1, DD1, DD1; VPXOR AA2, DD2, DD2; VPXOR AA3, DD3, DD3
	VPSHUFB  ·rol8<>(SB), DD0, DD0; VPSHUFB ·rol8<>(SB), DD1, DD1; VPSHUFB ·rol8<>(SB), DD2, DD2; VPSHUFB ·rol8<>(SB), DD3, DD3
	VPADDD   DD0, CC0, CC0; VPADDD DD1, CC1, CC1; VPADDD DD2, CC2, CC2; VPADDD DD3, CC3, CC3
	VPXOR    CC0, BB0, BB0; VPXOR CC1, BB1, BB1; VPXOR CC2, BB2, BB2; VPXOR CC3, BB3, BB3
	VMOVDQA  CC3, tmpStoreAVX2
	VPSLLD   $7, BB0, CC3; VPSRLD $25, BB0, BB0; VPXOR CC3, BB0, BB0
	VPSLLD   $7, BB1, CC3; VPSRLD $25, BB1, BB1; VPXOR CC3, BB1, BB1
	VPSLLD   $7, BB2, CC3; VPSRLD $25, BB2, BB2; VPXOR CC3, BB2, BB2
	VPSLLD   $7, BB3, CC3; VPSRLD $25, BB3, BB3; VPXOR CC3, BB3, BB3
	VMOVDQA  tmpStoreAVX2, CC3
	VPALIGNR $4, BB0, BB0, BB0; VPALIGNR $4, BB1, BB1, BB1; VPALIGNR $4, BB2, BB2, BB2; VPALIGNR $4, BB3, BB3, BB3
	VPALIGNR $8, CC0, CC0, CC0; VPALIGNR $8, CC1, CC1, CC1; VPALIGNR $8, CC2, CC2, CC2; VPALIGNR $8, CC3, CC3, CC3
	VPALIGNR $12, DD0, DD0, DD0; VPALIGNR $12, DD1, DD1, DD1; VPALIGNR $12, DD2, DD2, DD2; VPALIGNR $12, DD3, DD3, DD3
	VPADDD   BB0, AA0, AA0; VPADDD BB1, AA1, AA1; VPADDD BB2, AA2, AA2; VPADDD BB3, AA3, AA3
	VPXOR    AA0, DD0, DD0; VPXOR AA1, DD1, DD1; VPXOR AA2, DD2, DD2; VPXOR AA3, DD3, DD3
	VPSHUFB  ·rol16<>(SB), DD0, DD0; VPSHUFB ·rol16<>(SB), DD1, DD1; VPSHUFB ·rol16<>(SB), DD2, DD2; VPSHUFB ·rol16<>(SB), DD3, DD3
	VPADDD   DD0, CC0, CC0; VPADDD DD1, CC1, CC1; VPADDD DD2, CC2, CC2; VPADDD DD3, CC3, CC3
	VPXOR    CC0, BB0, BB0; VPXOR CC1, BB1, BB1; VPXOR CC2, BB2, BB2; VPXOR CC3, BB3, BB3
	polyAdd(2*8(oup))
	polyMulAVX2
	LEAQ     (4*8)(oup), oup
	VMOVDQA  CC3, tmpStoreAVX2
	VPSLLD   $12, BB0, CC3; VPSRLD $20, BB0, BB0; VPXOR CC3, BB0, BB0
	VPSLLD   $12, BB1, CC3; VPSRLD $20, BB1, BB1; VPXOR CC3, BB1, BB1
	VPSLLD   $12, BB2, CC3; VPSRLD $20, BB2, BB2; VPXOR CC3, BB2, BB2
	VPSLLD   $12, BB3, CC3; VPSRLD $20, BB3, BB3; VPXOR CC3, BB3, BB3
	VMOVDQA  tmpStoreAVX2, CC3
	VPADDD   BB0, AA0, AA0; VPADDD BB1, AA1, AA1; VPADDD BB2, AA2, AA2; VPADDD BB3, AA3, AA3
	VPXOR    AA0, DD0, DD0; VPXOR AA1, DD1, DD1; VPXOR AA2, DD2, DD2; VPXOR AA3, DD3, DD3
	VPSHUFB  ·rol8<>(SB), DD0, DD0; VPSHUFB ·rol8<>(SB), DD1, DD1; VPSHUFB ·rol8<>(SB), DD2, DD2; VPSHUFB ·rol8<>(SB), DD3, DD3
	VPADDD   DD0, CC0, CC0; VPADDD DD1, CC1, CC1; VPADDD DD2, CC2, CC2; VPADDD DD3, CC3, CC3
	VPXOR    CC0, BB0, BB0; VPXOR CC1, BB1, BB1; VPXOR CC2, BB2, BB2; VPXOR CC3, BB3, BB3
	VMOVDQA  CC3, tmpStoreAVX2
	VPSLLD   $7, BB0, CC3; VPSRLD $25, BB0, BB0; VPXOR CC3, BB0, BB0
	VPSLLD   $7, BB1, CC3; VPSRLD $25, BB1, BB1; VPXOR CC3, BB1, BB1
	VPSLLD   $7, BB2, CC3; VPSRLD $25, BB2, BB2; VPXOR CC3, BB2, BB2
	VPSLLD   $7, BB3, CC3; VPSRLD $25, BB3, BB3; VPXOR CC3, BB3, BB3
	VMOVDQA  tmpStoreAVX2, CC3
	VPALIGNR $12, BB0, BB0, BB0; VPALIGNR $12, BB1, BB1, BB1; VPALIGNR $12, BB2, BB2, BB2; VPALIGNR $12, BB3, BB3, BB3
	VPALIGNR $8, CC0, CC0, CC0; VPALIGNR $8, CC1, CC1, CC1; VPALIGNR $8, CC2, CC2, CC2; VPALIGNR $8, CC3, CC3, CC3
	VPALIGNR $4, DD0, DD0, DD0; VPALIGNR $4, DD1, DD1, DD1; VPALIGNR $4, DD2, DD2, DD2; VPALIGNR $4, DD3, DD3, DD3

	DECQ itr1
	JG   sealAVX2Tail512LoopA
	DECQ itr2
	JGE  sealAVX2Tail512LoopB

	VPADDD     ·chacha20Constants<>(SB), AA0, AA0; VPADDD ·chacha20Constants<>(SB), AA1, AA1; VPADDD ·chacha20Constants<>(SB), AA2, AA2; VPADDD ·chacha20Constants<>(SB), AA3, AA3
	VPADDD     state1StoreAVX2, BB0, BB0; VPADDD state1StoreAVX2, BB1, BB1; VPADDD state1StoreAVX2, BB2, BB2; VPADDD state1StoreAVX2, BB3, BB3
	VPADDD     state2StoreAVX2, CC0, CC0; VPADDD state2StoreAVX2, CC1, CC1; VPADDD state2StoreAVX2, CC2, CC2; VPADDD state2StoreAVX2, CC3, CC3
	VPADDD     ctr0StoreAVX2, DD0, DD0; VPADDD ctr1StoreAVX2, DD1, DD1; VPADDD ctr2StoreAVX2, DD2, DD2; VPADDD ctr3StoreAVX2, DD3, DD3
	VMOVDQA    CC3, tmpStoreAVX2
	VPERM2I128 $0x02, AA0, BB0, CC3
	VPXOR      (0*32)(inp), CC3, CC3
	VMOVDQU    CC3, (0*32)(oup)
	VPERM2I128 $0x02, CC0, DD0, CC3
	VPXOR      (1*32)(inp), CC3, CC3
	VMOVDQU    CC3, (1*32)(oup)
	VPERM2I128 $0x13, AA0, BB0, CC3
	VPXOR      (2*32)(inp), CC3, CC3
	VMOVDQU    CC3, (2*32)(oup)
	VPERM2I128 $0x13, CC0, DD0, CC3
	VPXOR      (3*32)(inp), CC3, CC3
	VMOVDQU    CC3, (3*32)(oup)

	VPERM2I128 $0x02, AA1, BB1, AA0
	VPERM2I128 $0x02, CC1, DD1, BB0
	VPERM2I128 $0x13, AA1, BB1, CC0
	VPERM2I128 $0x13, CC1, DD1, DD0
	VPXOR      (4*32)(inp), AA0, AA0; VPXOR (5*32)(inp), BB0, BB0; VPXOR (6*32)(inp), CC0, CC0; VPXOR (7*32)(inp), DD0, DD0
	VMOVDQU    AA0, (4*32)(oup); VMOVDQU BB0, (5*32)(oup); VMOVDQU CC0, (6*32)(oup); VMOVDQU DD0, (7*32)(oup)

	VPERM2I128 $0x02, AA2, BB2, AA0
	VPERM2I128 $0x02, CC2, DD2, BB0
	VPERM2I128 $0x13, AA2, BB2, CC0
	VPERM2I128 $0x13, CC2, DD2, DD0
	VPXOR      (8*32)(inp), AA0, AA0; VPXOR (9*32)(inp), BB0, BB0; VPXOR (10*32)(inp), CC0, CC0; VPXOR (11*32)(inp), DD0, DD0
	VMOVDQU    AA0, (8*32)(oup); VMOVDQU BB0, (9*32)(oup); VMOVDQU CC0, (10*32)(oup); VMOVDQU DD0, (11*32)(oup)

	MOVQ       $384, itr1
	LEAQ       384(inp), inp
	SUBQ       $384, inl
	VPERM2I128 $0x02, AA3, BB3, AA0
	VPERM2I128 $0x02, tmpStoreAVX2, DD3, BB0
	VPERM2I128 $0x13, AA3, BB3, CC0
	VPERM2I128 $0x13, tmpStoreAVX2, DD3, DD0

	JMP sealAVX2SealHash
