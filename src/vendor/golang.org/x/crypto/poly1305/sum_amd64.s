// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build amd64,!gccgo,!appengine

#include "textflag.h"

#define POLY1305_ADD(msg, h0, h1, h2) \
	ADDQ 0(msg), h0;  \
	ADCQ 8(msg), h1;  \
	ADCQ $1, h2;      \
	LEAQ 16(msg), msg

#define POLY1305_MUL(h0, h1, h2, r0, r1, t0, t1, t2, t3) \
	MOVQ  r0, AX;                  \
	MULQ  h0;                      \
	MOVQ  AX, t0;                  \
	MOVQ  DX, t1;                  \
	MOVQ  r0, AX;                  \
	MULQ  h1;                      \
	ADDQ  AX, t1;                  \
	ADCQ  $0, DX;                  \
	MOVQ  r0, t2;                  \
	IMULQ h2, t2;                  \
	ADDQ  DX, t2;                  \
	                               \
	MOVQ  r1, AX;                  \
	MULQ  h0;                      \
	ADDQ  AX, t1;                  \
	ADCQ  $0, DX;                  \
	MOVQ  DX, h0;                  \
	MOVQ  r1, t3;                  \
	IMULQ h2, t3;                  \
	MOVQ  r1, AX;                  \
	MULQ  h1;                      \
	ADDQ  AX, t2;                  \
	ADCQ  DX, t3;                  \
	ADDQ  h0, t2;                  \
	ADCQ  $0, t3;                  \
	                               \
	MOVQ  t0, h0;                  \
	MOVQ  t1, h1;                  \
	MOVQ  t2, h2;                  \
	ANDQ  $3, h2;                  \
	MOVQ  t2, t0;                  \
	ANDQ  $0xFFFFFFFFFFFFFFFC, t0; \
	ADDQ  t0, h0;                  \
	ADCQ  t3, h1;                  \
	ADCQ  $0, h2;                  \
	SHRQ  $2, t3, t2;              \
	SHRQ  $2, t3;                  \
	ADDQ  t2, h0;                  \
	ADCQ  t3, h1;                  \
	ADCQ  $0, h2

DATA ·poly1305Mask<>+0x00(SB)/8, $0x0FFFFFFC0FFFFFFF
DATA ·poly1305Mask<>+0x08(SB)/8, $0x0FFFFFFC0FFFFFFC
GLOBL ·poly1305Mask<>(SB), RODATA, $16

// func update(state *[7]uint64, msg []byte)
TEXT ·update(SB), $0-32
	MOVQ state+0(FP), DI
	MOVQ msg_base+8(FP), SI
	MOVQ msg_len+16(FP), R15

	MOVQ 0(DI), R8   // h0
	MOVQ 8(DI), R9   // h1
	MOVQ 16(DI), R10 // h2
	MOVQ 24(DI), R11 // r0
	MOVQ 32(DI), R12 // r1

	CMPQ R15, $16
	JB   bytes_between_0_and_15

loop:
	POLY1305_ADD(SI, R8, R9, R10)

multiply:
	POLY1305_MUL(R8, R9, R10, R11, R12, BX, CX, R13, R14)
	SUBQ $16, R15
	CMPQ R15, $16
	JAE  loop

bytes_between_0_and_15:
	TESTQ R15, R15
	JZ    done
	MOVQ  $1, BX
	XORQ  CX, CX
	XORQ  R13, R13
	ADDQ  R15, SI

flush_buffer:
	SHLQ $8, BX, CX
	SHLQ $8, BX
	MOVB -1(SI), R13
	XORQ R13, BX
	DECQ SI
	DECQ R15
	JNZ  flush_buffer

	ADDQ BX, R8
	ADCQ CX, R9
	ADCQ $0, R10
	MOVQ $16, R15
	JMP  multiply

done:
	MOVQ R8, 0(DI)
	MOVQ R9, 8(DI)
	MOVQ R10, 16(DI)
	RET

// func initialize(state *[7]uint64, key *[32]byte)
TEXT ·initialize(SB), $0-16
	MOVQ state+0(FP), DI
	MOVQ key+8(FP), SI

	// state[0...7] is initialized with zero
	MOVOU 0(SI), X0
	MOVOU 16(SI), X1
	MOVOU ·poly1305Mask<>(SB), X2
	PAND  X2, X0
	MOVOU X0, 24(DI)
	MOVOU X1, 40(DI)
	RET

// func finalize(tag *[TagSize]byte, state *[7]uint64)
TEXT ·finalize(SB), $0-16
	MOVQ tag+0(FP), DI
	MOVQ state+8(FP), SI

	MOVQ    0(SI), AX
	MOVQ    8(SI), BX
	MOVQ    16(SI), CX
	MOVQ    AX, R8
	MOVQ    BX, R9
	SUBQ    $0xFFFFFFFFFFFFFFFB, AX
	SBBQ    $0xFFFFFFFFFFFFFFFF, BX
	SBBQ    $3, CX
	CMOVQCS R8, AX
	CMOVQCS R9, BX
	ADDQ    40(SI), AX
	ADCQ    48(SI), BX

	MOVQ AX, 0(DI)
	MOVQ BX, 8(DI)
	RET
