// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build amd64,!gccgo,!appengine,go1.7

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

DATA poly1305Mask<>+0x00(SB)/8, $0x0FFFFFFC0FFFFFFF
DATA poly1305Mask<>+0x08(SB)/8, $0x0FFFFFFC0FFFFFFC
GLOBL poly1305Mask<>(SB), RODATA, $16

// func poly1305(out *[16]byte, m *byte, mlen uint64, key *[32]key)
TEXT ·poly1305(SB), $0-32
	MOVQ out+0(FP), DI
	MOVQ m+8(FP), SI
	MOVQ mlen+16(FP), R15
	MOVQ key+24(FP), AX

	MOVQ 0(AX), R11
	MOVQ 8(AX), R12
	ANDQ poly1305Mask<>(SB), R11   // r0
	ANDQ poly1305Mask<>+8(SB), R12 // r1
	XORQ R8, R8                    // h0
	XORQ R9, R9                    // h1
	XORQ R10, R10                  // h2

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
	MOVQ    R8, AX
	MOVQ    R9, BX
	SUBQ    $0xFFFFFFFFFFFFFFFB, AX
	SBBQ    $0xFFFFFFFFFFFFFFFF, BX
	SBBQ    $3, R10
	CMOVQCS R8, AX
	CMOVQCS R9, BX
	MOVQ    key+24(FP), R8
	ADDQ    16(R8), AX
	ADCQ    24(R8), BX

	MOVQ AX, 0(DI)
	MOVQ BX, 8(DI)
	RET
