// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !nacl

#include "textflag.h"

// Registers
#define Rdst	R0
#define Rsrc	R1
#define Rn	R2
#define Rstate	R3
#define Rpi	R4
#define Rpj	R5
#define Ri	R6
#define Rj	R7
#define Rk	R8
#define Rt	R11
#define Rt2	R12

// func xorKeyStream(dst, src *byte, n int, state *[256]byte, i, j *uint8)
TEXT ·xorKeyStream(SB),NOSPLIT,$0
	MOVW dst+0(FP), Rdst
	MOVW src+4(FP), Rsrc
	MOVW n+8(FP), Rn
	MOVW state+12(FP), Rstate
	MOVW pi+16(FP), Rpi
	MOVW pj+20(FP), Rpj
	MOVBU (Rpi), Ri
	MOVBU (Rpj), Rj
	MOVW $0, Rk

loop:
	// i += 1; j += state[i]
	ADD $1, Ri
	AND $0xff, Ri
	MOVBU Ri<<2(Rstate), Rt
	ADD Rt, Rj
	AND $0xff, Rj

	// swap state[i] <-> state[j]
	MOVBU Rj<<2(Rstate), Rt2
	MOVB Rt2, Ri<<2(Rstate)
	MOVB Rt, Rj<<2(Rstate)

	// dst[k] = src[k] ^ state[state[i] + state[j]]
	ADD Rt2, Rt
	AND $0xff, Rt
	MOVBU Rt<<2(Rstate), Rt
	MOVBU Rk<<0(Rsrc), Rt2
	EOR Rt, Rt2
	MOVB Rt2, Rk<<0(Rdst)

	ADD $1, Rk
	CMP Rk, Rn
	BNE loop

done:
	MOVB Ri, (Rpi)
	MOVB Rj, (Rpj)
	RET
