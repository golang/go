// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build amd64,!appengine,!gccgo

#include "textflag.h"

// func havePOPCNT() bool
TEXT ·havePOPCNT(SB),4,$0
	MOVQ	$1, AX
	CPUID
	SHRQ	$23, CX
	ANDQ	$1, CX
	MOVB	CX, ret+0(FP)
	RET

// func popcnt(word) int
TEXT ·popcnt(SB),NOSPLIT,$0-8
	XORQ	AX, AX
	MOVQ	x+0(FP), SI
	// POPCNT (SI), AX is not recognized by Go assembler,
	// so we assemble it ourselves.
	BYTE	$0xf3
	BYTE	$0x48
	BYTE	$0x0f
	BYTE	$0xb8
	BYTE	$0xc6
	MOVQ	AX, ret+8(FP)
	RET
