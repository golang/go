// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build gc && !purego

#include "textflag.h"

// func kimd(function code, chain *[200]byte, src []byte)
TEXT ·kimd(SB), NOFRAME|NOSPLIT, $0-40
	MOVD function+0(FP), R0
	MOVD chain+8(FP), R1
	LMG  src+16(FP), R2, R3 // R2=base, R3=len

continue:
	WORD $0xB93E0002 // KIMD --, R2
	BVS  continue    // continue if interrupted
	MOVD $0, R0      // reset R0 for pre-go1.8 compilers
	RET

// func klmd(function code, chain *[200]byte, dst, src []byte)
TEXT ·klmd(SB), NOFRAME|NOSPLIT, $0-64
	// TODO: SHAKE support
	MOVD function+0(FP), R0
	MOVD chain+8(FP), R1
	LMG  dst+16(FP), R2, R3 // R2=base, R3=len
	LMG  src+40(FP), R4, R5 // R4=base, R5=len

continue:
	WORD $0xB93F0024 // KLMD R2, R4
	BVS  continue    // continue if interrupted
	MOVD $0, R0      // reset R0 for pre-go1.8 compilers
	RET
