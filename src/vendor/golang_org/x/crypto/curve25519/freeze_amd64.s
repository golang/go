// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This code was translated into a form compatible with 6a from the public
// domain sources in SUPERCOP: http://bench.cr.yp.to/supercop.html

// +build amd64,!gccgo,!appengine

#include "const_amd64.h"

// func freeze(inout *[5]uint64)
TEXT ·freeze(SB),7,$0-8
	MOVQ inout+0(FP), DI

	MOVQ 0(DI),SI
	MOVQ 8(DI),DX
	MOVQ 16(DI),CX
	MOVQ 24(DI),R8
	MOVQ 32(DI),R9
	MOVQ $REDMASK51,AX
	MOVQ AX,R10
	SUBQ $18,R10
	MOVQ $3,R11
REDUCELOOP:
	MOVQ SI,R12
	SHRQ $51,R12
	ANDQ AX,SI
	ADDQ R12,DX
	MOVQ DX,R12
	SHRQ $51,R12
	ANDQ AX,DX
	ADDQ R12,CX
	MOVQ CX,R12
	SHRQ $51,R12
	ANDQ AX,CX
	ADDQ R12,R8
	MOVQ R8,R12
	SHRQ $51,R12
	ANDQ AX,R8
	ADDQ R12,R9
	MOVQ R9,R12
	SHRQ $51,R12
	ANDQ AX,R9
	IMUL3Q $19,R12,R12
	ADDQ R12,SI
	SUBQ $1,R11
	JA REDUCELOOP
	MOVQ $1,R12
	CMPQ R10,SI
	CMOVQLT R11,R12
	CMPQ AX,DX
	CMOVQNE R11,R12
	CMPQ AX,CX
	CMOVQNE R11,R12
	CMPQ AX,R8
	CMOVQNE R11,R12
	CMPQ AX,R9
	CMOVQNE R11,R12
	NEGQ R12
	ANDQ R12,AX
	ANDQ R12,R10
	SUBQ R10,SI
	SUBQ AX,DX
	SUBQ AX,CX
	SUBQ AX,R8
	SUBQ AX,R9
	MOVQ SI,0(DI)
	MOVQ DX,8(DI)
	MOVQ CX,16(DI)
	MOVQ R8,24(DI)
	MOVQ R9,32(DI)
	RET
