// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT ·asmAddFlags(SB),NOSPLIT,$0-24
	MOVD	x+0(FP), R0
	MOVD	y+8(FP), R1
	CMN	R0, R1
	WORD	$0xd53b4200 //	MOVD	NZCV, R0
	MOVD	R0, ret+16(FP)
	RET

TEXT ·asmSubFlags(SB),NOSPLIT,$0-24
	MOVD	x+0(FP), R0
	MOVD	y+8(FP), R1
	CMP	R1, R0
	WORD	$0xd53b4200 //	MOVD	NZCV, R0
	MOVD	R0, ret+16(FP)
	RET

TEXT ·asmAndFlags(SB),NOSPLIT,$0-24
	MOVD	x+0(FP), R0
	MOVD	y+8(FP), R1
	TST	R1, R0
	WORD	$0xd53b4200 //	MOVD	NZCV, R0
	BIC	$0x30000000, R0 // clear C, V bits, as TST does not change those flags
	MOVD	R0, ret+16(FP)
	RET
