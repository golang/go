// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !linux,arm

#include "textflag.h"

// TODO(minux): this is only valid for ARMv6+
// bool armcas(int32 *val, int32 old, int32 new)
// Atomically:
//	if(*val == old){
//		*val = new;
//		return 1;
//	}else
//		return 0;
TEXT	·Cas(SB),NOSPLIT,$0
	JMP	·armcas(SB)

TEXT	·Casp1(SB),NOSPLIT,$0
	JMP	·Cas(SB)

TEXT runtime∕internal∕atomic·Load(SB),NOSPLIT|NOFRAME,$0-8
	MOVW	addr+0(FP), R0
	MOVW	(R0), R1

	MOVB	runtime·goarm(SB), R11
	CMP	$7, R11
	BLT	2(PC)
	DMB	MB_ISH

	MOVW	R1, ret+4(FP)
	RET
