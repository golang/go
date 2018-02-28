// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// bool armcas(int32 *val, int32 old, int32 new)
// Atomically:
//	if(*val == old){
//		*val = new;
//		return 1;
//	}else
//		return 0;
//
// To implement runtime∕internal∕atomic·cas in sys_$GOOS_arm.s
// using the native instructions, use:
//
//	TEXT runtime∕internal∕atomic·cas(SB),NOSPLIT,$0
//		B	runtime∕internal∕atomic·armcas(SB)
//
TEXT runtime∕internal∕atomic·armcas(SB),NOSPLIT,$0-13
	MOVW	ptr+0(FP), R1
	MOVW	old+4(FP), R2
	MOVW	new+8(FP), R3
casl:
	LDREX	(R1), R0
	CMP	R0, R2
	BNE	casfail

	MOVB	runtime·goarm(SB), R11
	CMP	$7, R11
	BLT	2(PC)
	WORD	$0xf57ff05a	// dmb ishst

	STREX	R3, (R1), R0
	CMP	$0, R0
	BNE	casl
	MOVW	$1, R0

	MOVB	runtime·goarm(SB), R11
	CMP	$7, R11
	BLT	2(PC)
	WORD	$0xf57ff05b	// dmb ish

	MOVB	R0, ret+12(FP)
	RET
casfail:
	MOVW	$0, R0
	MOVB	R0, ret+12(FP)
	RET

TEXT runtime∕internal∕atomic·Casuintptr(SB),NOSPLIT,$0-13
	B	runtime∕internal∕atomic·Cas(SB)

TEXT runtime∕internal∕atomic·Loaduintptr(SB),NOSPLIT,$0-8
	B	runtime∕internal∕atomic·Load(SB)

TEXT runtime∕internal∕atomic·Loaduint(SB),NOSPLIT,$0-8
	B	runtime∕internal∕atomic·Load(SB)

TEXT runtime∕internal∕atomic·Storeuintptr(SB),NOSPLIT,$0-8
	B	runtime∕internal∕atomic·Store(SB)

TEXT runtime∕internal∕atomic·Xadduintptr(SB),NOSPLIT,$0-12
	B	runtime∕internal∕atomic·Xadd(SB)

TEXT runtime∕internal∕atomic·Loadint64(SB),NOSPLIT,$0-12
	B	runtime∕internal∕atomic·Load64(SB)

TEXT runtime∕internal∕atomic·Xaddint64(SB),NOSPLIT,$0-20
	B	runtime∕internal∕atomic·Xadd64(SB)
