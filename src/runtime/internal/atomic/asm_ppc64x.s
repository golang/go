// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ppc64 ppc64le

#include "textflag.h"

// bool cas(uint32 *ptr, uint32 old, uint32 new)
// Atomically:
//	if(*val == old){
//		*val = new;
//		return 1;
//	} else
//		return 0;
TEXT runtime∕internal∕atomic·Cas(SB), NOSPLIT, $0-17
	MOVD	ptr+0(FP), R3
	MOVWZ	old+8(FP), R4
	MOVWZ	new+12(FP), R5
	SYNC
cas_again:
	LWAR	(R3), R6
	CMPW	R6, R4
	BNE	cas_fail
	STWCCC	R5, (R3)
	BNE	cas_again
	MOVD	$1, R3
	ISYNC
	MOVB	R3, ret+16(FP)
	RET
cas_fail:
	MOVB	R0, ret+16(FP)
	RET

// bool	runtime∕internal∕atomic·Cas64(uint64 *ptr, uint64 old, uint64 new)
// Atomically:
//	if(*val == *old){
//		*val = new;
//		return 1;
//	} else {
//		return 0;
//	}
TEXT runtime∕internal∕atomic·Cas64(SB), NOSPLIT, $0-25
	MOVD	ptr+0(FP), R3
	MOVD	old+8(FP), R4
	MOVD	new+16(FP), R5
	SYNC
cas64_again:
	LDAR	(R3), R6
	CMP	R6, R4
	BNE	cas64_fail
	STDCCC	R5, (R3)
	BNE	cas64_again
	MOVD	$1, R3
	ISYNC
	MOVB	R3, ret+24(FP)
	RET
cas64_fail:
	MOVB	R0, ret+24(FP)
	RET

TEXT runtime∕internal∕atomic·Casuintptr(SB), NOSPLIT, $0-25
	BR	runtime∕internal∕atomic·Cas64(SB)

TEXT runtime∕internal∕atomic·Loaduintptr(SB),  NOSPLIT|NOFRAME, $0-16
	BR	runtime∕internal∕atomic·Load64(SB)

TEXT runtime∕internal∕atomic·Loaduint(SB), NOSPLIT|NOFRAME, $0-16
	BR	runtime∕internal∕atomic·Load64(SB)

TEXT runtime∕internal∕atomic·Storeuintptr(SB), NOSPLIT, $0-16
	BR	runtime∕internal∕atomic·Store64(SB)

TEXT runtime∕internal∕atomic·Xadduintptr(SB), NOSPLIT, $0-24
	BR	runtime∕internal∕atomic·Xadd64(SB)

TEXT runtime∕internal∕atomic·Loadint64(SB), NOSPLIT, $0-16
	BR	runtime∕internal∕atomic·Load64(SB)

TEXT runtime∕internal∕atomic·Xaddint64(SB), NOSPLIT, $0-24
	BR	runtime∕internal∕atomic·Xadd64(SB)

// bool casp(void **val, void *old, void *new)
// Atomically:
//	if(*val == old){
//		*val = new;
//		return 1;
//	} else
//		return 0;
TEXT runtime∕internal∕atomic·Casp1(SB), NOSPLIT, $0-25
	BR runtime∕internal∕atomic·Cas64(SB)

// uint32 xadd(uint32 volatile *ptr, int32 delta)
// Atomically:
//	*val += delta;
//	return *val;
TEXT runtime∕internal∕atomic·Xadd(SB), NOSPLIT, $0-20
	MOVD	ptr+0(FP), R4
	MOVW	delta+8(FP), R5
	SYNC
	LWAR	(R4), R3
	ADD	R5, R3
	STWCCC	R3, (R4)
	BNE	-3(PC)
	ISYNC
	MOVW	R3, ret+16(FP)
	RET

TEXT runtime∕internal∕atomic·Xadd64(SB), NOSPLIT, $0-24
	MOVD	ptr+0(FP), R4
	MOVD	delta+8(FP), R5
	SYNC
	LDAR	(R4), R3
	ADD	R5, R3
	STDCCC	R3, (R4)
	BNE	-3(PC)
	ISYNC
	MOVD	R3, ret+16(FP)
	RET

TEXT runtime∕internal∕atomic·Xchg(SB), NOSPLIT, $0-20
	MOVD	ptr+0(FP), R4
	MOVW	new+8(FP), R5
	SYNC
	LWAR	(R4), R3
	STWCCC	R5, (R4)
	BNE	-2(PC)
	ISYNC
	MOVW	R3, ret+16(FP)
	RET

TEXT runtime∕internal∕atomic·Xchg64(SB), NOSPLIT, $0-24
	MOVD	ptr+0(FP), R4
	MOVD	new+8(FP), R5
	SYNC
	LDAR	(R4), R3
	STDCCC	R5, (R4)
	BNE	-2(PC)
	ISYNC
	MOVD	R3, ret+16(FP)
	RET

TEXT runtime∕internal∕atomic·Xchguintptr(SB), NOSPLIT, $0-24
	BR	runtime∕internal∕atomic·Xchg64(SB)


TEXT runtime∕internal∕atomic·StorepNoWB(SB), NOSPLIT, $0-16
	BR	runtime∕internal∕atomic·Store64(SB)

TEXT runtime∕internal∕atomic·Store(SB), NOSPLIT, $0-12
	MOVD	ptr+0(FP), R3
	MOVW	val+8(FP), R4
	SYNC
	MOVW	R4, 0(R3)
	RET

TEXT runtime∕internal∕atomic·Store64(SB), NOSPLIT, $0-16
	MOVD	ptr+0(FP), R3
	MOVD	val+8(FP), R4
	SYNC
	MOVD	R4, 0(R3)
	RET

// void runtime∕internal∕atomic·Or8(byte volatile*, byte);
TEXT runtime∕internal∕atomic·Or8(SB), NOSPLIT, $0-9
	MOVD	ptr+0(FP), R3
	MOVBZ	val+8(FP), R4
	SYNC
again:
	LBAR	(R3), R6
	OR	R4, R6
	STBCCC	R6, (R3)
	BNE	again
	ISYNC
	RET

// void runtime∕internal∕atomic·And8(byte volatile*, byte);
TEXT runtime∕internal∕atomic·And8(SB), NOSPLIT, $0-9
	MOVD	ptr+0(FP), R3
	MOVBZ	val+8(FP), R4
	SYNC
again:
	LBAR	(R3),R6
	AND	R4,R6
	STBCCC	R6,(R3)
	BNE	again
	ISYNC
	RET
