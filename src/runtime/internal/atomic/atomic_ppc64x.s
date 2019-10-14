// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ppc64 ppc64le

#include "textflag.h"

// uint32 runtime∕internal∕atomic·Load(uint32 volatile* ptr)
TEXT ·Load(SB),NOSPLIT|NOFRAME,$-8-12
	MOVD	ptr+0(FP), R3
	SYNC
	MOVWZ	0(R3), R3
	CMPW	R3, R3, CR7
	BC	4, 30, 1(PC) // bne- cr7,0x4
	ISYNC
	MOVW	R3, ret+8(FP)
	RET

// uint8 runtime∕internal∕atomic·Load8(uint8 volatile* ptr)
TEXT ·Load8(SB),NOSPLIT|NOFRAME,$-8-9
	MOVD	ptr+0(FP), R3
	SYNC
	MOVBZ	0(R3), R3
	CMP	R3, R3, CR7
	BC	4, 30, 1(PC) // bne- cr7,0x4
	ISYNC
	MOVB	R3, ret+8(FP)
	RET

// uint64 runtime∕internal∕atomic·Load64(uint64 volatile* ptr)
TEXT ·Load64(SB),NOSPLIT|NOFRAME,$-8-16
	MOVD	ptr+0(FP), R3
	SYNC
	MOVD	0(R3), R3
	CMP	R3, R3, CR7
	BC	4, 30, 1(PC) // bne- cr7,0x4
	ISYNC
	MOVD	R3, ret+8(FP)
	RET

// void *runtime∕internal∕atomic·Loadp(void *volatile *ptr)
TEXT ·Loadp(SB),NOSPLIT|NOFRAME,$-8-16
	MOVD	ptr+0(FP), R3
	SYNC
	MOVD	0(R3), R3
	CMP	R3, R3, CR7
	BC	4, 30, 1(PC) // bne- cr7,0x4
	ISYNC
	MOVD	R3, ret+8(FP)
	RET

// uint32 runtime∕internal∕atomic·LoadAcq(uint32 volatile* ptr)
TEXT ·LoadAcq(SB),NOSPLIT|NOFRAME,$-8-12
	MOVD   ptr+0(FP), R3
	MOVWZ  0(R3), R3
	CMPW   R3, R3, CR7
	BC     4, 30, 1(PC) // bne- cr7, 0x4
	MOVW   R3, ret+8(FP)
	RET
