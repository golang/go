// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build mips64 mips64le

#include "textflag.h"

#define SYNC	WORD $0xf

// uint32 runtime∕internal∕atomic·Load(uint32 volatile* ptr)
TEXT ·Load(SB),NOSPLIT|NOFRAME,$0-12
	MOVV	ptr+0(FP), R1
	SYNC
	MOVWU	0(R1), R1
	SYNC
	MOVW	R1, ret+8(FP)
	RET

// uint8 runtime∕internal∕atomic·Load8(uint8 volatile* ptr)
TEXT ·Load8(SB),NOSPLIT|NOFRAME,$0-9
	MOVV	ptr+0(FP), R1
	SYNC
	MOVBU	0(R1), R1
	SYNC
	MOVB	R1, ret+8(FP)
	RET

// uint64 runtime∕internal∕atomic·Load64(uint64 volatile* ptr)
TEXT ·Load64(SB),NOSPLIT|NOFRAME,$0-16
	MOVV	ptr+0(FP), R1
	SYNC
	MOVV	0(R1), R1
	SYNC
	MOVV	R1, ret+8(FP)
	RET

// void *runtime∕internal∕atomic·Loadp(void *volatile *ptr)
TEXT ·Loadp(SB),NOSPLIT|NOFRAME,$0-16
	MOVV	ptr+0(FP), R1
	SYNC
	MOVV	0(R1), R1
	SYNC
	MOVV	R1, ret+8(FP)
	RET

// uint32 runtime∕internal∕atomic·LoadAcq(uint32 volatile* ptr)
TEXT ·LoadAcq(SB),NOSPLIT|NOFRAME,$0-12
	JMP	atomic·Load(SB)
