// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ppc64 ppc64le

#include "textflag.h"

// uint32 runtime·atomicload(uint32 volatile* addr)
TEXT ·atomicload(SB),NOSPLIT,$-8-12
	MOVD	0(FP), R3
	SYNC
	MOVWZ	0(R3), R3
	CMPW	R3, R3, CR7
	BC	4, 30, 1(PC) // bne- cr7,0x4
	ISYNC
	MOVW	R3, ret+8(FP)
	RETURN

// uint64 runtime·atomicload64(uint64 volatile* addr)
TEXT ·atomicload64(SB),NOSPLIT,$-8-16
	MOVD	0(FP), R3
	SYNC
	MOVD	0(R3), R3
	CMP	R3, R3, CR7
	BC	4, 30, 1(PC) // bne- cr7,0x4
	ISYNC
	MOVD	R3, ret+8(FP)
	RETURN

// void *runtime·atomicloadp(void *volatile *addr)
TEXT ·atomicloadp(SB),NOSPLIT,$-8-16
	MOVD	0(FP), R3
	SYNC
	MOVD	0(R3), R3
	CMP	R3, R3, CR7
	BC	4, 30, 1(PC) // bne- cr7,0x4
	ISYNC
	MOVD	R3, ret+8(FP)
	RETURN
