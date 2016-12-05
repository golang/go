// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// bool Cas(int32 *val, int32 old, int32 new)
// Atomically:
//	if(*val == old){
//		*val = new;
//		return 1;
//	}else
//		return 0;
TEXT runtime∕internal∕atomic·Cas(SB), NOSPLIT, $0-13
	MOVL	ptr+0(FP), BX
	MOVL	old+4(FP), AX
	MOVL	new+8(FP), CX
	LOCK
	CMPXCHGL	CX, 0(BX)
	SETEQ	ret+12(FP)
	RET

TEXT runtime∕internal∕atomic·Casuintptr(SB), NOSPLIT, $0-13
	JMP	runtime∕internal∕atomic·Cas(SB)

TEXT runtime∕internal∕atomic·Loaduintptr(SB), NOSPLIT, $0-8
	JMP	runtime∕internal∕atomic·Load(SB)

TEXT runtime∕internal∕atomic·Loaduint(SB), NOSPLIT, $0-8
	JMP	runtime∕internal∕atomic·Load(SB)

TEXT runtime∕internal∕atomic·Storeuintptr(SB), NOSPLIT, $0-8
	JMP	runtime∕internal∕atomic·Store(SB)

TEXT runtime∕internal∕atomic·Xadduintptr(SB), NOSPLIT, $0-12
	JMP runtime∕internal∕atomic·Xadd(SB)

TEXT runtime∕internal∕atomic·Loadint64(SB), NOSPLIT, $0-12
	JMP runtime∕internal∕atomic·Load64(SB)

TEXT runtime∕internal∕atomic·Xaddint64(SB), NOSPLIT, $0-20
	JMP runtime∕internal∕atomic·Xadd64(SB)


// bool runtime∕internal∕atomic·Cas64(uint64 *val, uint64 old, uint64 new)
// Atomically:
//	if(*val == *old){
//		*val = new;
//		return 1;
//	} else {
//		return 0;
//	}
TEXT runtime∕internal∕atomic·Cas64(SB), NOSPLIT, $0-21
	MOVL	ptr+0(FP), BP
	TESTL	$7, BP
	JZ	2(PC)
	MOVL	0, BP // crash with nil ptr deref
	MOVL	old_lo+4(FP), AX
	MOVL	old_hi+8(FP), DX
	MOVL	new_lo+12(FP), BX
	MOVL	new_hi+16(FP), CX
	LOCK
	CMPXCHG8B	0(BP)
	SETEQ	ret+20(FP)
	RET

// bool Casp1(void **p, void *old, void *new)
// Atomically:
//	if(*p == old){
//		*p = new;
//		return 1;
//	}else
//		return 0;
TEXT runtime∕internal∕atomic·Casp1(SB), NOSPLIT, $0-13
	MOVL	ptr+0(FP), BX
	MOVL	old+4(FP), AX
	MOVL	new+8(FP), CX
	LOCK
	CMPXCHGL	CX, 0(BX)
	SETEQ	ret+12(FP)
	RET

// uint32 Xadd(uint32 volatile *val, int32 delta)
// Atomically:
//	*val += delta;
//	return *val;
TEXT runtime∕internal∕atomic·Xadd(SB), NOSPLIT, $0-12
	MOVL	ptr+0(FP), BX
	MOVL	delta+4(FP), AX
	MOVL	AX, CX
	LOCK
	XADDL	AX, 0(BX)
	ADDL	CX, AX
	MOVL	AX, ret+8(FP)
	RET

TEXT runtime∕internal∕atomic·Xchg(SB), NOSPLIT, $0-12
	MOVL	ptr+0(FP), BX
	MOVL	new+4(FP), AX
	XCHGL	AX, 0(BX)
	MOVL	AX, ret+8(FP)
	RET

TEXT runtime∕internal∕atomic·Xchguintptr(SB), NOSPLIT, $0-12
	JMP	runtime∕internal∕atomic·Xchg(SB)


TEXT runtime∕internal∕atomic·StorepNoWB(SB), NOSPLIT, $0-8
	MOVL	ptr+0(FP), BX
	MOVL	val+4(FP), AX
	XCHGL	AX, 0(BX)
	RET

TEXT runtime∕internal∕atomic·Store(SB), NOSPLIT, $0-8
	MOVL	ptr+0(FP), BX
	MOVL	val+4(FP), AX
	XCHGL	AX, 0(BX)
	RET

// uint64 atomicload64(uint64 volatile* addr);
TEXT runtime∕internal∕atomic·Load64(SB), NOSPLIT, $0-12
	MOVL	ptr+0(FP), AX
	TESTL	$7, AX
	JZ	2(PC)
	MOVL	0, AX // crash with nil ptr deref
	LEAL	ret_lo+4(FP), BX
	// MOVQ (%EAX), %MM0
	BYTE $0x0f; BYTE $0x6f; BYTE $0x00
	// MOVQ %MM0, 0(%EBX)
	BYTE $0x0f; BYTE $0x7f; BYTE $0x03
	// EMMS
	BYTE $0x0F; BYTE $0x77
	RET

// void runtime∕internal∕atomic·Store64(uint64 volatile* addr, uint64 v);
TEXT runtime∕internal∕atomic·Store64(SB), NOSPLIT, $0-12
	MOVL	ptr+0(FP), AX
	TESTL	$7, AX
	JZ	2(PC)
	MOVL	0, AX // crash with nil ptr deref
	// MOVQ and EMMS were introduced on the Pentium MMX.
	// MOVQ 0x8(%ESP), %MM0
	BYTE $0x0f; BYTE $0x6f; BYTE $0x44; BYTE $0x24; BYTE $0x08
	// MOVQ %MM0, (%EAX)
	BYTE $0x0f; BYTE $0x7f; BYTE $0x00 
	// EMMS
	BYTE $0x0F; BYTE $0x77
	// This is essentially a no-op, but it provides required memory fencing.
	// It can be replaced with MFENCE, but MFENCE was introduced only on the Pentium4 (SSE2).
	MOVL	$0, AX
	LOCK
	XADDL	AX, (SP)
	RET

// void	runtime∕internal∕atomic·Or8(byte volatile*, byte);
TEXT runtime∕internal∕atomic·Or8(SB), NOSPLIT, $0-5
	MOVL	ptr+0(FP), AX
	MOVB	val+4(FP), BX
	LOCK
	ORB	BX, (AX)
	RET

// void	runtime∕internal∕atomic·And8(byte volatile*, byte);
TEXT runtime∕internal∕atomic·And8(SB), NOSPLIT, $0-5
	MOVL	ptr+0(FP), AX
	MOVB	val+4(FP), BX
	LOCK
	ANDB	BX, (AX)
	RET
