// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"
#include "funcdata.h"

// bool Cas(int32 *val, int32 old, int32 new)
// Atomically:
//	if(*val == old){
//		*val = new;
//		return 1;
//	}else
//		return 0;
TEXT ·Cas(SB), NOSPLIT, $0-13
	MOVL	ptr+0(FP), BX
	MOVL	old+4(FP), AX
	MOVL	new+8(FP), CX
	LOCK
	CMPXCHGL	CX, 0(BX)
	SETEQ	ret+12(FP)
	RET

TEXT ·Casint32(SB), NOSPLIT, $0-13
	JMP	·Cas(SB)

TEXT ·Casint64(SB), NOSPLIT, $0-21
	JMP	·Cas64(SB)

TEXT ·Casuintptr(SB), NOSPLIT, $0-13
	JMP	·Cas(SB)

TEXT ·CasRel(SB), NOSPLIT, $0-13
	JMP	·Cas(SB)

TEXT ·Loaduintptr(SB), NOSPLIT, $0-8
	JMP	·Load(SB)

TEXT ·Loaduint(SB), NOSPLIT, $0-8
	JMP	·Load(SB)

TEXT ·Storeint32(SB), NOSPLIT, $0-8
	JMP	·Store(SB)

TEXT ·Storeint64(SB), NOSPLIT, $0-12
	JMP	·Store64(SB)

TEXT ·Storeuintptr(SB), NOSPLIT, $0-8
	JMP	·Store(SB)

TEXT ·Xadduintptr(SB), NOSPLIT, $0-12
	JMP	·Xadd(SB)

TEXT ·Loadint32(SB), NOSPLIT, $0-8
	JMP	·Load(SB)

TEXT ·Loadint64(SB), NOSPLIT, $0-12
	JMP	·Load64(SB)

TEXT ·Xaddint32(SB), NOSPLIT, $0-12
	JMP	·Xadd(SB)

TEXT ·Xaddint64(SB), NOSPLIT, $0-20
	JMP	·Xadd64(SB)

// bool ·Cas64(uint64 *val, uint64 old, uint64 new)
// Atomically:
//	if(*val == old){
//		*val = new;
//		return 1;
//	} else {
//		return 0;
//	}
TEXT ·Cas64(SB), NOSPLIT, $0-21
	NO_LOCAL_POINTERS
	MOVL	ptr+0(FP), BP
	TESTL	$7, BP
	JZ	2(PC)
	CALL	·panicUnaligned(SB)
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
TEXT ·Casp1(SB), NOSPLIT, $0-13
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
TEXT ·Xadd(SB), NOSPLIT, $0-12
	MOVL	ptr+0(FP), BX
	MOVL	delta+4(FP), AX
	MOVL	AX, CX
	LOCK
	XADDL	AX, 0(BX)
	ADDL	CX, AX
	MOVL	AX, ret+8(FP)
	RET

TEXT ·Xadd64(SB), NOSPLIT, $0-20
	NO_LOCAL_POINTERS
	// no XADDQ so use CMPXCHG8B loop
	MOVL	ptr+0(FP), BP
	TESTL	$7, BP
	JZ	2(PC)
	CALL	·panicUnaligned(SB)
	// DI:SI = delta
	MOVL	delta_lo+4(FP), SI
	MOVL	delta_hi+8(FP), DI
	// DX:AX = *addr
	MOVL	0(BP), AX
	MOVL	4(BP), DX
addloop:
	// CX:BX = DX:AX (*addr) + DI:SI (delta)
	MOVL	AX, BX
	MOVL	DX, CX
	ADDL	SI, BX
	ADCL	DI, CX

	// if *addr == DX:AX {
	//	*addr = CX:BX
	// } else {
	//	DX:AX = *addr
	// }
	// all in one instruction
	LOCK
	CMPXCHG8B	0(BP)

	JNZ	addloop

	// success
	// return CX:BX
	MOVL	BX, ret_lo+12(FP)
	MOVL	CX, ret_hi+16(FP)
	RET

TEXT ·Xchg(SB), NOSPLIT, $0-12
	MOVL	ptr+0(FP), BX
	MOVL	new+4(FP), AX
	XCHGL	AX, 0(BX)
	MOVL	AX, ret+8(FP)
	RET

TEXT ·Xchgint32(SB), NOSPLIT, $0-12
	JMP	·Xchg(SB)

TEXT ·Xchgint64(SB), NOSPLIT, $0-20
	JMP	·Xchg64(SB)

TEXT ·Xchguintptr(SB), NOSPLIT, $0-12
	JMP	·Xchg(SB)

TEXT ·Xchg64(SB),NOSPLIT,$0-20
	NO_LOCAL_POINTERS
	// no XCHGQ so use CMPXCHG8B loop
	MOVL	ptr+0(FP), BP
	TESTL	$7, BP
	JZ	2(PC)
	CALL	·panicUnaligned(SB)
	// CX:BX = new
	MOVL	new_lo+4(FP), BX
	MOVL	new_hi+8(FP), CX
	// DX:AX = *addr
	MOVL	0(BP), AX
	MOVL	4(BP), DX
swaploop:
	// if *addr == DX:AX
	//	*addr = CX:BX
	// else
	//	DX:AX = *addr
	// all in one instruction
	LOCK
	CMPXCHG8B	0(BP)
	JNZ	swaploop

	// success
	// return DX:AX
	MOVL	AX, ret_lo+12(FP)
	MOVL	DX, ret_hi+16(FP)
	RET

TEXT ·StorepNoWB(SB), NOSPLIT, $0-8
	MOVL	ptr+0(FP), BX
	MOVL	val+4(FP), AX
	XCHGL	AX, 0(BX)
	RET

TEXT ·Store(SB), NOSPLIT, $0-8
	MOVL	ptr+0(FP), BX
	MOVL	val+4(FP), AX
	XCHGL	AX, 0(BX)
	RET

TEXT ·StoreRel(SB), NOSPLIT, $0-8
	JMP	·Store(SB)

TEXT ·StoreReluintptr(SB), NOSPLIT, $0-8
	JMP	·Store(SB)

// uint64 atomicload64(uint64 volatile* addr);
TEXT ·Load64(SB), NOSPLIT, $0-12
	NO_LOCAL_POINTERS
	MOVL	ptr+0(FP), AX
	TESTL	$7, AX
	JZ	2(PC)
	CALL	·panicUnaligned(SB)
	MOVQ	(AX), M0
	MOVQ	M0, ret+4(FP)
	EMMS
	RET

// void ·Store64(uint64 volatile* addr, uint64 v);
TEXT ·Store64(SB), NOSPLIT, $0-12
	NO_LOCAL_POINTERS
	MOVL	ptr+0(FP), AX
	TESTL	$7, AX
	JZ	2(PC)
	CALL	·panicUnaligned(SB)
	// MOVQ and EMMS were introduced on the Pentium MMX.
	MOVQ	val+4(FP), M0
	MOVQ	M0, (AX)
	EMMS
	// This is essentially a no-op, but it provides required memory fencing.
	// It can be replaced with MFENCE, but MFENCE was introduced only on the Pentium4 (SSE2).
	XORL	AX, AX
	LOCK
	XADDL	AX, (SP)
	RET

// void	·Or8(byte volatile*, byte);
TEXT ·Or8(SB), NOSPLIT, $0-5
	MOVL	ptr+0(FP), AX
	MOVB	val+4(FP), BX
	LOCK
	ORB	BX, (AX)
	RET

// void	·And8(byte volatile*, byte);
TEXT ·And8(SB), NOSPLIT, $0-5
	MOVL	ptr+0(FP), AX
	MOVB	val+4(FP), BX
	LOCK
	ANDB	BX, (AX)
	RET

TEXT ·Store8(SB), NOSPLIT, $0-5
	MOVL	ptr+0(FP), BX
	MOVB	val+4(FP), AX
	XCHGB	AX, 0(BX)
	RET

// func Or(addr *uint32, v uint32)
TEXT ·Or(SB), NOSPLIT, $0-8
	MOVL	ptr+0(FP), AX
	MOVL	val+4(FP), BX
	LOCK
	ORL	BX, (AX)
	RET

// func And(addr *uint32, v uint32)
TEXT ·And(SB), NOSPLIT, $0-8
	MOVL	ptr+0(FP), AX
	MOVL	val+4(FP), BX
	LOCK
	ANDL	BX, (AX)
	RET

// func And32(addr *uint32, v uint32) old uint32
TEXT ·And32(SB), NOSPLIT, $0-12
	MOVL	ptr+0(FP), BX
	MOVL	val+4(FP), CX
casloop:
	MOVL 	CX, DX
	MOVL	(BX), AX
	ANDL	AX, DX
	LOCK
	CMPXCHGL	DX, (BX)
	JNZ casloop
	MOVL 	AX, ret+8(FP)
	RET

// func Or32(addr *uint32, v uint32) old uint32
TEXT ·Or32(SB), NOSPLIT, $0-12
	MOVL	ptr+0(FP), BX
	MOVL	val+4(FP), CX
casloop:
	MOVL 	CX, DX
	MOVL	(BX), AX
	ORL	AX, DX
	LOCK
	CMPXCHGL	DX, (BX)
	JNZ casloop
	MOVL 	AX, ret+8(FP)
	RET

// func And64(addr *uint64, v uint64) old uint64
TEXT ·And64(SB), NOSPLIT, $0-20
	MOVL	ptr+0(FP), BP
	// DI:SI = v
	MOVL	val_lo+4(FP), SI
	MOVL	val_hi+8(FP), DI
	// DX:AX = *addr
	MOVL	0(BP), AX
	MOVL	4(BP), DX
casloop:
	// CX:BX = DX:AX (*addr) & DI:SI (mask)
	MOVL	AX, BX
	MOVL	DX, CX
	ANDL	SI, BX
	ANDL	DI, CX
	LOCK
	CMPXCHG8B	0(BP)
	JNZ casloop
	MOVL	AX, ret_lo+12(FP)
	MOVL	DX, ret_hi+16(FP)
	RET


// func Or64(addr *uint64, v uint64) old uint64
TEXT ·Or64(SB), NOSPLIT, $0-20
	MOVL	ptr+0(FP), BP
	// DI:SI = v
	MOVL	val_lo+4(FP), SI
	MOVL	val_hi+8(FP), DI
	// DX:AX = *addr
	MOVL	0(BP), AX
	MOVL	4(BP), DX
casloop:
	// CX:BX = DX:AX (*addr) | DI:SI (mask)
	MOVL	AX, BX
	MOVL	DX, CX
	ORL	SI, BX
	ORL	DI, CX
	LOCK
	CMPXCHG8B	0(BP)
	JNZ casloop
	MOVL	AX, ret_lo+12(FP)
	MOVL	DX, ret_hi+16(FP)
	RET

// func Anduintptr(addr *uintptr, v uintptr) old uintptr
TEXT ·Anduintptr(SB), NOSPLIT, $0-12
	JMP	·And32(SB)

// func Oruintptr(addr *uintptr, v uintptr) old uintptr
TEXT ·Oruintptr(SB), NOSPLIT, $0-12
	JMP	·Or32(SB)
