// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !race

#include "textflag.h"

TEXT ·SwapInt32(SB),NOSPLIT,$0-12
	JMP	·SwapUint32(SB)

TEXT ·SwapUint32(SB),NOSPLIT,$0-12
	MOVL	addr+0(FP), BP
	MOVL	new+4(FP), AX
	XCHGL	AX, 0(BP)
	MOVL	AX, old+8(FP)
	RET

TEXT ·SwapInt64(SB),NOSPLIT,$0-20
	JMP	·SwapUint64(SB)

TEXT ·SwapUint64(SB),NOSPLIT,$0-20
	// no XCHGQ so use CMPXCHG8B loop
	MOVL	addr+0(FP), BP
	TESTL	$7, BP
	JZ	2(PC)
	MOVL	0, AX // crash with nil ptr deref
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
	MOVL	AX, old_lo+12(FP)
	MOVL	DX, old_hi+16(FP)
	RET

TEXT ·SwapUintptr(SB),NOSPLIT,$0-12
	JMP	·SwapUint32(SB)

TEXT ·SwapPointer(SB),NOSPLIT,$0-12
	JMP	·SwapUint32(SB)

TEXT ·CompareAndSwapInt32(SB),NOSPLIT,$0-13
	JMP	·CompareAndSwapUint32(SB)

TEXT ·CompareAndSwapUint32(SB),NOSPLIT,$0-13
	MOVL	addr+0(FP), BP
	MOVL	old+4(FP), AX
	MOVL	new+8(FP), CX
	// CMPXCHGL was introduced on the 486.
	LOCK
	CMPXCHGL	CX, 0(BP)
	SETEQ	swapped+12(FP)
	RET

TEXT ·CompareAndSwapUintptr(SB),NOSPLIT,$0-13
	JMP	·CompareAndSwapUint32(SB)

TEXT ·CompareAndSwapPointer(SB),NOSPLIT,$0-13
	JMP	·CompareAndSwapUint32(SB)

TEXT ·CompareAndSwapInt64(SB),NOSPLIT,$0-21
	JMP	·CompareAndSwapUint64(SB)

TEXT ·CompareAndSwapUint64(SB),NOSPLIT,$0-21
	MOVL	addr+0(FP), BP
	TESTL	$7, BP
	JZ	2(PC)
	MOVL	0, AX // crash with nil ptr deref
	MOVL	old_lo+4(FP), AX
	MOVL	old_hi+8(FP), DX
	MOVL	new_lo+12(FP), BX
	MOVL	new_hi+16(FP), CX
	// CMPXCHG8B was introduced on the Pentium.
	LOCK
	CMPXCHG8B	0(BP)
	SETEQ	swapped+20(FP)
	RET

TEXT ·AddInt32(SB),NOSPLIT,$0-12
	JMP	·AddUint32(SB)

TEXT ·AddUint32(SB),NOSPLIT,$0-12
	MOVL	addr+0(FP), BP
	MOVL	delta+4(FP), AX
	MOVL	AX, CX
	// XADD was introduced on the 486.
	LOCK
	XADDL	AX, 0(BP)
	ADDL	AX, CX
	MOVL	CX, new+8(FP)
	RET

TEXT ·AddUintptr(SB),NOSPLIT,$0-12
	JMP	·AddUint32(SB)

TEXT ·AddInt64(SB),NOSPLIT,$0-20
	JMP	·AddUint64(SB)

TEXT ·AddUint64(SB),NOSPLIT,$0-20
	// no XADDQ so use CMPXCHG8B loop
	MOVL	addr+0(FP), BP
	TESTL	$7, BP
	JZ	2(PC)
	MOVL	0, AX // crash with nil ptr deref
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
	MOVL	BX, new_lo+12(FP)
	MOVL	CX, new_hi+16(FP)
	RET

TEXT ·LoadInt32(SB),NOSPLIT,$0-8
	JMP	·LoadUint32(SB)

TEXT ·LoadUint32(SB),NOSPLIT,$0-8
	MOVL	addr+0(FP), AX
	MOVL	0(AX), AX
	MOVL	AX, val+4(FP)
	RET

TEXT ·LoadInt64(SB),NOSPLIT,$0-12
	JMP	·LoadUint64(SB)

TEXT ·LoadUint64(SB),NOSPLIT,$0-12
	MOVL	addr+0(FP), AX
	TESTL	$7, AX
	JZ	2(PC)
	MOVL	0, AX // crash with nil ptr deref
	// MOVQ and EMMS were introduced on the Pentium MMX.
	// MOVQ (%EAX), %MM0
	BYTE $0x0f; BYTE $0x6f; BYTE $0x00
	// MOVQ %MM0, 0x8(%ESP)
	BYTE $0x0f; BYTE $0x7f; BYTE $0x44; BYTE $0x24; BYTE $0x08
	EMMS
	RET

TEXT ·LoadUintptr(SB),NOSPLIT,$0-8
	JMP	·LoadUint32(SB)

TEXT ·LoadPointer(SB),NOSPLIT,$0-8
	JMP	·LoadUint32(SB)

TEXT ·StoreInt32(SB),NOSPLIT,$0-8
	JMP	·StoreUint32(SB)

TEXT ·StoreUint32(SB),NOSPLIT,$0-8
	MOVL	addr+0(FP), BP
	MOVL	val+4(FP), AX
	XCHGL	AX, 0(BP)
	RET

TEXT ·StoreInt64(SB),NOSPLIT,$0-12
	JMP	·StoreUint64(SB)

TEXT ·StoreUint64(SB),NOSPLIT,$0-12
	MOVL	addr+0(FP), AX
	TESTL	$7, AX
	JZ	2(PC)
	MOVL	0, AX // crash with nil ptr deref
	// MOVQ and EMMS were introduced on the Pentium MMX.
	// MOVQ 0x8(%ESP), %MM0
	BYTE $0x0f; BYTE $0x6f; BYTE $0x44; BYTE $0x24; BYTE $0x08
	// MOVQ %MM0, (%EAX)
	BYTE $0x0f; BYTE $0x7f; BYTE $0x00 
	EMMS
	// This is essentially a no-op, but it provides required memory fencing.
	// It can be replaced with MFENCE, but MFENCE was introduced only on the Pentium4 (SSE2).
	XORL	AX, AX
	LOCK
	XADDL	AX, (SP)
	RET

TEXT ·StoreUintptr(SB),NOSPLIT,$0-8
	JMP	·StoreUint32(SB)

TEXT ·StorePointer(SB),NOSPLIT,$0-8
	JMP	·StoreUint32(SB)
