// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Note: some of these functions are semantically inlined
// by the compiler (in src/cmd/compile/internal/gc/ssa.go).

#include "textflag.h"

TEXT ·SwapInt32(SB),NOSPLIT,$0-12
	JMP	·SwapUint32(SB)

TEXT ·SwapUint32(SB),NOSPLIT,$0-12
	MOVL	addr+0(FP), BX
	MOVL	new+4(FP), AX
	XCHGL	AX, 0(BX)
	MOVL	AX, old+8(FP)
	RET

TEXT ·SwapInt64(SB),NOSPLIT,$0-24
	JMP	·SwapUint64(SB)

TEXT ·SwapUint64(SB),NOSPLIT,$0-24
	MOVL	addr+0(FP), BX
	TESTL	$7, BX
	JZ	2(PC)
	MOVL	0, BX // crash with nil ptr deref
	MOVQ	new+8(FP), AX
	XCHGQ	AX, 0(BX)
	MOVQ	AX, old+16(FP)
	RET

TEXT ·SwapUintptr(SB),NOSPLIT,$0-12
	JMP	·SwapUint32(SB)

TEXT ·CompareAndSwapInt32(SB),NOSPLIT,$0-17
	JMP	·CompareAndSwapUint32(SB)

TEXT ·CompareAndSwapUint32(SB),NOSPLIT,$0-17
	MOVL	addr+0(FP), BX
	MOVL	old+4(FP), AX
	MOVL	new+8(FP), CX
	LOCK
	CMPXCHGL	CX, 0(BX)
	SETEQ	swapped+16(FP)
	RET

TEXT ·CompareAndSwapUintptr(SB),NOSPLIT,$0-17
	JMP	·CompareAndSwapUint32(SB)

TEXT ·CompareAndSwapInt64(SB),NOSPLIT,$0-25
	JMP	·CompareAndSwapUint64(SB)

TEXT ·CompareAndSwapUint64(SB),NOSPLIT,$0-25
	MOVL	addr+0(FP), BX
	MOVQ	old+8(FP), AX
	MOVQ	new+16(FP), CX
	LOCK
	CMPXCHGQ	CX, 0(BX)
	SETEQ	swapped+24(FP)
	RET

TEXT ·AddInt32(SB),NOSPLIT,$0-12
	JMP	·AddUint32(SB)

TEXT ·AddUint32(SB),NOSPLIT,$0-12
	MOVL	addr+0(FP), BX
	MOVL	delta+4(FP), AX
	MOVL	AX, CX
	LOCK
	XADDL	AX, 0(BX)
	ADDL	AX, CX
	MOVL	CX, new+8(FP)
	RET

TEXT ·AddUintptr(SB),NOSPLIT,$0-12
	JMP	·AddUint32(SB)

TEXT ·AddInt64(SB),NOSPLIT,$0-24
	JMP	·AddUint64(SB)

TEXT ·AddUint64(SB),NOSPLIT,$0-24
	MOVL	addr+0(FP), BX
	MOVQ	delta+8(FP), AX
	MOVQ	AX, CX
	LOCK
	XADDQ	AX, 0(BX)
	ADDQ	AX, CX
	MOVQ	CX, new+16(FP)
	RET

TEXT ·LoadInt32(SB),NOSPLIT,$0-12
	JMP	·LoadUint32(SB)

TEXT ·LoadUint32(SB),NOSPLIT,$0-12
	MOVL	addr+0(FP), AX
	MOVL	0(AX), AX
	MOVL	AX, val+8(FP)
	RET

TEXT ·LoadInt64(SB),NOSPLIT,$0-16
	JMP	·LoadUint64(SB)

TEXT ·LoadUint64(SB),NOSPLIT,$0-16
	MOVL	addr+0(FP), AX
	MOVQ	0(AX), AX
	MOVQ	AX, val+8(FP)
	RET

TEXT ·LoadUintptr(SB),NOSPLIT,$0-12
	JMP	·LoadPointer(SB)

TEXT ·LoadPointer(SB),NOSPLIT,$0-12
	MOVL	addr+0(FP), AX
	MOVL	0(AX), AX
	MOVL	AX, val+8(FP)
	RET

TEXT ·StoreInt32(SB),NOSPLIT,$0-8
	JMP	·StoreUint32(SB)

TEXT ·StoreUint32(SB),NOSPLIT,$0-8
	MOVL	addr+0(FP), BX
	MOVL	val+4(FP), AX
	XCHGL	AX, 0(BX)
	RET

TEXT ·StoreInt64(SB),NOSPLIT,$0-16
	JMP	·StoreUint64(SB)

TEXT ·StoreUint64(SB),NOSPLIT,$0-16
	MOVL	addr+0(FP), BX
	MOVQ	val+8(FP), AX
	XCHGQ	AX, 0(BX)
	RET

TEXT ·StoreUintptr(SB),NOSPLIT,$0-8
	JMP	·StoreUint32(SB)
