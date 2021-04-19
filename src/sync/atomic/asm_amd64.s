// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Note: some of these functions are semantically inlined
// by the compiler (in src/cmd/compile/internal/gc/ssa.go).

// +build !race

#include "textflag.h"

TEXT ·SwapInt32(SB),NOSPLIT,$0-20
	JMP	·SwapUint32(SB)

TEXT ·SwapUint32(SB),NOSPLIT,$0-20
	MOVQ	addr+0(FP), BP
	MOVL	new+8(FP), AX
	XCHGL	AX, 0(BP)
	MOVL	AX, old+16(FP)
	RET

TEXT ·SwapInt64(SB),NOSPLIT,$0-24
	JMP	·SwapUint64(SB)

TEXT ·SwapUint64(SB),NOSPLIT,$0-24
	MOVQ	addr+0(FP), BP
	MOVQ	new+8(FP), AX
	XCHGQ	AX, 0(BP)
	MOVQ	AX, old+16(FP)
	RET

TEXT ·SwapUintptr(SB),NOSPLIT,$0-24
	JMP	·SwapUint64(SB)

TEXT ·CompareAndSwapInt32(SB),NOSPLIT,$0-17
	JMP	·CompareAndSwapUint32(SB)

TEXT ·CompareAndSwapUint32(SB),NOSPLIT,$0-17
	MOVQ	addr+0(FP), BP
	MOVL	old+8(FP), AX
	MOVL	new+12(FP), CX
	LOCK
	CMPXCHGL	CX, 0(BP)
	SETEQ	swapped+16(FP)
	RET

TEXT ·CompareAndSwapUintptr(SB),NOSPLIT,$0-25
	JMP	·CompareAndSwapUint64(SB)

TEXT ·CompareAndSwapInt64(SB),NOSPLIT,$0-25
	JMP	·CompareAndSwapUint64(SB)

TEXT ·CompareAndSwapUint64(SB),NOSPLIT,$0-25
	MOVQ	addr+0(FP), BP
	MOVQ	old+8(FP), AX
	MOVQ	new+16(FP), CX
	LOCK
	CMPXCHGQ	CX, 0(BP)
	SETEQ	swapped+24(FP)
	RET

TEXT ·AddInt32(SB),NOSPLIT,$0-20
	JMP	·AddUint32(SB)

TEXT ·AddUint32(SB),NOSPLIT,$0-20
	MOVQ	addr+0(FP), BP
	MOVL	delta+8(FP), AX
	MOVL	AX, CX
	LOCK
	XADDL	AX, 0(BP)
	ADDL	AX, CX
	MOVL	CX, new+16(FP)
	RET

TEXT ·AddUintptr(SB),NOSPLIT,$0-24
	JMP	·AddUint64(SB)

TEXT ·AddInt64(SB),NOSPLIT,$0-24
	JMP	·AddUint64(SB)

TEXT ·AddUint64(SB),NOSPLIT,$0-24
	MOVQ	addr+0(FP), BP
	MOVQ	delta+8(FP), AX
	MOVQ	AX, CX
	LOCK
	XADDQ	AX, 0(BP)
	ADDQ	AX, CX
	MOVQ	CX, new+16(FP)
	RET

TEXT ·LoadInt32(SB),NOSPLIT,$0-12
	JMP	·LoadUint32(SB)

TEXT ·LoadUint32(SB),NOSPLIT,$0-12
	MOVQ	addr+0(FP), AX
	MOVL	0(AX), AX
	MOVL	AX, val+8(FP)
	RET

TEXT ·LoadInt64(SB),NOSPLIT,$0-16
	JMP	·LoadUint64(SB)

TEXT ·LoadUint64(SB),NOSPLIT,$0-16
	MOVQ	addr+0(FP), AX
	MOVQ	0(AX), AX
	MOVQ	AX, val+8(FP)
	RET

TEXT ·LoadUintptr(SB),NOSPLIT,$0-16
	JMP	·LoadPointer(SB)

TEXT ·LoadPointer(SB),NOSPLIT,$0-16
	MOVQ	addr+0(FP), AX
	MOVQ	0(AX), AX
	MOVQ	AX, val+8(FP)
	RET

TEXT ·StoreInt32(SB),NOSPLIT,$0-12
	JMP	·StoreUint32(SB)

TEXT ·StoreUint32(SB),NOSPLIT,$0-12
	MOVQ	addr+0(FP), BP
	MOVL	val+8(FP), AX
	XCHGL	AX, 0(BP)
	RET

TEXT ·StoreInt64(SB),NOSPLIT,$0-16
	JMP	·StoreUint64(SB)

TEXT ·StoreUint64(SB),NOSPLIT,$0-16
	MOVQ	addr+0(FP), BP
	MOVQ	val+8(FP), AX
	XCHGQ	AX, 0(BP)
	RET

TEXT ·StoreUintptr(SB),NOSPLIT,$0-16
	JMP	·StoreUint64(SB)
