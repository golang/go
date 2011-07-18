// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

TEXT ·CompareAndSwapInt32(SB),7,$0
	JMP	·CompareAndSwapUint32(SB)

TEXT ·CompareAndSwapUint32(SB),7,$0
	MOVL	valptr+0(FP), BP
	MOVL	old+4(FP), AX
	MOVL	new+8(FP), CX
	// CMPXCHGL was introduced on the 486.
	LOCK
	CMPXCHGL	CX, 0(BP)
	SETEQ	ret+12(FP)
	RET

TEXT ·CompareAndSwapUintptr(SB),7,$0
	JMP	·CompareAndSwapUint32(SB)

TEXT ·CompareAndSwapInt64(SB),7,$0
	JMP	·CompareAndSwapUint64(SB)

TEXT ·CompareAndSwapUint64(SB),7,$0
	MOVL	valptr+0(FP), BP
	MOVL	oldlo+4(FP), AX
	MOVL	oldhi+8(FP), DX
	MOVL	newlo+12(FP), BX
	MOVL	newhi+16(FP), CX
	// CMPXCHG8B was introduced on the Pentium.
	LOCK
	CMPXCHG8B	0(BP)
	SETEQ	ret+20(FP)
	RET

TEXT ·AddInt32(SB),7,$0
	JMP	·AddUint32(SB)

TEXT ·AddUint32(SB),7,$0
	MOVL	valptr+0(FP), BP
	MOVL	delta+4(FP), AX
	MOVL	AX, CX
	// XADD was introduced on the 486.
	LOCK
	XADDL	AX, 0(BP)
	ADDL	AX, CX
	MOVL	CX, ret+8(FP)
	RET

TEXT ·AddUintptr(SB),7,$0
	JMP	·AddUint32(SB)

TEXT ·AddInt64(SB),7,$0
	JMP	·AddUint64(SB)

TEXT ·AddUint64(SB),7,$0
	// no XADDQ so use CMPXCHG8B loop
	MOVL	valptr+0(FP), BP
	// DI:SI = delta
	MOVL	deltalo+4(FP), SI
	MOVL	deltahi+8(FP), DI
	// DX:AX = *valptr
	MOVL	0(BP), AX
	MOVL	4(BP), DX
addloop:
	// CX:BX = DX:AX (*valptr) + DI:SI (delta)
	MOVL	AX, BX
	MOVL	DX, CX
	ADDL	SI, BX
	ADCL	DI, CX

	// if *valptr == DX:AX {
	//	*valptr = CX:BX
	// } else {
	//	DX:AX = *valptr
	// }
	// all in one instruction
	LOCK
	CMPXCHG8B	0(BP)

	JNZ	addloop

	// success
	// return CX:BX
	MOVL	BX, retlo+12(FP)
	MOVL	CX, rethi+16(FP)
	RET

TEXT ·LoadInt32(SB),7,$0
	JMP	·LoadUint32(SB)

TEXT ·LoadUint32(SB),7,$0
	MOVL	addrptr+0(FP), AX
	MOVL	0(AX), AX
	MOVL	AX, ret+4(FP)
	RET
