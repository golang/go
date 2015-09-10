// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build mips64 mips64le

#include "textflag.h"

#define LL(base, rt)	WORD	$((060<<26)|((base)<<21)|((rt)<<16))
#define LLV(base, rt)	WORD	$((064<<26)|((base)<<21)|((rt)<<16))
#define SC(base, rt)	WORD	$((070<<26)|((base)<<21)|((rt)<<16))
#define SCV(base, rt)	WORD	$((074<<26)|((base)<<21)|((rt)<<16))
#define SYNC	WORD $0xf

TEXT ·SwapInt32(SB),NOSPLIT,$0-20
	JMP	·SwapUint32(SB)

TEXT ·SwapUint32(SB),NOSPLIT,$0-20
	MOVV	addr+0(FP), R2
	MOVW	new+8(FP), R5
	SYNC
	MOVV	R5, R3
	LL(2, 1)	// R1 = *R2
	SC(2, 3)	// *R2 = R3
	BEQ	R3, -3(PC)
	MOVW	R1, old+16(FP)
	SYNC
	RET

TEXT ·SwapInt64(SB),NOSPLIT,$0-24
	JMP	·SwapUint64(SB)

TEXT ·SwapUint64(SB),NOSPLIT,$0-24
	MOVV	addr+0(FP), R2
	MOVV	new+8(FP), R5
	SYNC
	MOVV	R5, R3
	LLV(2, 1)	// R1 = *R2
	SCV(2, 3)	// *R2 = R3
	BEQ	R3, -3(PC)
	MOVV	R1, old+16(FP)
	SYNC
	RET

TEXT ·SwapUintptr(SB),NOSPLIT,$0-24
	JMP	·SwapUint64(SB)

TEXT ·CompareAndSwapInt32(SB),NOSPLIT,$0-17
	JMP	·CompareAndSwapUint32(SB)

TEXT ·CompareAndSwapUint32(SB),NOSPLIT,$0-17
	MOVV	addr+0(FP), R1
	MOVW	old+8(FP), R2
	MOVW	new+12(FP), R5
	SYNC
cas_again:
	MOVV	R5, R3
	LL(1, 4)	// R4 = *R1
	BNE	R2, R4, cas_fail
	SC(1, 3)	// *R1 = R3
	BEQ	R3, cas_again
	MOVV	$1, R1
	MOVB	R1, swapped+16(FP)
	SYNC
	RET
cas_fail:
	MOVV	$0, R1
	JMP	-4(PC)

TEXT ·CompareAndSwapUintptr(SB),NOSPLIT,$0-25
	JMP	·CompareAndSwapUint64(SB)

TEXT ·CompareAndSwapInt64(SB),NOSPLIT,$0-25
	JMP	·CompareAndSwapUint64(SB)

TEXT ·CompareAndSwapUint64(SB),NOSPLIT,$0-25
	MOVV	addr+0(FP), R1
	MOVV	old+8(FP), R2
	MOVV	new+16(FP), R5
	SYNC
cas64_again:
	MOVV	R5, R3
	LLV(1, 4)	// R4 = *R1
	BNE	R2, R4, cas64_fail
	SCV(1, 3)	// *R1 = R3
	BEQ	R3, cas64_again
	MOVV	$1, R1
	MOVB	R1, swapped+24(FP)
	SYNC
	RET
cas64_fail:
	MOVV	$0, R1
	JMP	-4(PC)

TEXT ·AddInt32(SB),NOSPLIT,$0-20
	JMP	·AddUint32(SB)

TEXT ·AddUint32(SB),NOSPLIT,$0-20
	MOVV	addr+0(FP), R2
	MOVW	delta+8(FP), R3
	SYNC
	LL(2, 1)	// R1 = *R2
	ADDU	R1, R3, R4
	MOVV	R4, R1
	SC(2, 4)	// *R2 = R4
	BEQ	R4, -4(PC)
	MOVW	R1, ret+16(FP)
	SYNC
	RET

TEXT ·AddUintptr(SB),NOSPLIT,$0-24
	JMP	·AddUint64(SB)

TEXT ·AddInt64(SB),NOSPLIT,$0-24
	JMP	·AddUint64(SB)

TEXT ·AddUint64(SB),NOSPLIT,$0-24
	MOVV	addr+0(FP), R2
	MOVV	delta+8(FP), R3
	SYNC
	LLV(2, 1)	// R1 = *R2
	ADDVU	R1, R3, R4
	MOVV	R4, R1
	SCV(2, 4)	// *R2 = R4
	BEQ	R4, -4(PC)
	MOVV	R1, ret+16(FP)
	SYNC
	RET

TEXT ·LoadInt32(SB),NOSPLIT,$0-12
	JMP	·LoadUint32(SB)

TEXT ·LoadUint32(SB),NOSPLIT,$0-12
	MOVV	addr+0(FP), R1
	SYNC
	MOVWU	0(R1), R1
	SYNC
	MOVW	R1, val+8(FP)
	RET

TEXT ·LoadInt64(SB),NOSPLIT,$0-16
	JMP	·LoadUint64(SB)

TEXT ·LoadUint64(SB),NOSPLIT,$0-16
	MOVV	addr+0(FP), R1
	SYNC
	MOVV	0(R1), R1
	SYNC
	MOVV	R1, val+8(FP)
	RET

TEXT ·LoadUintptr(SB),NOSPLIT,$0-16
	JMP	·LoadPointer(SB)

TEXT ·LoadPointer(SB),NOSPLIT,$0-16
	JMP	·LoadUint64(SB)

TEXT ·StoreInt32(SB),NOSPLIT,$0-12
	JMP	·StoreUint32(SB)

TEXT ·StoreUint32(SB),NOSPLIT,$0-12
	MOVV	addr+0(FP), R1
	MOVW	val+8(FP), R2
	SYNC
	MOVW	R2, 0(R1)
	SYNC
	RET

TEXT ·StoreInt64(SB),NOSPLIT,$0-16
	JMP	·StoreUint64(SB)

TEXT ·StoreUint64(SB),NOSPLIT,$0-16
	MOVV	addr+0(FP), R1
	MOVV	val+8(FP), R2
	SYNC
	MOVV	R2, 0(R1)
	SYNC
	RET

TEXT ·StoreUintptr(SB),NOSPLIT,$0-16
	JMP	·StoreUint64(SB)
