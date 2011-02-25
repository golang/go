// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Linux/ARM atomic operations.

// Because there is so much variation in ARM devices,
// the Linux kernel provides an appropriate compare-and-swap
// implementation at address 0xffff0fc0.  Caller sets:
//	R0 = old value
//	R1 = new value
//	R2 = valptr
//	LR = return address
// The function returns with CS true if the swap happened.
// http://lxr.linux.no/linux+v2.6.37.2/arch/arm/kernel/entry-armv.S#L850
TEXT cas<>(SB),7,$0
	MOVW	$0xffff0fc0, PC

TEXT ·CompareAndSwapInt32(SB),7,$0
	B	·CompareAndSwapUint32(SB)

// Implement using kernel cas for portability.
TEXT ·CompareAndSwapUint32(SB),7,$0
	MOVW	valptr+0(FP), R2
	MOVW	old+4(FP), R0
	MOVW	new+8(FP), R1
	BL cas<>(SB)
	MOVW	$0, R0
	MOVW.CS	$1, R0
	MOVW	R0, ret+12(FP)
	RET

TEXT ·CompareAndSwapUintptr(SB),7,$0
	B	·CompareAndSwapUint32(SB)

TEXT ·AddInt32(SB),7,$0
	B	·AddUint32(SB)

// Implement using kernel cas for portability.
TEXT ·AddUint32(SB),7,$0
	MOVW	valptr+0(FP), R2
	MOVW	delta+4(FP), R4
addloop1:
	MOVW	0(R2), R0
	MOVW	R0, R1
	ADD	R4, R1
	BL	cas<>(SB)
	BCC	addloop1
	MOVW	R1, ret+8(FP)
	RET

TEXT ·AddUintptr(SB),7,$0
	B	·AddUint32(SB)

// The kernel provides no 64-bit compare-and-swap,
// so use native ARM instructions, which will only work on
// ARM 11 and later devices.
TEXT ·CompareAndSwapInt64(SB),7,$0
	B	·armCompareAndSwapUint64(SB)

TEXT ·CompareAndSwapUint64(SB),7,$0
	B	·armCompareAndSwapUint64(SB)

TEXT ·AddInt64(SB),7,$0
	B	·armAddUint64(SB)

TEXT ·AddUint64(SB),7,$0
	B	·armAddUint64(SB)
