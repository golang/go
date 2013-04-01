// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !race

// Linux/ARM atomic operations.

// Because there is so much variation in ARM devices,
// the Linux kernel provides an appropriate compare-and-swap
// implementation at address 0xffff0fc0.  Caller sets:
//	R0 = old value
//	R1 = new value
//	R2 = addr
//	LR = return address
// The function returns with CS true if the swap happened.
// http://lxr.linux.no/linux+v2.6.37.2/arch/arm/kernel/entry-armv.S#L850
// On older kernels (before 2.6.24) the function can incorrectly
// report a conflict, so we have to double-check the compare ourselves
// and retry if necessary.
//
// http://git.kernel.org/?p=linux/kernel/git/torvalds/linux-2.6.git;a=commit;h=b49c0f24cf6744a3f4fd09289fe7cade349dead5
//
TEXT cas<>(SB),7,$0
	MOVW	$0xffff0fc0, PC

TEXT ·CompareAndSwapInt32(SB),7,$0
	B	·CompareAndSwapUint32(SB)

// Implement using kernel cas for portability.
TEXT ·CompareAndSwapUint32(SB),7,$0
	MOVW	addr+0(FP), R2
	MOVW	old+4(FP), R0
casagain:
	MOVW	new+8(FP), R1
	BL cas<>(SB)
	BCC	cascheck
	MOVW	$1, R0
casret:
	MOVW	R0, ret+12(FP)
	RET
cascheck:
	// Kernel lies; double-check.
	MOVW	addr+0(FP), R2
	MOVW	old+4(FP), R0
	MOVW	0(R2), R3
	CMP	R0, R3
	BEQ	casagain
	MOVW	$0, R0
	B	casret

TEXT ·CompareAndSwapUintptr(SB),7,$0
	B	·CompareAndSwapUint32(SB)

TEXT ·CompareAndSwapPointer(SB),7,$0
	B	·CompareAndSwapUint32(SB)

TEXT ·AddInt32(SB),7,$0
	B	·AddUint32(SB)

// Implement using kernel cas for portability.
TEXT ·AddUint32(SB),7,$0
	MOVW	addr+0(FP), R2
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

TEXT cas64<>(SB),7,$0
	MOVW	$0xffff0f60, PC // __kuser_cmpxchg64: Linux-3.1 and above

TEXT kernelCAS64<>(SB),7,$0
	// int (*__kuser_cmpxchg64_t)(const int64_t *oldval, const int64_t *newval, volatile int64_t *ptr);
	MOVW	addr+0(FP), R2 // ptr
	// make unaligned atomic access panic
	AND.S	$7, R2, R1
	BEQ 	2(PC)
	MOVW	R1, (R1)
	MOVW	$4(FP), R0 // oldval
	MOVW	$12(FP), R1 // newval
	BL		cas64<>(SB)
	MOVW.CS	$1, R0 // C is set if the kernel has changed *ptr
	MOVW.CC	$0, R0
	MOVW	R0, 20(FP)
	RET

TEXT generalCAS64<>(SB),7,$20
	// bool runtime·cas64(uint64 volatile *addr, uint64 *old, uint64 new)
	MOVW	addr+0(FP), R0
	// make unaligned atomic access panic
	AND.S	$7, R0, R1
	BEQ 	2(PC)
	MOVW	R1, (R1)
	MOVW	R0, 4(R13)
	MOVW	$4(FP), R1 // oldval
	MOVW	R1, 8(R13)
	MOVW	newlo+12(FP), R2
	MOVW	R2, 12(R13)
	MOVW	newhi+16(FP), R3
	MOVW	R3, 16(R13)
	BL  	runtime·cas64(SB)
	MOVW	R0, 20(FP)
	RET

GLOBL armCAS64(SB), $4

TEXT setupAndCallCAS64<>(SB),7,$-4
	MOVW	$0xffff0ffc, R0 // __kuser_helper_version
	MOVW	(R0), R0
	// __kuser_cmpxchg64 only present if helper version >= 5
	CMP 	$5, R0
	MOVW.CS	$kernelCAS64<>(SB), R1
	MOVW.CS	R1, armCAS64(SB)
	MOVW.CS	R1, PC
	MOVB	runtime·armArch(SB), R0
	// LDREXD, STREXD only present on ARMv6K or higher
	CMP		$6, R0 // TODO(minux): how to differentiate ARMv6 with ARMv6K?
	MOVW.CS	$·armCompareAndSwapUint64(SB), R1
	MOVW.CS	R1, armCAS64(SB)
	MOVW.CS	R1, PC
	// we are out of luck, can only use runtime's emulated 64-bit cas
	MOVW	$generalCAS64<>(SB), R1
	MOVW	R1, armCAS64(SB)
	MOVW	R1, PC

TEXT ·CompareAndSwapInt64(SB),7,$0
	B   	·CompareAndSwapUint64(SB)

TEXT ·CompareAndSwapUint64(SB),7,$-4
	MOVW	armCAS64(SB), R0
	CMP 	$0, R0
	MOVW.NE	R0, PC
	B		setupAndCallCAS64<>(SB)

TEXT ·AddInt64(SB),7,$0
	B	·addUint64(SB)

TEXT ·AddUint64(SB),7,$0
	B	·addUint64(SB)

TEXT ·LoadInt32(SB),7,$0
	B	·LoadUint32(SB)

TEXT ·LoadUint32(SB),7,$0
	MOVW	addr+0(FP), R2
loadloop1:
	MOVW	0(R2), R0
	MOVW	R0, R1
	BL	cas<>(SB)
	BCC	loadloop1
	MOVW	R1, val+4(FP)
	RET

TEXT ·LoadInt64(SB),7,$0
	B	·loadUint64(SB)

TEXT ·LoadUint64(SB),7,$0
	B	·loadUint64(SB)

TEXT ·LoadUintptr(SB),7,$0
	B	·LoadUint32(SB)

TEXT ·LoadPointer(SB),7,$0
	B	·LoadUint32(SB)

TEXT ·StoreInt32(SB),7,$0
	B	·StoreUint32(SB)

TEXT ·StoreUint32(SB),7,$0
	MOVW	addr+0(FP), R2
	MOVW	val+4(FP), R1
storeloop1:
	MOVW	0(R2), R0
	BL	cas<>(SB)
	BCC	storeloop1
	RET

TEXT ·StoreInt64(SB),7,$0
	B	·storeUint64(SB)

TEXT ·StoreUint64(SB),7,$0
	B	·storeUint64(SB)

TEXT ·StoreUintptr(SB),7,$0
	B	·StoreUint32(SB)

TEXT ·StorePointer(SB),7,$0
	B	·StoreUint32(SB)
