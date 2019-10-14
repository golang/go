// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

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
// https://git.kernel.org/?p=linux/kernel/git/torvalds/linux-2.6.git;a=commit;h=b49c0f24cf6744a3f4fd09289fe7cade349dead5
//
TEXT cas<>(SB),NOSPLIT,$0
	MOVW	$0xffff0fc0, R15 // R15 is hardware PC.

TEXT runtime∕internal∕atomic·Cas(SB),NOSPLIT|NOFRAME,$0
	MOVB	runtime·goarm(SB), R11
	CMP	$7, R11
	BLT	2(PC)
	JMP	·armcas(SB)
	JMP	·kernelcas<>(SB)

TEXT runtime∕internal∕atomic·kernelcas<>(SB),NOSPLIT,$0
	MOVW	ptr+0(FP), R2
	// trigger potential paging fault here,
	// because we don't know how to traceback through __kuser_cmpxchg
	MOVW    (R2), R0
	MOVW	old+4(FP), R0
loop:
	MOVW	new+8(FP), R1
	BL	cas<>(SB)
	BCC	check
	MOVW	$1, R0
	MOVB	R0, ret+12(FP)
	RET
check:
	// Kernel lies; double-check.
	MOVW	ptr+0(FP), R2
	MOVW	old+4(FP), R0
	MOVW	0(R2), R3
	CMP	R0, R3
	BEQ	loop
	MOVW	$0, R0
	MOVB	R0, ret+12(FP)
	RET

// As for cas, memory barriers are complicated on ARM, but the kernel
// provides a user helper. ARMv5 does not support SMP and has no
// memory barrier instruction at all. ARMv6 added SMP support and has
// a memory barrier, but it requires writing to a coprocessor
// register. ARMv7 introduced the DMB instruction, but it's expensive
// even on single-core devices. The kernel helper takes care of all of
// this for us.

// Use kernel helper version of memory_barrier, when compiled with GOARM < 7.
TEXT memory_barrier<>(SB),NOSPLIT|NOFRAME,$0
	MOVW	$0xffff0fa0, R15 // R15 is hardware PC.

TEXT	·Load(SB),NOSPLIT,$0-8
	MOVW	addr+0(FP), R0
	MOVW	(R0), R1

	MOVB	runtime·goarm(SB), R11
	CMP	$7, R11
	BGE	native_barrier
	BL	memory_barrier<>(SB)
	B	end
native_barrier:
	DMB	MB_ISH
end:
	MOVW	R1, ret+4(FP)
	RET

TEXT	·Store(SB),NOSPLIT,$0-8
	MOVW	addr+0(FP), R1
	MOVW	v+4(FP), R2

	MOVB	runtime·goarm(SB), R8
	CMP	$7, R8
	BGE	native_barrier
	BL	memory_barrier<>(SB)
	B	store
native_barrier:
	DMB	MB_ISH

store:
	MOVW	R2, (R1)

	CMP	$7, R8
	BGE	native_barrier2
	BL	memory_barrier<>(SB)
	RET
native_barrier2:
	DMB	MB_ISH
	RET

TEXT	·Load8(SB),NOSPLIT,$0-5
	MOVW	addr+0(FP), R0
	MOVB	(R0), R1

	MOVB	runtime·goarm(SB), R11
	CMP	$7, R11
	BGE	native_barrier
	BL	memory_barrier<>(SB)
	B	end
native_barrier:
	DMB	MB_ISH
end:
	MOVB	R1, ret+4(FP)
	RET

