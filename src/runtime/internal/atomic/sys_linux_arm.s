// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// Use kernel version instead of native armcas in asm_arm.s.
// See ../../../sync/atomic/asm_linux_arm.s for details.
TEXT cas<>(SB),NOSPLIT,$0
	MOVW	$0xffff0fc0, R15 // R15 is hardware PC.

TEXT runtime∕internal∕atomic·Cas(SB),NOSPLIT,$0
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

TEXT runtime∕internal∕atomic·Casp1(SB),NOSPLIT,$0
	B	runtime∕internal∕atomic·Cas(SB)

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

TEXT runtime∕internal∕atomic·Load(SB),NOSPLIT,$0-8
	MOVW	addr+0(FP), R0
	MOVW	(R0), R1

	MOVB	runtime·goarm(SB), R11
	CMP	$7, R11
	BGE	native_barrier
	BL	memory_barrier<>(SB)
	B	prolog
native_barrier:
	DMB	MB_ISH

prolog:
	MOVW	R1, ret+4(FP)
	RET
