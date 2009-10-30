// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//
// System calls and other sys.stuff for arm, Linux
//

#include "arm/asm.h"

#define SYS_BASE 0x00900000
#define SYS_exit (SYS_BASE + 1)
#define SYS_write (SYS_BASE + 4)
#define SYS_clone (SYS_BASE + 120)
#define SYS_mmap2 (SYS_BASE + 192)
#define SYS_gettid (SYS_BASE + 224)
#define SYS_futex (SYS_BASE + 240)
#define SYS_exit_group (SYS_BASE + 248)

TEXT write(SB),7,$0
	MOVW	0(FP), R0
	MOVW	4(FP), R1
	MOVW	8(FP), R2
    	SWI	$SYS_write
	RET

TEXT exit(SB),7,$-4
	MOVW	0(FP), R0
	SWI	$SYS_exit_group
	MOVW	$1234, R0
	MOVW	$1002, R1
	MOVW	R0, (R1)	// fail hard

TEXT exit1(SB),7,$-4
	MOVW	0(FP), R0
	SWI	$SYS_exit
	MOVW	$1234, R0
	MOVW	$1003, R1
	MOVW	R0, (R1)	// fail hard

TEXT runtime·mmap(SB),7,$0
	MOVW	0(FP), R0
	MOVW	4(FP), R1
	MOVW	8(FP), R2
	MOVW	12(FP), R3
	MOVW	16(FP), R4
	MOVW	20(FP), R5
	SWI	$SYS_mmap2
	RET

// int32 futex(int32 *uaddr, int32 op, int32 val,
//	struct timespec *timeout, int32 *uaddr2, int32 val2);
TEXT futex(SB),7,$0
	MOVW	4(SP), R0
	MOVW	8(SP), R1
	MOVW	12(SP), R2
	MOVW	16(SP), R3
	MOVW	20(SP), R4
	MOVW	24(SP), R5
	SWI	$SYS_futex
	RET


// int32 clone(int32 flags, void *stack, M *m, G *g, void (*fn)(void));
TEXT clone(SB),7,$0
	MOVW	flags+0(FP), R0
	MOVW	stack+4(FP), R1
	MOVW	$0, R2	// parent tid ptr
	MOVW	$0, R3	// tls_val
	MOVW	$0, R4	// child tid ptr
	MOVW	$0, R5

	// Copy m, g, fn off parent stack for use by child.
	// TODO(kaib): figure out which registers are clobbered by clone and avoid stack copying
	MOVW	$-16(R1), R1
	MOVW	mm+8(FP), R6
	MOVW	R6, 0(R1)
	MOVW	gg+12(FP), R6
	MOVW	R6, 4(R1)
	MOVW	fn+16(FP), R6
	MOVW	R6, 8(R1)
	MOVW	$1234, R6
	MOVW	R6, 12(R1)

	SWI	$SYS_clone

	// In parent, return.
	CMP	$0, R0
	BEQ	2(PC)
	RET

	// Paranoia: check that SP is as we expect. Use R13 to avoid linker 'fixup'
	MOVW	12(R13), R0
	MOVW	$1234, R1
	CMP	R0, R1
	BEQ	2(PC)
	B	abort(SB)

	MOVW	0(R13), m
	MOVW	4(R13), g

	// paranoia; check they are not nil
	MOVW	0(m), R0
	MOVW	0(g), R0

	BL	emptyfunc(SB)	// fault if stack check is wrong

	// Initialize m->procid to Linux tid
	SWI	$SYS_gettid
	MOVW	R0, m_procid(m)

	// Call fn
	MOVW	8(R13), R0
	MOVW	$16(R13), R13
	BL	(R0)

	MOVW	$0, R0
	MOVW	R0, 4(R13)
	BL	exit1(SB)

	// It shouldn't return
	MOVW	$1234, R0
	MOVW	$1005, R1
	MOVW	R0, (R1)


