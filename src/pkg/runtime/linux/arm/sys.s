// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//
// System calls and other sys.stuff for arm, Linux
//

#define SYS_BASE 0x00900000
#define SYS_exit (SYS_BASE + 1)
#define SYS_write (SYS_BASE + 4)
#define SYS_mmap2 (SYS_BASE + 192)

TEXT write(SB),7,$0
	MOVW	0(FP), R0
	MOVW	4(FP), R1
	MOVW	8(FP), R2
    	SWI	$SYS_write
	RET

TEXT exit(SB),7,$0
	// Exit value already in R0
	SWI	$SYS_exit

TEXT sys·mmap(SB),7,$0
	MOVW	0(FP), R0
	MOVW	4(FP), R1
	MOVW	8(FP), R2
	MOVW	12(FP), R3
	MOVW	16(FP), R4
	MOVW	20(FP), R5
	SWI	$SYS_mmap2
	RET

// int64 futex(int32 *uaddr, int32 op, int32 val,
//	struct timespec *timeout, int32 *uaddr2, int32 val2);
TEXT futex(SB),7,$0
	BL  abort(SB)
	RET

// int64 clone(int32 flags, void *stack, M *m, G *g, void (*fn)(void));
TEXT clone(SB),7,$0
	BL  abort(SB)
    	RET
