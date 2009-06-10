// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//
// System calls and other sys.stuff for arm, Linux
//

TEXT write(SB),7,$0
	MOVW	8(SP), R1
	MOVW	12(SP), R2
    	SWI	$0x00900004  // syscall write
	RET

TEXT exit(SB),7,$0
	SWI         $0x00900001 // exit value in R0

TEXT sys·write(SB),7,$0
	MOVW	8(SP), R1
	MOVW	12(SP), R2
    	SWI	$0x00900004  // syscall write
	RET

TEXT sys·mmap(SB),7,$0
	BL  abort(SB)
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
