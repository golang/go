// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//
// System call support for AMD64, Darwin
//

// func Syscall(trap int64, a1, a2, a3 int64) (r1, r2, err int64);
// Trap # in AX, args in DI SI DX, return in AX DX

TEXT	syscall·Syscall(SB),7,$-8
	MOVQ	16(SP), DI
	MOVQ	24(SP), SI
	MOVQ	32(SP), DX
	MOVQ	8(SP), AX	// syscall entry
	ADDQ	$0x2000000, AX
	SYSCALL
	JCC	5(PC)
	MOVQ	$-1, 40(SP)	// r1
	MOVQ	$0, 48(SP)	// r2
	MOVQ	AX, 56(SP)  // errno
	RET
	MOVQ	AX, 40(SP)	// r1
	MOVQ	DX, 48(SP)	// r2
	MOVQ	$0, 56(SP)	// errno
	RET

// conversion operators - really just casts
TEXT	syscall·AddrToInt(SB),7,$-8
	MOVQ	8(SP), AX
	MOVQ	AX, 16(SP)
	RET

TEXT	syscall·StatToInt(SB),7,$-8
	MOVQ	8(SP), AX
	MOVQ	AX, 16(SP)
	RET
