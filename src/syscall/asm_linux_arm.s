// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"
#include "funcdata.h"

//
// System calls for arm, Linux
//

#define SYS__LLSEEK 140  /* from zsysnum_linux_arm.go */
// func seek(fd int, offset int64, whence int) (newoffset int64, errno int)
// Implemented in assembly to avoid allocation when
// taking the address of the return value newoffset.
// Underlying system call is
//	llseek(int fd, int offhi, int offlo, int64 *result, int whence)
TEXT ·seek(SB),NOSPLIT,$0-28
	BL	runtime·entersyscall(SB)
	MOVW	$SYS__LLSEEK, R7	// syscall entry
	MOVW	fd+0(FP), R0
	MOVW	offset_hi+8(FP), R1
	MOVW	offset_lo+4(FP), R2
	MOVW	$newoffset_lo+16(FP), R3
	MOVW	whence+12(FP), R4
	SWI	$0
	MOVW	$0xfffff001, R6
	CMP	R6, R0
	BLS	okseek
	MOVW	$0, R1
	MOVW	R1, newoffset_lo+16(FP)
	MOVW	R1, newoffset_hi+20(FP)
	RSB	$0, R0, R0
	MOVW	R0, err+24(FP)
	BL	runtime·exitsyscall(SB)
	RET
okseek:
	// system call filled in newoffset already
	MOVW	$0, R0
	MOVW	R0, err+24(FP)
	BL	runtime·exitsyscall(SB)
	RET

// func rawVforkSyscall(trap, a1, a2, a3 uintptr) (r1, err uintptr)
TEXT ·rawVforkSyscall(SB),NOSPLIT|NOFRAME,$0-24
	MOVW	trap+0(FP), R7	// syscall entry
	MOVW	a1+4(FP), R0
	MOVW	a2+8(FP), R1
	MOVW	a3+12(FP), R2
	SWI	$0
	MOVW	$0xfffff001, R1
	CMP	R1, R0
	BLS	ok
	MOVW	$-1, R1
	MOVW	R1, r1+16(FP)
	RSB	$0, R0, R0
	MOVW	R0, err+20(FP)
	RET
ok:
	MOVW	R0, r1+16(FP)
	MOVW	$0, R0
	MOVW	R0, err+20(FP)
	RET

// func rawSyscallNoError(trap uintptr, a1, a2, a3 uintptr) (r1, r2 uintptr);
TEXT ·rawSyscallNoError(SB),NOSPLIT,$0-24
	MOVW	trap+0(FP), R7	// syscall entry
	MOVW	a1+4(FP), R0
	MOVW	a2+8(FP), R1
	MOVW	a3+12(FP), R2
	SWI	$0
	MOVW	R0, r1+16(FP)
	MOVW	$0, R0
	MOVW	R0, r2+20(FP)
	RET
