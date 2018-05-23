// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"
#include "funcdata.h"

#define SYS_ERRSTR 41	/* from zsysnum_plan9.go */
#define SYS_SEEK 39	/* from zsysnum_plan9.go */

// System call support for plan9 on arm

//func Syscall(trap, a1, a2, a3 uintptr) (r1, r2 uintptr, err ErrorString)
TEXT	·Syscall(SB),NOSPLIT,$144-32
	NO_LOCAL_POINTERS
	BL		runtime·entersyscall(SB)
	MOVW	$a1+4(FP), R0	// move syscall args
	MOVW	$sysargs-144(SP), R1
	MOVM.IA	(R0), [R2-R4]
	MOVM.IA	[R2-R4], (R1)
	MOVW	trap+0(FP), R0	// syscall num
	SWI		$0
	MOVW	$0, R2
	MOVW	$r1+16(FP), R3
	MOVM.IA	[R0,R2], (R3)
	CMP		$-1, R0
	B.EQ	syscallerr
	BL		runtime·exitsyscall(SB)
	MOVW	$·emptystring+0(SB), R2
	B		syscallok
syscallerr:
	MOVW	$errbuf-128(SP), R2
	MOVW	$128, R3
	MOVM.IA	[R2,R3], (R1)
	MOVW	$SYS_ERRSTR, R0
	SWI		$0
	BL		runtime·exitsyscall(SB)
	BL		runtime·gostring(SB)
	MOVW	$str-140(SP), R2
syscallok:
	MOVW	$err+24(FP), R1
	MOVM.IA	(R2), [R3-R4]
	MOVM.IA	[R3-R4], (R1)
	RET


//func Syscall6(trap, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err ErrorString)
// Actually Syscall5 but the rest of the code expects it to be named Syscall6.
TEXT	·Syscall6(SB),NOSPLIT,$144-44
	NO_LOCAL_POINTERS
	BL		runtime·entersyscall(SB)
	MOVW	$a1+4(FP), R0	// move syscall args
	MOVW	$sysargs-144(SP), R1
	MOVM.IA	(R0), [R2-R6]
	MOVM.IA	[R2-R6], (R1)
	MOVW	trap+0(FP), R0	// syscall num
	SWI		$0
	MOVW	$0, R2
	MOVW	$r1+28(FP), R3
	MOVM.IA.W	[R0,R2], (R3)
	CMP		$-1, R0
	B.EQ	syscall6err
	BL		runtime·exitsyscall(SB)
	MOVW	$·emptystring+0(SB), R2
	B		syscall6ok
syscall6err:
	MOVW	$errbuf-128(SP), R2
	MOVW	$128, R3
	MOVM.IA	[R2,R3], (R1)
	MOVW	$SYS_ERRSTR, R0
	SWI		$0
	BL		runtime·exitsyscall(SB)
	BL		runtime·gostring(SB)
	MOVW	$str-140(SP), R2
syscall6ok:
	MOVW	$err+36(FP), R1
	MOVM.IA	(R2), [R3-R4]
	MOVM.IA	[R3-R4], (R1)
	RET

//func RawSyscall(trap, a1, a2, a3 uintptr) (r1, r2, err uintptr)
TEXT ·RawSyscall(SB),NOSPLIT,$12-28
	MOVW	$a1+4(FP), R0	// move syscall args
	MOVW	$sysargs-12(SP), R1
	MOVM.IA	(R0), [R2-R4]
	MOVM.IA	[R2-R4], (R1)
	MOVW	trap+0(FP), R0	// syscall num
	SWI		$0
	MOVW	R0, r1+16(FP)
	MOVW	R0, r2+20(FP)
	MOVW	R0, err+24(FP)
	RET

//func RawSyscall6(trap, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, err uintptr)
// Actually RawSyscall5 but the rest of the code expects it to be named RawSyscall6.
TEXT	·RawSyscall6(SB),NOSPLIT,$20-40
	MOVW	$a1+4(FP), R0	// move syscall args
	MOVW	$sysargs-20(SP), R1
	MOVM.IA	(R0), [R2-R6]
	MOVM.IA	[R2-R6], (R1)
	MOVW	trap+0(FP), R0	// syscall num
	SWI		$0
	MOVW	R0, r1+28(FP)
	MOVW	R0, r2+32(FP)
	MOVW	R0, err+36(FP)
	RET

//func seek(placeholder uintptr, fd int, offset int64, whence int) (newoffset int64, err string)
TEXT ·seek(SB),NOSPLIT,$20-36
	NO_LOCAL_POINTERS
	MOVW	$newoffset_lo+20(FP), R6
	MOVW	R6, sysargs-20(SP)	// dest for return value
	MOVW	$fd+4(FP), R0		// move syscall args
	MOVW	$sysarg1-16(SP), R1
	MOVM.IA	(R0), [R2-R5]
	MOVM.IA	[R2-R5], (R1)
	MOVW	$SYS_SEEK, R0		// syscall num
	SWI		$0
	CMP		$-1, R0
	B.EQ	seekerr
	MOVW	$·emptystring+0(SB), R2
	B		seekok
seekerr:
	MOVW	R0, 0(R6)
	MOVW	R0, 4(R6)
	BL		·errstr(SB)
	MOVW	$ret-20(SP), R2
seekok:
	MOVW	$err+28(FP), R1
	MOVM.IA	(R2), [R3-R4]
	MOVM.IA	[R3-R4], (R1)
	RET
