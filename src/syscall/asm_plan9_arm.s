// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"
#include "funcdata.h"

#define SYS_SEEK 39	/* from zsysnum_plan9.go */

// System call support for plan9 on arm

TEXT	sysresult<>(SB),NOSPLIT,$12
	MOVW	$runtime·emptystring+0(SB), R2
	CMP		$-1, R0
	B.NE	ok
	MOVW	R1, save-4(SP)
	BL		runtime·errstr(SB)
	MOVW	save-4(SP), R1
	MOVW	$err-12(SP), R2
ok:
	MOVM.IA	(R2), [R3-R4]
	MOVM.IA	[R3-R4], (R1)
	RET
	
//func Syscall(trap, a1, a2, a3 uintptr) (r1, r2 uintptr, err ErrorString)
TEXT	·Syscall(SB),NOSPLIT,$0-32
	BL		runtime·entersyscall(SB)
	MOVW	trap+0(FP), R0	// syscall num
	MOVM.IA.W	(R13),[R1-R2]	// pop LR and caller's LR
	SWI		$0
	MOVM.DB.W	[R1-R2],(R13)	// push LR and caller's LR
	MOVW	$0, R2
	MOVW	$r1+16(FP), R1
	MOVM.IA.W	[R0,R2], (R1)
	BL		sysresult<>(SB)
	BL		runtime·exitsyscall(SB)
	RET

//func Syscall6(trap, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err ErrorString)
// Actually Syscall5 but the rest of the code expects it to be named Syscall6.
TEXT	·Syscall6(SB),NOSPLIT,$0-44
	BL		runtime·entersyscall(SB)
	MOVW	trap+0(FP), R0	// syscall num
	MOVM.IA.W	(R13),[R1-R2]	// pop LR and caller's LR
	SWI		$0
	MOVM.DB.W	[R1-R2],(R13)	// push LR and caller's LR
	MOVW	$0, R1
	MOVW	$r1+28(FP), R1
	MOVM.IA.W	[R0,R2], (R1)
	BL		sysresult<>(SB)
	BL		runtime·exitsyscall(SB)
	RET

//func RawSyscall(trap, a1, a2, a3 uintptr) (r1, r2, err uintptr)
TEXT ·RawSyscall(SB),NOSPLIT,$0-28
	MOVW	trap+0(FP), R0	// syscall num
	MOVM.IA.W	(R13),[R1]		// pop caller's LR
	SWI		$0
	MOVM.DB.W	[R1],(R13)		// push caller's LR
	MOVW	R0, r1+16(FP)
	MOVW	R0, r2+20(FP)
	MOVW	R0, err+24(FP)
	RET

//func RawSyscall6(trap, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, err uintptr)
// Actually RawSyscall5 but the rest of the code expects it to be named RawSyscall6.
TEXT	·RawSyscall6(SB),NOSPLIT,$0-40
	MOVW	trap+0(FP), R0	// syscall num
	MOVM.IA.W	(R13),[R1]		// pop caller's LR
	SWI		$0
	MOVM.DB.W	[R1],(R13)		// push caller's LR
	MOVW	R0, r1+28(FP)
	MOVW	R0, r2+32(FP)
	MOVW	R0, err+36(FP)
	RET

//func seek(placeholder uintptr, fd int, offset int64, whence int) (newoffset int64, err string)
TEXT ·seek(SB),NOSPLIT,$0-36
	MOVW	$newoffset_lo+20(FP), R5
	MOVW	R5, placeholder+0(FP)	//placeholder = dest for return value
	MOVW	$SYS_SEEK, R0		// syscall num
	MOVM.IA.W	(R13),[R1]		// pop LR
	SWI		$0
	MOVM.DB.W	[R1],(R13)		// push LR
	CMP		$-1, R0
	MOVW.EQ	R0, 0(R5)
	MOVW.EQ	R0, 4(R5)
	MOVW	$err+28(FP), R1
	BL		sysresult<>(SB)
	RET

//func exit(code int)
// Import runtime·exit for cleanly exiting.
TEXT ·exit(SB),NOSPLIT,$4-4
	NO_LOCAL_POINTERS
	MOVW	code+0(FP), R0
	MOVW	R0, e-4(SP)
	BL		runtime·exit(SB)
	RET
