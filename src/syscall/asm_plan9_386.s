// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"
#include "funcdata.h"

//
// System call support for 386, Plan 9
//

//func Syscall(trap, a1, a2, a3 uintptr) (r1, r2 uintptr, err string)
//func Syscall6(trap, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err string)
//func RawSyscall(trap, a1, a2, a3 uintptr) (r1, r2, err uintptr)
//func RawSyscall6(trap, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, err uintptr)

#define SYS_ERRSTR 41	/* from zsysnum_plan9.go */

// Trap # in AX, args on stack above caller pc.
TEXT	·Syscall(SB),NOSPLIT,$148-32
	NO_LOCAL_POINTERS
	CALL	runtime·entersyscall(SB)
	MOVL	trap+0(FP), AX	// syscall entry
	// copy args down
	LEAL	a1+4(FP), SI
	LEAL	sysargs-144(SP), DI
	CLD
	MOVSL
	MOVSL
	MOVSL
	INT	$64
	MOVL	AX, r1+16(FP)
	MOVL	$0, r2+20(FP)
	CMPL	AX, $-1
	JNE	ok3

	LEAL	errbuf-128(SP), AX
	MOVL	AX, sysargs-144(SP)
	MOVL	$128, sysargs1-140(SP)
	MOVL	$SYS_ERRSTR, AX
	INT	$64
	CALL	runtime·exitsyscall(SB)
	MOVL	sysargs-144(SP), AX
	MOVL	AX, errbuf-148(SP)
	CALL	runtime·gostring(SB)
	LEAL	str-144(SP), SI
	JMP	copyresult3

ok3:
	CALL	runtime·exitsyscall(SB)
	LEAL	·emptystring(SB), SI

copyresult3:
	LEAL	err+24(FP), DI

	CLD
	MOVSL
	MOVSL

	RET

TEXT	·Syscall6(SB),NOSPLIT,$148-44
	NO_LOCAL_POINTERS
	CALL	runtime·entersyscall(SB)
	MOVL	trap+0(FP), AX	// syscall entry
	// copy args down
	LEAL	a1+4(FP), SI
	LEAL	sysargs-144(SP), DI
	CLD
	MOVSL
	MOVSL
	MOVSL
	MOVSL
	MOVSL
	MOVSL
	INT	$64
	MOVL	AX, r1+28(FP)
	MOVL	$0, r2+32(FP)
	CMPL	AX, $-1
	JNE	ok4

	LEAL	errbuf-128(SP), AX
	MOVL	AX, sysargs-144(SP)
	MOVL	$128, sysargs1-140(SP)
	MOVL	$SYS_ERRSTR, AX
	INT	$64
	CALL	runtime·exitsyscall(SB)
	MOVL	sysargs-144(SP), AX
	MOVL	AX, errbuf-148(SP)
	CALL	runtime·gostring(SB)
	LEAL	str-144(SP), SI
	JMP	copyresult4

ok4:
	CALL	runtime·exitsyscall(SB)
	LEAL	·emptystring(SB), SI

copyresult4:
	LEAL	err+36(FP), DI

	CLD
	MOVSL
	MOVSL

	RET

TEXT ·RawSyscall(SB),NOSPLIT,$0-28
	MOVL	trap+0(FP), AX	// syscall entry
	// slide args down on top of system call number
	LEAL	a1+4(FP), SI
	LEAL	trap+0(FP), DI
	CLD
	MOVSL
	MOVSL
	MOVSL
	INT	$64
	MOVL	AX, r1+16(FP)
	MOVL	AX, r2+20(FP)
	MOVL	AX, err+24(FP)
	RET

TEXT	·RawSyscall6(SB),NOSPLIT,$0-40
	MOVL	trap+0(FP), AX	// syscall entry
	// slide args down on top of system call number
	LEAL	a1+4(FP), SI
	LEAL	trap+0(FP), DI
	CLD
	MOVSL
	MOVSL
	MOVSL
	MOVSL
	MOVSL
	MOVSL
	INT	$64
	MOVL	AX, r1+28(FP)
	MOVL	AX, r2+32(FP)
	MOVL	AX, err+36(FP)
	RET

#define SYS_SEEK 39	/* from zsysnum_plan9.go */

//func seek(placeholder uintptr, fd int, offset int64, whence int) (newoffset int64, err string)
TEXT ·seek(SB),NOSPLIT,$24-36
	NO_LOCAL_POINTERS
	LEAL	newoffset+20(FP), AX
	MOVL	AX, placeholder+0(FP)

	// copy args down
	LEAL	placeholder+0(FP), SI
	LEAL	sysargs-20(SP), DI
	CLD
	MOVSL
	MOVSL
	MOVSL
	MOVSL
	MOVSL
	MOVL	$SYS_SEEK, AX	// syscall entry
	INT	$64

	CMPL	AX, $-1
	JNE	ok6
	MOVL	AX, newoffset_lo+20(FP)
	MOVL	AX, newoffset_hi+24(FP)

	CALL	syscall·errstr(SB)
	MOVL	SP, SI
	JMP	copyresult6

ok6:
	LEAL	·emptystring(SB), SI

copyresult6:
	LEAL	err+28(FP), DI

	CLD
	MOVSL
	MOVSL
	RET
