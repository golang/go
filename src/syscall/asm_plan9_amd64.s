// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"
#include "funcdata.h"

//
// System call support for Plan 9
//

//func Syscall(trap, a1, a2, a3 uintptr) (r1, r2 uintptr, err string)
//func Syscall6(trap, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err string)
//func RawSyscall(trap, a1, a2, a3 uintptr) (r1, r2, err uintptr)
//func RawSyscall6(trap, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, err uintptr)

TEXT	·Syscall(SB),NOSPLIT,$0-64
	CALL	runtime·entersyscall(SB)
	MOVQ	trap+0(FP), BP	// syscall entry
	// slide args down on top of system call number
	LEAQ	a1+8(FP), SI
	LEAQ	trap+0(FP), DI
	CLD
	MOVSQ
	MOVSQ
	MOVSQ
	SYSCALL
	MOVQ	AX, r1+32(FP)
	MOVQ	$0, r2+40(FP)
	CMPL	AX, $-1
	JNE	ok3

	SUBQ	$16, SP
	CALL	runtime·errstr(SB)
	MOVQ	SP, SI
	ADDQ	$16, SP
	JMP	copyresult3
	
ok3:
	LEAQ	runtime·emptystring(SB), SI	
	
copyresult3:
	LEAQ	err+48(FP), DI

	CLD
	MOVSQ
	MOVSQ

	CALL	runtime·exitsyscall(SB)
	RET

TEXT	·Syscall6(SB),NOSPLIT,$0-88
	CALL	runtime·entersyscall(SB)
	MOVQ	trap+0(FP), BP	// syscall entry
	// slide args down on top of system call number
	LEAQ	a1+8(FP), SI
	LEAQ	trap+0(FP), DI
	CLD
	MOVSQ
	MOVSQ
	MOVSQ
	MOVSQ
	MOVSQ
	MOVSQ
	SYSCALL
	MOVQ	AX, r1+56(FP)
	MOVQ	$0, r2+64(FP)
	CMPL	AX, $-1
	JNE	ok4
	
	SUBQ	$16, SP
	CALL	runtime·errstr(SB)
	MOVQ	SP, SI
	ADDQ	$16, SP
	JMP	copyresult4
	
ok4:
	LEAQ	runtime·emptystring(SB), SI
	
copyresult4:
	LEAQ	err+72(FP), DI

	CLD
	MOVSQ
	MOVSQ

	CALL	runtime·exitsyscall(SB)
	RET

TEXT ·RawSyscall(SB),NOSPLIT,$0-56
	MOVQ	trap+0(FP), BP	// syscall entry
	// slide args down on top of system call number
	LEAQ	a1+8(FP), SI
	LEAQ	trap+0(FP), DI
	CLD
	MOVSQ
	MOVSQ
	MOVSQ
	SYSCALL
	MOVQ	AX, r1+32(FP)
	MOVQ	AX, r2+40(FP)
	MOVQ	AX, err+48(FP)
	RET

TEXT	·RawSyscall6(SB),NOSPLIT,$0-80
	MOVQ	trap+0(FP), BP	// syscall entry
	// slide args down on top of system call number
	LEAQ	a1+8(FP), SI
	LEAQ	trap+0(FP), DI
	CLD
	MOVSQ
	MOVSQ
	MOVSQ
	MOVSQ
	MOVSQ
	MOVSQ
	SYSCALL
	MOVQ	AX, r1+56(FP)
	MOVQ	AX, r2+64(FP)
	MOVQ	AX, err+72(FP)
	RET

#define SYS_SEEK 39	/* from zsysnum_plan9_amd64.go */

//func seek(placeholder uintptr, fd int, offset int64, whence int) (newoffset int64, err string)
TEXT ·seek(SB),NOSPLIT,$0-56
	LEAQ	newoffset+32(FP), AX
	MOVQ	AX, placeholder+0(FP)
	
	MOVQ	$SYS_SEEK, BP	// syscall entry
	SYSCALL
	
	CMPL	AX, $-1
	JNE	ok6
	MOVQ	$-1, newoffset+32(FP)
	
	SUBQ	$16, SP
	CALL	syscall·errstr(SB)
	MOVQ	SP, SI
	ADDQ	$16, SP	
	JMP	copyresult6
	
ok6:
	LEAQ	runtime·emptystring(SB), SI
	
copyresult6:
	LEAQ	err+40(FP), DI

	CLD
	MOVSQ
	MOVSQ
	RET

//func exit(code int)
// Import runtime·exit for cleanly exiting.
TEXT ·exit(SB),NOSPLIT,$8-8
	NO_LOCAL_POINTERS
	MOVQ	code+0(FP), AX
	MOVQ	AX, 0(SP)
	CALL	runtime·exit(SB)
	RET
