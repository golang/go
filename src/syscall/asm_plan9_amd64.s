// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(rsc): Rewrite all nn(SP) references into name+(nn-8)(FP)
// so that go vet can check that they are correct.

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
	MOVQ	8(SP), BP	// syscall entry
	// slide args down on top of system call number
	LEAQ	16(SP), SI
	LEAQ	8(SP), DI
	CLD
	MOVSQ
	MOVSQ
	MOVSQ
	SYSCALL
	MOVQ	AX, r1+40(SP)
	MOVQ	$0, r2+48(SP)
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
	LEAQ	err+56(SP), DI

	CLD
	MOVSQ
	MOVSQ

	CALL	runtime·exitsyscall(SB)
	RET

TEXT	·Syscall6(SB),NOSPLIT,$0-88
	CALL	runtime·entersyscall(SB)
	MOVQ	8(SP), BP	// syscall entry
	// slide args down on top of system call number
	LEAQ		16(SP), SI
	LEAQ		8(SP), DI
	CLD
	MOVSQ
	MOVSQ
	MOVSQ
	MOVSQ
	MOVSQ
	MOVSQ
	SYSCALL
	MOVQ	AX, r1+64(SP)
	MOVQ	$0, r2+72(SP)
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
	LEAQ	err+80(SP), DI

	CLD
	MOVSQ
	MOVSQ

	CALL	runtime·exitsyscall(SB)
	RET

TEXT ·RawSyscall(SB),NOSPLIT,$0-56
	MOVQ	8(SP), BP	// syscall entry
	// slide args down on top of system call number
	LEAQ		16(SP), SI
	LEAQ		8(SP), DI
	CLD
	MOVSQ
	MOVSQ
	MOVSQ
	SYSCALL
	MOVQ	AX, r1+40(SP)
	MOVQ	AX, r2+48(SP)
	MOVQ	AX, err+56(SP)
	RET

TEXT	·RawSyscall6(SB),NOSPLIT,$0-80
	MOVQ	8(SP), BP	// syscall entry
	// slide args down on top of system call number
	LEAQ		16(SP), SI
	LEAQ		8(SP), DI
	CLD
	MOVSQ
	MOVSQ
	MOVSQ
	MOVSQ
	MOVSQ
	MOVSQ
	SYSCALL
	MOVQ	AX, r1+64(SP)
	MOVQ	AX, r2+72(SP)
	MOVQ	AX, err+80(SP)		
	RET

#define SYS_SEEK 39	/* from zsysnum_plan9_amd64.go */

//func seek(placeholder uintptr, fd int, offset int64, whence int) (newoffset int64, err string)
TEXT ·seek(SB),NOSPLIT,$0-56
	LEAQ	newoffset+40(SP), AX
	MOVQ	AX, placeholder+8(SP)
	
	MOVQ	$SYS_SEEK, BP	// syscall entry
	SYSCALL
	
	CMPL	AX, $-1
	JNE	ok6
	MOVQ	$-1, newoffset+40(SP)
	
	SUBQ	$16, SP
	CALL	syscall·errstr(SB)
	MOVQ	SP, SI
	ADDQ	$16, SP	
	JMP	copyresult6
	
ok6:
	LEAQ	runtime·emptystring(SB), SI
	
copyresult6:
	LEAQ	err+48(SP), DI

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
