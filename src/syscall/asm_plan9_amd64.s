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

#define SYS_ERRSTR 41	/* from zsysnum_plan9.go */

TEXT	·Syscall(SB),NOSPLIT,$168-64
	NO_LOCAL_POINTERS
	CALL	runtime·entersyscall<ABIInternal>(SB)
	MOVQ	trap+0(FP), BP	// syscall entry
	// copy args down
	LEAQ	a1+8(FP), SI
	LEAQ	sysargs-160(SP), DI
	CLD
	MOVSQ
	MOVSQ
	MOVSQ
	SYSCALL
	MOVQ	AX, r1+32(FP)
	MOVQ	$0, r2+40(FP)
	CMPL	AX, $-1
	JNE	ok3

	LEAQ	errbuf-128(SP), AX
	MOVQ	AX, sysargs-160(SP)
	MOVQ	$128, sysargs1-152(SP)
	MOVQ	$SYS_ERRSTR, BP
	SYSCALL
	CALL	runtime·exitsyscall(SB) // call via ABI wrapper, ensuring ABIInternal fixed registers are set
	MOVQ	sysargs-160(SP), AX
	MOVQ	AX, errbuf-168(SP)
	CALL	runtime·gostring(SB)
	LEAQ	str-160(SP), SI
	JMP	copyresult3

ok3:
	CALL	runtime·exitsyscall(SB) // call via ABI wrapper, ensuring ABIInternal fixed registers are set
	LEAQ	·emptystring(SB), SI

copyresult3:
	LEAQ	err+48(FP), DI

	CLD
	MOVSQ
	MOVSQ

	RET

TEXT	·Syscall6(SB),NOSPLIT,$168-88
	NO_LOCAL_POINTERS
	CALL	runtime·entersyscall<ABIInternal>(SB)
	MOVQ	trap+0(FP), BP	// syscall entry
	// copy args down
	LEAQ	a1+8(FP), SI
	LEAQ	sysargs-160(SP), DI
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

	LEAQ	errbuf-128(SP), AX
	MOVQ	AX, sysargs-160(SP)
	MOVQ	$128, sysargs1-152(SP)
	MOVQ	$SYS_ERRSTR, BP
	SYSCALL
	CALL	runtime·exitsyscall(SB) // call via ABI wrapper, ensuring ABIInternal fixed registers are set
	MOVQ	sysargs-160(SP), AX
	MOVQ	AX, errbuf-168(SP)
	CALL	runtime·gostring(SB)
	LEAQ	str-160(SP), SI
	JMP	copyresult4

ok4:
	CALL	runtime·exitsyscall(SB) // call via ABI wrapper, ensuring ABIInternal fixed registers are set
	LEAQ	·emptystring(SB), SI

copyresult4:
	LEAQ	err+72(FP), DI

	CLD
	MOVSQ
	MOVSQ

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

#define SYS_SEEK 39	/* from zsysnum_plan9.go */

//func seek(placeholder uintptr, fd int, offset int64, whence int) (newoffset int64, err string)
TEXT ·seek(SB),NOSPLIT,$48-56
	NO_LOCAL_POINTERS
	LEAQ	newoffset+32(FP), AX
	MOVQ	AX, placeholder+0(FP)

	// copy args down
	LEAQ	placeholder+0(FP), SI
	LEAQ	sysargs-40(SP), DI
	CLD
	MOVSQ
	MOVSQ
	MOVSQ
	MOVSQ
	MOVSQ
	MOVQ	$SYS_SEEK, BP	// syscall entry
	SYSCALL

	CMPL	AX, $-1
	JNE	ok6
	MOVQ	AX, newoffset+32(FP)

	CALL	syscall·errstr(SB)
	MOVQ	SP, SI
	JMP	copyresult6

ok6:
	LEAQ	·emptystring(SB), SI

copyresult6:
	LEAQ	err+40(FP), DI

	CLD
	MOVSQ
	MOVSQ
	RET
