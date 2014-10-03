// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(rsc): Rewrite all nn(SP) references into name+(nn-8)(FP)
// so that go vet can check that they are correct.

#include "textflag.h"
#include "funcdata.h"

//
// System calls for 386, Linux
//

// func Syscall(trap uintptr, a1, a2, a3 uintptr) (r1, r2, err uintptr);
// Trap # in AX, args in BX CX DX SI DI, return in AX

TEXT	·Syscall(SB),NOSPLIT,$0-28
	CALL	runtime·entersyscall(SB)
	MOVL	4(SP), AX	// syscall entry
	MOVL	8(SP), BX
	MOVL	12(SP), CX
	MOVL	16(SP), DX
	MOVL	$0, SI
	MOVL	$0,  DI
	CALL	*runtime·_vdso(SB)
	CMPL	AX, $0xfffff001
	JLS	ok
	MOVL	$-1, 20(SP)	// r1
	MOVL	$0, 24(SP)	// r2
	NEGL	AX
	MOVL	AX, 28(SP)  // errno
	CALL	runtime·exitsyscall(SB)
	RET
ok:
	MOVL	AX, 20(SP)	// r1
	MOVL	DX, 24(SP)	// r2
	MOVL	$0, 28(SP)	// errno
	CALL	runtime·exitsyscall(SB)
	RET

// func Syscall6(trap uintptr, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, err uintptr);
TEXT	·Syscall6(SB),NOSPLIT,$0-40
	CALL	runtime·entersyscall(SB)
	MOVL	4(SP), AX	// syscall entry
	MOVL	8(SP), BX
	MOVL	12(SP), CX
	MOVL	16(SP), DX
	MOVL	20(SP), SI
	MOVL	24(SP), DI
	MOVL	28(SP), BP
	CALL	*runtime·_vdso(SB)
	CMPL	AX, $0xfffff001
	JLS	ok6
	MOVL	$-1, 32(SP)	// r1
	MOVL	$0, 36(SP)	// r2
	NEGL	AX
	MOVL	AX, 40(SP)  // errno
	CALL	runtime·exitsyscall(SB)
	RET
ok6:
	MOVL	AX, 32(SP)	// r1
	MOVL	DX, 36(SP)	// r2
	MOVL	$0, 40(SP)	// errno
	CALL	runtime·exitsyscall(SB)
	RET

// func RawSyscall(trap uintptr, a1, a2, a3 uintptr) (r1, r2, err uintptr);
TEXT ·RawSyscall(SB),NOSPLIT,$0-28
	MOVL	4(SP), AX	// syscall entry
	MOVL	8(SP), BX
	MOVL	12(SP), CX
	MOVL	16(SP), DX
	MOVL	$0, SI
	MOVL	$0,  DI
	CALL	*runtime·_vdso(SB)
	CMPL	AX, $0xfffff001
	JLS	ok1
	MOVL	$-1, 20(SP)	// r1
	MOVL	$0, 24(SP)	// r2
	NEGL	AX
	MOVL	AX, 28(SP)  // errno
	RET
ok1:
	MOVL	AX, 20(SP)	// r1
	MOVL	DX, 24(SP)	// r2
	MOVL	$0, 28(SP)	// errno
	RET

// func RawSyscall6(trap uintptr, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, err uintptr);
TEXT	·RawSyscall6(SB),NOSPLIT,$0-40
	MOVL	4(SP), AX	// syscall entry
	MOVL	8(SP), BX
	MOVL	12(SP), CX
	MOVL	16(SP), DX
	MOVL	20(SP), SI
	MOVL	24(SP), DI
	MOVL	28(SP), BP
	CALL	*runtime·_vdso(SB)
	CMPL	AX, $0xfffff001
	JLS	ok2
	MOVL	$-1, 32(SP)	// r1
	MOVL	$0, 36(SP)	// r2
	NEGL	AX
	MOVL	AX, 40(SP)  // errno
	RET
ok2:
	MOVL	AX, 32(SP)	// r1
	MOVL	DX, 36(SP)	// r2
	MOVL	$0, 40(SP)	// errno
	RET

#define SYS_SOCKETCALL 102	/* from zsysnum_linux_386.go */

// func socketcall(call int, a0, a1, a2, a3, a4, a5 uintptr) (n int, errno int)
// Kernel interface gets call sub-number and pointer to a0.
TEXT ·socketcall(SB),NOSPLIT,$0-36
	CALL	runtime·entersyscall(SB)
	MOVL	$SYS_SOCKETCALL, AX	// syscall entry
	MOVL	4(SP), BX	// socket call number
	LEAL		8(SP), CX	// pointer to call arguments
	MOVL	$0, DX
	MOVL	$0, SI
	MOVL	$0,  DI
	CALL	*runtime·_vdso(SB)
	CMPL	AX, $0xfffff001
	JLS	oksock
	MOVL	$-1, 32(SP)	// n
	NEGL	AX
	MOVL	AX, 36(SP)  // errno
	CALL	runtime·exitsyscall(SB)
	RET
oksock:
	MOVL	AX, 32(SP)	// n
	MOVL	$0, 36(SP)	// errno
	CALL	runtime·exitsyscall(SB)
	RET

// func rawsocketcall(call int, a0, a1, a2, a3, a4, a5 uintptr) (n int, errno int)
// Kernel interface gets call sub-number and pointer to a0.
TEXT ·rawsocketcall(SB),NOSPLIT,$0-36
	MOVL	$SYS_SOCKETCALL, AX	// syscall entry
	MOVL	4(SP), BX	// socket call number
	LEAL		8(SP), CX	// pointer to call arguments
	MOVL	$0, DX
	MOVL	$0, SI
	MOVL	$0,  DI
	CALL	*runtime·_vdso(SB)
	CMPL	AX, $0xfffff001
	JLS	oksock1
	MOVL	$-1, 32(SP)	// n
	NEGL	AX
	MOVL	AX, 36(SP)  // errno
	RET
oksock1:
	MOVL	AX, 32(SP)	// n
	MOVL	$0, 36(SP)	// errno
	RET

#define SYS__LLSEEK 140	/* from zsysnum_linux_386.go */
// func Seek(fd int, offset int64, whence int) (newoffset int64, errno int)
// Implemented in assembly to avoid allocation when
// taking the address of the return value newoffset.
// Underlying system call is
//	llseek(int fd, int offhi, int offlo, int64 *result, int whence)
TEXT ·seek(SB),NOSPLIT,$0-28
	CALL	runtime·entersyscall(SB)
	MOVL	$SYS__LLSEEK, AX	// syscall entry
	MOVL	4(SP), BX	// fd
	MOVL	12(SP), CX	// offset-high
	MOVL	8(SP), DX	// offset-low
	LEAL	20(SP), SI	// result pointer
	MOVL	16(SP),  DI	// whence
	CALL	*runtime·_vdso(SB)
	CMPL	AX, $0xfffff001
	JLS	okseek
	MOVL	$-1, 20(SP)	// newoffset low
	MOVL	$-1, 24(SP)	// newoffset high
	NEGL	AX
	MOVL	AX, 28(SP)  // errno
	CALL	runtime·exitsyscall(SB)
	RET
okseek:
	// system call filled in newoffset already
	MOVL	$0, 28(SP)	// errno
	CALL	runtime·exitsyscall(SB)
	RET
