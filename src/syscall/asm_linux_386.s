// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"
#include "funcdata.h"

//
// System calls for 386, Linux
//

// func Syscall(trap uintptr, a1, a2, a3 uintptr) (r1, r2, err uintptr);
// Trap # in AX, args in BX CX DX SI DI, return in AX

// Most linux systems use glibc's dynamic linker, which puts the
// __kernel_vsyscall vdso helper at 0x10(GS) for easy access from position
// independent code and setldt in runtime does the same in the statically
// linked case. Android, however, uses bionic's dynamic linker, which does not
// save the helper anywhere, and so the only way to invoke a syscall from
// position independent code is boring old int $0x80 (which is also what
// bionic's syscall wrappers use).
#ifdef GOOS_android
#define INVOKE_SYSCALL	INT	$0x80
#else
#define INVOKE_SYSCALL	CALL	0x10(GS)
#endif

TEXT	·Syscall(SB),NOSPLIT,$0-28
	CALL	runtime·entersyscall(SB)
	MOVL	trap+0(FP), AX	// syscall entry
	MOVL	a1+4(FP), BX
	MOVL	a2+8(FP), CX
	MOVL	a3+12(FP), DX
	MOVL	$0, SI
	MOVL	$0,  DI
	INVOKE_SYSCALL
	CMPL	AX, $0xfffff001
	JLS	ok
	MOVL	$-1, r1+16(FP)
	MOVL	$0, r2+20(FP)
	NEGL	AX
	MOVL	AX, err+24(FP)
	CALL	runtime·exitsyscall(SB)
	RET
ok:
	MOVL	AX, r1+16(FP)
	MOVL	DX, r2+20(FP)
	MOVL	$0, err+24(FP)
	CALL	runtime·exitsyscall(SB)
	RET

// func Syscall6(trap uintptr, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, err uintptr);
TEXT	·Syscall6(SB),NOSPLIT,$0-40
	CALL	runtime·entersyscall(SB)
	MOVL	trap+0(FP), AX	// syscall entry
	MOVL	a1+4(FP), BX
	MOVL	a2+8(FP), CX
	MOVL	a3+12(FP), DX
	MOVL	a4+16(FP), SI
	MOVL	a5+20(FP), DI
	MOVL	a6+24(FP), BP
	INVOKE_SYSCALL
	CMPL	AX, $0xfffff001
	JLS	ok6
	MOVL	$-1, r1+28(FP)
	MOVL	$0, r2+32(FP)
	NEGL	AX
	MOVL	AX, err+36(FP)
	CALL	runtime·exitsyscall(SB)
	RET
ok6:
	MOVL	AX, r1+28(FP)
	MOVL	DX, r2+32(FP)
	MOVL	$0, err+36(FP)
	CALL	runtime·exitsyscall(SB)
	RET

// func RawSyscall(trap uintptr, a1, a2, a3 uintptr) (r1, r2, err uintptr);
TEXT ·RawSyscall(SB),NOSPLIT,$0-28
	MOVL	trap+0(FP), AX	// syscall entry
	MOVL	a1+4(FP), BX
	MOVL	a2+8(FP), CX
	MOVL	a3+12(FP), DX
	MOVL	$0, SI
	MOVL	$0,  DI
	INVOKE_SYSCALL
	CMPL	AX, $0xfffff001
	JLS	ok1
	MOVL	$-1, r1+16(FP)
	MOVL	$0, r2+20(FP)
	NEGL	AX
	MOVL	AX, err+24(FP)
	RET
ok1:
	MOVL	AX, r1+16(FP)
	MOVL	DX, r2+20(FP)
	MOVL	$0, err+24(FP)
	RET

// func RawSyscall6(trap uintptr, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, err uintptr);
TEXT	·RawSyscall6(SB),NOSPLIT,$0-40
	MOVL	trap+0(FP), AX	// syscall entry
	MOVL	a1+4(FP), BX
	MOVL	a2+8(FP), CX
	MOVL	a3+12(FP), DX
	MOVL	a4+16(FP), SI
	MOVL	a5+20(FP), DI
	MOVL	a6+24(FP), BP
	INVOKE_SYSCALL
	CMPL	AX, $0xfffff001
	JLS	ok2
	MOVL	$-1, r1+28(FP)
	MOVL	$0, r2+32(FP)
	NEGL	AX
	MOVL	AX, err+36(FP)
	RET
ok2:
	MOVL	AX, r1+28(FP)
	MOVL	DX, r2+32(FP)
	MOVL	$0, err+36(FP)
	RET

#define SYS_SOCKETCALL 102	/* from zsysnum_linux_386.go */

// func socketcall(call int, a0, a1, a2, a3, a4, a5 uintptr) (n int, err int)
// Kernel interface gets call sub-number and pointer to a0.
TEXT ·socketcall(SB),NOSPLIT,$0-36
	CALL	runtime·entersyscall(SB)
	MOVL	$SYS_SOCKETCALL, AX	// syscall entry
	MOVL	call+0(FP), BX	// socket call number
	LEAL		a0+4(FP), CX	// pointer to call arguments
	MOVL	$0, DX
	MOVL	$0, SI
	MOVL	$0,  DI
	INVOKE_SYSCALL
	CMPL	AX, $0xfffff001
	JLS	oksock
	MOVL	$-1, n+28(FP)
	NEGL	AX
	MOVL	AX, err+32(FP)
	CALL	runtime·exitsyscall(SB)
	RET
oksock:
	MOVL	AX, n+28(FP)
	MOVL	$0, err+32(FP)
	CALL	runtime·exitsyscall(SB)
	RET

// func rawsocketcall(call int, a0, a1, a2, a3, a4, a5 uintptr) (n int, err int)
// Kernel interface gets call sub-number and pointer to a0.
TEXT ·rawsocketcall(SB),NOSPLIT,$0-36
	MOVL	$SYS_SOCKETCALL, AX	// syscall entry
	MOVL	call+0(FP), BX	// socket call number
	LEAL		a0+4(FP), CX	// pointer to call arguments
	MOVL	$0, DX
	MOVL	$0, SI
	MOVL	$0,  DI
	INVOKE_SYSCALL
	CMPL	AX, $0xfffff001
	JLS	oksock1
	MOVL	$-1, n+28(FP)
	NEGL	AX
	MOVL	AX, err+32(FP)
	RET
oksock1:
	MOVL	AX, n+28(FP)
	MOVL	$0, err+32(FP)
	RET

#define SYS__LLSEEK 140	/* from zsysnum_linux_386.go */
// func Seek(fd int, offset int64, whence int) (newoffset int64, err int)
// Implemented in assembly to avoid allocation when
// taking the address of the return value newoffset.
// Underlying system call is
//	llseek(int fd, int offhi, int offlo, int64 *result, int whence)
TEXT ·seek(SB),NOSPLIT,$0-28
	CALL	runtime·entersyscall(SB)
	MOVL	$SYS__LLSEEK, AX	// syscall entry
	MOVL	fd+0(FP), BX
	MOVL	offset_hi+8(FP), CX
	MOVL	offset_lo+4(FP), DX
	LEAL	newoffset_lo+16(FP), SI	// result pointer
	MOVL	whence+12(FP),  DI
	INVOKE_SYSCALL
	CMPL	AX, $0xfffff001
	JLS	okseek
	MOVL	$-1, newoffset_lo+16(FP)
	MOVL	$-1, newoffset_hi+20(FP)
	NEGL	AX
	MOVL	AX, err+24(FP)
	CALL	runtime·exitsyscall(SB)
	RET
okseek:
	// system call filled in newoffset already
	MOVL	$0, err+24(FP)
	CALL	runtime·exitsyscall(SB)
	RET
