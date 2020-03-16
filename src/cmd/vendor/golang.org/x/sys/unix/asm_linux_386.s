// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !gccgo

#include "textflag.h"

//
// System calls for 386, Linux
//

// See ../runtime/sys_linux_386.s for the reason why we always use int 0x80
// instead of the glibc-specific "CALL 0x10(GS)".
#define INVOKE_SYSCALL	INT	$0x80

// Just jump to package syscall's implementation for all these functions.
// The runtime may know about them.

TEXT ·Syscall(SB),NOSPLIT,$0-28
	JMP	syscall·Syscall(SB)

TEXT ·Syscall6(SB),NOSPLIT,$0-40
	JMP	syscall·Syscall6(SB)

TEXT ·SyscallNoError(SB),NOSPLIT,$0-24
	CALL	runtime·entersyscall(SB)
	MOVL	trap+0(FP), AX  // syscall entry
	MOVL	a1+4(FP), BX
	MOVL	a2+8(FP), CX
	MOVL	a3+12(FP), DX
	MOVL	$0, SI
	MOVL	$0, DI
	INVOKE_SYSCALL
	MOVL	AX, r1+16(FP)
	MOVL	DX, r2+20(FP)
	CALL	runtime·exitsyscall(SB)
	RET

TEXT ·RawSyscall(SB),NOSPLIT,$0-28
	JMP	syscall·RawSyscall(SB)

TEXT ·RawSyscall6(SB),NOSPLIT,$0-40
	JMP	syscall·RawSyscall6(SB)

TEXT ·RawSyscallNoError(SB),NOSPLIT,$0-24
	MOVL	trap+0(FP), AX  // syscall entry
	MOVL	a1+4(FP), BX
	MOVL	a2+8(FP), CX
	MOVL	a3+12(FP), DX
	MOVL	$0, SI
	MOVL	$0, DI
	INVOKE_SYSCALL
	MOVL	AX, r1+16(FP)
	MOVL	DX, r2+20(FP)
	RET

TEXT ·socketcall(SB),NOSPLIT,$0-36
	JMP	syscall·socketcall(SB)

TEXT ·rawsocketcall(SB),NOSPLIT,$0-36
	JMP	syscall·rawsocketcall(SB)

TEXT ·seek(SB),NOSPLIT,$0-28
	JMP	syscall·seek(SB)
