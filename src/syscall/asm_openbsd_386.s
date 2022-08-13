// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

//
// System call support for 386, OpenBSD
//

// Provide these function names via assembly so they are provided as ABI0,
// rather than ABIInternal.

// func Syscall(trap, a1, a2, a3 uintptr) (r1, r2 uintptr, err Errno)
TEXT	·Syscall(SB),NOSPLIT,$0-28
	JMP	·syscallInternal(SB)

// func Syscall6(trap, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err Errno)
TEXT	·Syscall6(SB),NOSPLIT,$0-40
	JMP	·syscall6Internal(SB)

// func RawSyscall(trap, a1, a2, a3 uintptr) (r1, r2 uintptr, err Errno)
TEXT	·RawSyscall(SB),NOSPLIT,$0-28
	JMP	·rawSyscallInternal(SB)

// func RawSyscall6(trap, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err Errno)
TEXT	·RawSyscall6(SB),NOSPLIT,$0-40
	JMP	·rawSyscall6Internal(SB)

// func Syscall9(trap, a1, a2, a3, a4, a5, a6, a7, a8, a9 uintptr) (r1, r2 uintptr, err Errno)
TEXT	·Syscall9(SB),NOSPLIT,$0-52
	JMP	·syscall9Internal(SB)
