// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

//
// System calls for aix/ppc64 are implemented in ../runtime/syscall_aix.go
//

TEXT ·syscall6(SB),NOSPLIT,$0
	JMP	runtime·syscall_syscall6(SB)

TEXT ·rawSyscall6(SB),NOSPLIT,$0
	JMP	runtime·syscall_rawSyscall6(SB)

TEXT ·RawSyscall(SB),NOSPLIT,$0
	JMP	runtime·syscall_RawSyscall(SB)

TEXT ·Syscall(SB),NOSPLIT,$0
	JMP	runtime·syscall_Syscall(SB)
