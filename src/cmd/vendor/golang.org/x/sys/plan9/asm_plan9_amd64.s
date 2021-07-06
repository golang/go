// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

//
// System call support for amd64, Plan 9
//

// Just jump to package syscall's implementation for all these functions.
// The runtime may know about them.

TEXT	·Syscall(SB),NOSPLIT,$0-64
	JMP	syscall·Syscall(SB)

TEXT	·Syscall6(SB),NOSPLIT,$0-88
	JMP	syscall·Syscall6(SB)

TEXT ·RawSyscall(SB),NOSPLIT,$0-56
	JMP	syscall·RawSyscall(SB)

TEXT	·RawSyscall6(SB),NOSPLIT,$0-80
	JMP	syscall·RawSyscall6(SB)

TEXT ·seek(SB),NOSPLIT,$0-56
	JMP	syscall·seek(SB)

TEXT ·exit(SB),NOSPLIT,$8-8
	JMP	syscall·exit(SB)
