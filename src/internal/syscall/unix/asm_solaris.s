// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// System calls for Solaris are implemented in runtime/syscall_solaris.go

TEXT ·syscall6(SB),NOSPLIT,$0-88
	JMP	syscall·sysvicall6(SB)

TEXT ·rawSyscall6(SB),NOSPLIT,$0-88
	JMP	syscall·rawSysvicall6(SB)
