// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "zasm_GOOS_GOARCH.h"
#include "textflag.h"

TEXT syscall·Syscall(SB),NOSPLIT,$0-0
	JMP	runtime·syscall_Syscall(SB)

TEXT syscall·Syscall6(SB),NOSPLIT,$0-0
	JMP	runtime·syscall_Syscall6(SB)

TEXT syscall·Syscall9(SB),NOSPLIT,$0-0
	JMP	runtime·syscall_Syscall9(SB)

TEXT syscall·Syscall12(SB),NOSPLIT,$0-0
	JMP	runtime·syscall_Syscall12(SB)

TEXT syscall·Syscall15(SB),NOSPLIT,$0-0
	JMP	runtime·syscall_Syscall15(SB)

TEXT syscall·loadlibrary(SB),NOSPLIT,$0-0
	JMP	runtime·syscall_loadlibrary(SB)

TEXT syscall·getprocaddress(SB),NOSPLIT,$0-0
	JMP	runtime·syscall_getprocaddress(SB)

TEXT syscall·compileCallback(SB),NOSPLIT,$0
	JMP	runtime·compileCallback(SB)
