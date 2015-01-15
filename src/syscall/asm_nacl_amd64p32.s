// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"
#include "funcdata.h"
#include "../runtime/syscall_nacl.h"

//
// System call support for amd64, Native Client
//

#define NACL_SYSCALL(code) \
	MOVL $(0x10000 + ((code)<<5)), AX; CALL AX

#define NACL_SYSJMP(code) \
	MOVL $(0x10000 + ((code)<<5)), AX; JMP AX

TEXT ·Syscall(SB),NOSPLIT,$0-28
	CALL	runtime·entersyscall(SB)
	MOVL	trap+0(FP), AX
	MOVL	a1+4(FP), DI
	MOVL	a2+8(FP), SI
	MOVL	a3+12(FP), DX
	// more args would use CX, R8, R9
	SHLL	$5, AX
	ADDL	$0x10000, AX
	CALL	AX
	CMPL	AX, $0
	JGE	ok
	MOVL	$-1, r1+16(FP)
	MOVL	$-1, r2+20(FP)
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
