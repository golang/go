// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"
#include "funcdata.h"
#include "../runtime/syscall_nacl.h"

//
// System call support for ARM, Native Client
//

#define NACL_SYSCALL(code) \
	MOVW $(0x10000 + ((code)<<5)), R8; BL (R8)

#define NACL_SYSJMP(code) \
	MOVW $(0x10000 + ((code)<<5)), R8; B (R8)

TEXT ·Syscall(SB),NOSPLIT,$0-28
	BL	runtime·entersyscall(SB)
	MOVW	trap+0(FP), R8
	MOVW	a1+4(FP), R0
	MOVW	a2+8(FP), R1
	MOVW	a3+12(FP), R2
	// more args would use R3, and then stack.
	MOVW	$0x10000, R7
	ADD	R8<<5, R7
	BL	(R7)
	CMP	$0, R0
	BGE	ok
	MOVW	$-1, R1
	MOVW	R1, r1+16(FP)
	MOVW	R1, r2+20(FP)
	RSB	$0, R0
	MOVW	R0, err+24(FP)
	BL	runtime·exitsyscall(SB)
	RET
ok:
	MOVW	R0, r1+16(FP)
	MOVW	R1, r2+20(FP)
	MOVW	$0, R2
	MOVW	R2, err+24(FP)
	BL	runtime·exitsyscall(SB)
	RET	
