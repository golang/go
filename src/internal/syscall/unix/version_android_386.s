// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

// These trampolines help convert from Go calling convention to C calling convention.
// They should be called with asmcgocall - note that while asmcgocall does
// stack alignment, creation of a frame undoes it again.
// A pointer to the arguments is passed on the stack.
// A single int32 result is returned in AX.
// (For more results, make an args/results structure.)
TEXT ·__system_property_get_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$8, SP
	MOVL	16(SP), DX		// pointer to args
	MOVL	0(DX), AX
	MOVL	4(DX), DX
	MOVL	AX, 0(SP)		// arg 1 - name
	MOVL	DX, 4(SP)		// arg 2 - value
	MOVL	$_GLOBAL_OFFSET_TABLE_(SB), BX
	CALL	libc___system_property_get(SB)
	MOVL	BP, SP
	POPL	BP
	RET
