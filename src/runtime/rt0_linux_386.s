// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT _rt0_386_linux(SB),NOSPLIT,$8
	MOVL	8(SP), AX
	LEAL	12(SP), BX
	MOVL	AX, 0(SP)
	MOVL	BX, 4(SP)
	CALL	runtime·linux_setup_vdso(SB)
	CALL	main(SB)
	INT	$3

TEXT main(SB),NOSPLIT,$0
	JMP	runtime·rt0_go(SB)

TEXT _fallback_vdso(SB),NOSPLIT,$0
	INT	$0x80
	RET

DATA	runtime·_vdso(SB)/4, $_fallback_vdso(SB)
GLOBL	runtime·_vdso(SB), NOPTR, $4

