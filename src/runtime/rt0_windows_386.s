// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT _rt0_386_windows(SB),NOSPLIT,$12
	MOVL	12(SP), AX
	LEAL	16(SP), BX
	MOVL	AX, 4(SP)
	MOVL	BX, 8(SP)
	MOVL	$-1, 0(SP) // return PC for main
	JMP	_main(SB)

TEXT _main(SB),NOSPLIT,$0
	JMP	runtime·rt0_go(SB)
