// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

TEXT _rt0_386_windows(SB),7,$12
	MOVL	12(SP), AX
	LEAL	16(SP), BX
	MOVL	AX, 4(SP)
	MOVL	BX, 8(SP)
	MOVL	$-1, 0(SP) // return PC for main
	JMP	main(SB)

TEXT main(SB),7,$0
	JMP	_rt0_386(SB)


DATA  runtime·iswindows(SB)/4, $1
GLOBL runtime·iswindows(SB), $4
