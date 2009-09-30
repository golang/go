// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Darwin and Linux use the same linkage to main

TEXT	_rt0_386_linux(SB),7,$0
	MOVL	initcgo(SB), AX
	TESTL	AX, AX
	JZ	2(PC)
	CALL	AX

	JMP	_rt0_386(SB)

GLOBL initcgo(SB), $4
