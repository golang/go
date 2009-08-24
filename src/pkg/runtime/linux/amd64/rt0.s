// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Darwin and Linux use the same linkage to main

TEXT	_rt0_amd64_linux(SB),7,$-8
	MOVQ	_initffi(SB), AX
	TESTQ	AX, AX
	JZ	2(PC)
	CALL	AX

	MOVQ	$_rt0_amd64(SB), AX
	JMP	AX

GLOBL _initffi(SB), $8
