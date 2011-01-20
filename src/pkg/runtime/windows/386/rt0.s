// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

TEXT _rt0_386_windows(SB),7,$0
	// Set up SEH frame for bootstrap m
	PUSHL	$runtime·sigtramp(SB)
	PUSHL	0(FS)
	MOVL	SP, 0(FS)

	JMP	_rt0_386(SB)

DATA  runtime·iswindows(SB)/4, $1
GLOBL runtime·iswindows(SB), $4
