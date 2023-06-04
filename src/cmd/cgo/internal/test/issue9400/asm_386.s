// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build gc

#include "textflag.h"

TEXT ·RewindAndSetgid(SB),NOSPLIT,$0-0
	MOVL	$·Baton(SB), BX
	// Rewind stack pointer so anything that happens on the stack
	// will clobber the test pattern created by the caller
	ADDL	$(1024 * 8), SP

	// Ask signaller to setgid
	MOVL	$1, (BX)

	// Wait for setgid completion
loop:
	PAUSE
	MOVL	(BX), AX
	CMPL	AX, $0
	JNE	loop

	// Restore stack
	SUBL	$(1024 * 8), SP
	RET
