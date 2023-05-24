// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (amd64 || amd64p32) && gc

#include "textflag.h"

TEXT ·RewindAndSetgid(SB),NOSPLIT,$0-0
	// Rewind stack pointer so anything that happens on the stack
	// will clobber the test pattern created by the caller
	ADDQ	$(1024 * 8), SP

	// Ask signaller to setgid
	MOVL	$1, ·Baton(SB)

	// Wait for setgid completion
loop:
	PAUSE
	MOVL	·Baton(SB), AX
	CMPL	AX, $0
	JNE	loop

	// Restore stack
	SUBQ	$(1024 * 8), SP
	RET
