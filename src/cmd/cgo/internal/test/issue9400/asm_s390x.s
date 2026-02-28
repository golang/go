// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build gc

#include "textflag.h"

TEXT ·RewindAndSetgid(SB),NOSPLIT,$0-0
	// Rewind stack pointer so anything that happens on the stack
	// will clobber the test pattern created by the caller
	ADD	$(1024 * 8), R15

	// Ask signaller to setgid
	MOVD	$·Baton(SB), R5
	MOVW	$1, 0(R5)

	// Wait for setgid completion
loop:
	SYNC
	MOVW	·Baton(SB), R3
	CMPBNE	R3, $0, loop

	// Restore stack
	SUB	$(1024 * 8), R15
	RET
