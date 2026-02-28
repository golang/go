// Copyright 2020 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build riscv64
// +build gc

#include "textflag.h"

TEXT ·RewindAndSetgid(SB),NOSPLIT|NOFRAME,$0-0
	// Rewind stack pointer so anything that happens on the stack
	// will clobber the test pattern created by the caller
	ADD	$(1024*8), X2

	// Ask signaller to setgid
	MOV	$1, X5
	FENCE
	MOVW	X5, ·Baton(SB)
	FENCE

	// Wait for setgid completion
loop:
	FENCE
	MOVW	·Baton(SB), X5
	OR	X6, X6, X6	// hint that we're in a spin loop
	BNE	ZERO, X5, loop
	FENCE

	// Restore stack
	ADD	$(-1024*8), X2
	RET
