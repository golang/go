// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build mips mipsle
// +build !gccgo

#include "textflag.h"

TEXT ·RewindAndSetgid(SB),NOSPLIT,$-4-0
	// Rewind stack pointer so anything that happens on the stack
	// will clobber the test pattern created by the caller
	ADDU	$(1024*8), R29

	// Ask signaller to setgid
	MOVW	$1, R1
	SYNC
	MOVW	R1, ·Baton(SB)
	SYNC

	// Wait for setgid completion
loop:
	SYNC
	MOVW	·Baton(SB), R1
	OR	R2, R2, R2	// hint that we're in a spin loop
	BNE	R1, loop
	SYNC

	// Restore stack
	ADDU	$(-1024*8), R29
	RET
