// Copyright 2016 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build mips64 mips64le
// +build gc

#include "textflag.h"

#define SYNC	WORD $0xf

TEXT ·RewindAndSetgid(SB),NOSPLIT|NOFRAME,$0-0
	// Rewind stack pointer so anything that happens on the stack
	// will clobber the test pattern created by the caller
	ADDV	$(1024*8), R29

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
	ADDV	$(-1024*8), R29
	RET
