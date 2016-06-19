// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ppc64 ppc64le
// +build !gccgo

#include "textflag.h"

TEXT ·RewindAndSetgid(SB),NOSPLIT|NOFRAME,$0-0
	// Rewind stack pointer so anything that happens on the stack
	// will clobber the test pattern created by the caller
	ADD	$(1024 * 8), R1

	// Ask signaller to setgid
	MOVW	$1, R3
	SYNC
	MOVW	R3, ·Baton(SB)

	// Wait for setgid completion
loop:
	SYNC
	MOVW	·Baton(SB), R3
	CMP	R3, $0
	// Hint that we're in a spin loop
	OR	R1, R1, R1
	BNE	loop
	ISYNC

	// Restore stack
	SUB	$(1024 * 8), R1
	RET
