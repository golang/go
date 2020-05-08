// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !gccgo

#include "textflag.h"

TEXT ·RewindAndSetgid(SB),NOSPLIT|NOFRAME,$0-0
	// Save link register
	MOVD	R30, R9

	// Rewind stack pointer so anything that happens on the stack
	// will clobber the test pattern created by the caller
	ADD	$(1024 * 8), RSP

	// Ask signaller to setgid
	MOVD	$·Baton(SB), R0
	MOVD	$1, R1
storeloop:
	LDAXRW	(R0), R2
	STLXRW	R1, (R0), R3
	CBNZ	R3, storeloop

	// Wait for setgid completion
	MOVW	$0, R1
	MOVW	$0, R2
loop:
	LDAXRW	(R0), R3
	CMPW	R1, R3
	BNE	loop
	STLXRW	R2, (R0), R3
	CBNZ	R3, loop

	// Restore stack
	SUB	$(1024 * 8), RSP

	MOVD	R9, R30
	RET
