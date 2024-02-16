// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build gc

#include "textflag.h"

TEXT cas<>(SB),NOSPLIT,$0
	MOVW	$0xffff0fc0, R15 // R15 is PC

TEXT ·RewindAndSetgid(SB),NOSPLIT|NOFRAME,$0-0
	// Save link register
	MOVW	R14, R4

	// Rewind stack pointer so anything that happens on the stack
	// will clobber the test pattern created by the caller
	ADD	$(1024 * 8), R13

	// Ask signaller to setgid
	MOVW	$·Baton(SB), R2
storeloop:
	MOVW	0(R2), R0
	MOVW	$1, R1
	BL	cas<>(SB)
	BCC	storeloop

	// Wait for setgid completion
loop:
	MOVW	$0, R0
	MOVW	$0, R1
	BL	cas<>(SB)
	BCC	loop

	// Restore stack
	SUB	$(1024 * 8), R13

	MOVW	R4, R14
	RET
