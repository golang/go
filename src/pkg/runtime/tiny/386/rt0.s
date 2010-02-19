// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

TEXT _rt0_386_pchw(SB), 7, $0
	// Disable interrupts.
	CLI
	
	// Establish stack.
	MOVL	$0x10000, AX
	MOVL	AX, SP
	
	// Set up memory hardware.
	CALL	msetup(SB)

	// _rt0_386 expects to find argc, argv, envv on stack.
	// Set up argv=["kernel"] and envv=[].
	SUBL	$64, SP
	MOVL	$1, 0(SP)
	MOVL	$kernel(SB), 4(SP)
	MOVL	$0, 8(SP)
	MOVL	$0, 12(SP)
	JMP	_rt0_386(SB)

DATA kernel+0(SB)/7, $"kernel\z"
GLOBL kernel(SB), $7
	

