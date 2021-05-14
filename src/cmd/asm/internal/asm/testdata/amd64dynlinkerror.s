// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test to make sure that if we use R15 after it is clobbered by
// a global variable access while dynamic linking, we get an error.
// See issue 43661.

TEXT ·a1(SB), 0, $0-0
	CMPL runtime·writeBarrier(SB), $0
	MOVL $0, R15
	RET
TEXT ·a2(SB), 0, $0-0
	CMPL runtime·writeBarrier(SB), $0
	MOVQ $0, R15
	RET
TEXT ·a3(SB), 0, $0-0
	CMPL runtime·writeBarrier(SB), $0
	XORL R15, R15
	RET
TEXT ·a4(SB), 0, $0-0
	CMPL runtime·writeBarrier(SB), $0
	XORQ R15, R15
	RET
TEXT ·a5(SB), 0, $0-0
	CMPL runtime·writeBarrier(SB), $0
	XORL R15, R15
	RET
TEXT ·a6(SB), 0, $0-0
	CMPL runtime·writeBarrier(SB), $0
	POPQ R15
	PUSHQ R15
	RET
TEXT ·a7(SB), 0, $0-0
	CMPL runtime·writeBarrier(SB), $0
	MOVQ R15, AX // ERROR "when dynamic linking, R15 is clobbered by a global variable access and is used here"
	RET
TEXT ·a8(SB), 0, $0-0
	CMPL runtime·writeBarrier(SB), $0
	ADDQ AX, R15 // ERROR "when dynamic linking, R15 is clobbered by a global variable access and is used here"
	RET
TEXT ·a9(SB), 0, $0-0
	CMPL runtime·writeBarrier(SB), $0
	ORQ R15, R15 // ERROR "when dynamic linking, R15 is clobbered by a global variable access and is used here"
	RET
TEXT ·a10(SB), 0, $0-0
	CMPL runtime·writeBarrier(SB), $0
	JEQ one
	ORQ R15, R15 // ERROR "when dynamic linking, R15 is clobbered by a global variable access and is used here"
one:
	RET
TEXT ·a11(SB), 0, $0-0
	CMPL runtime·writeBarrier(SB), $0
	JEQ one
	JMP two
one:
	ORQ R15, R15 // ERROR "when dynamic linking, R15 is clobbered by a global variable access and is used here"
two:
	RET
TEXT ·a12(SB), 0, $0-0
	CMPL runtime·writeBarrier(SB), $0
	JMP one
two:
	ORQ R15, R15
	RET
one:
	MOVL $0, R15
	JMP two
