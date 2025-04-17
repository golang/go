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

// Ensure 3-arg instructions get GOT-rewritten without errors.
// See issue 58735.
TEXT ·a13(SB), 0, $0-0
	MULXQ runtime·writeBarrier(SB), AX, CX
	RET

// Various special cases in the use-R15-after-global-access-when-dynlinking check.
// See issue 58632.
TEXT ·a14(SB), 0, $0-0
	CMPL runtime·writeBarrier(SB), $0
	MULXQ R15, AX, BX // ERROR "when dynamic linking, R15 is clobbered by a global variable access and is used here"
	RET
TEXT ·a15(SB), 0, $0-0
	CMPL runtime·writeBarrier(SB), $0
	MULXQ AX, R15, BX
	ADDQ $1, R15
	RET
TEXT ·a16(SB), 0, $0-0
	CMPL runtime·writeBarrier(SB), $0
	MULXQ AX, BX, R15
	ADDQ $1, R15
	RET
TEXT ·a17(SB), 0, $0-0
	CMPL runtime·writeBarrier(SB), $0
	MOVQ (R15), AX // ERROR "when dynamic linking, R15 is clobbered by a global variable access and is used here"
	RET
TEXT ·a18(SB), 0, $0-0
	CMPL runtime·writeBarrier(SB), $0
	MOVQ (CX)(R15*1), AX // ERROR "when dynamic linking, R15 is clobbered by a global variable access and is used here"
	RET
TEXT ·a19(SB), 0, $0-0
	CMPL runtime·writeBarrier(SB), $0
	MOVQ AX, (R15) // ERROR "when dynamic linking, R15 is clobbered by a global variable access and is used here"
	RET
TEXT ·a20(SB), 0, $0-0
	CMPL runtime·writeBarrier(SB), $0
	MOVQ AX, (CX)(R15*1) // ERROR "when dynamic linking, R15 is clobbered by a global variable access and is used here"
	RET
TEXT ·a21(SB), 0, $0-0
	CMPL runtime·writeBarrier(SB), $0
	MOVBLSX AX, R15
	ADDQ $1, R15
	RET
TEXT ·a22(SB), 0, $0-0
	CMPL runtime·writeBarrier(SB), $0
	PMOVMSKB X0, R15
	ADDQ $1, R15
	RET
TEXT ·a23(SB), 0, $0-0
	CMPL runtime·writeBarrier(SB), $0
	LEAQ (AX)(CX*1), R15
	RET
TEXT ·a24(SB), 0, $0-0
	CMPL runtime·writeBarrier(SB), $0
	LEAQ (R15)(AX*1), AX // ERROR "when dynamic linking, R15 is clobbered by a global variable access and is used here"
	RET
TEXT ·a25(SB), 0, $0-0
	CMPL runtime·writeBarrier(SB), $0
	LEAQ (AX)(R15*1), AX // ERROR "when dynamic linking, R15 is clobbered by a global variable access and is used here"
	RET
TEXT ·a26(SB), 0, $0-0
	CMPL runtime·writeBarrier(SB), $0
	IMUL3Q $33, AX, R15
	ADDQ $1, R15
	RET
TEXT ·a27(SB), 0, $0-0
	CMPL runtime·writeBarrier(SB), $0
	IMUL3Q $33, R15, AX // ERROR "when dynamic linking, R15 is clobbered by a global variable access and is used here"
	RET
TEXT ·a28(SB), 0, $0-0
	CMPL runtime·writeBarrier(SB), $0
	PEXTRD $0, X0, R15
	ADDQ $1, R15
	RET
TEXT ·a29(SB), 0, $0-0
	CMPL runtime·writeBarrier(SB), $0
	VPEXTRD $0, X0, R15
	ADDQ $1, R15
	RET
TEXT ·a30(SB), 0, $0-0
	CMPL runtime·writeBarrier(SB), $0
	BSFQ R15, AX // ERROR "when dynamic linking, R15 is clobbered by a global variable access and is used here"
	RET
TEXT ·a31(SB), 0, $0-0
	CMPL runtime·writeBarrier(SB), $0
	BSFQ AX, R15
	ADDQ $1, R15
	RET
TEXT ·a32(SB), 0, $0-0
	CMPL runtime·writeBarrier(SB), $0
	POPCNTL R15, AX // ERROR "when dynamic linking, R15 is clobbered by a global variable access and is used here"
	RET
TEXT ·a33(SB), 0, $0-0
	CMPL runtime·writeBarrier(SB), $0
	POPCNTL AX, R15
	ADDQ $1, R15
	RET
TEXT ·a34(SB), 0, $0-0
	CMPL runtime·writeBarrier(SB), $0
	SHLXQ AX, CX, R15
	ADDQ $1, R15
	RET
