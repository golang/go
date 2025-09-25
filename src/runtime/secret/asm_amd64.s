// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Note: this assembly file is used for testing only.
// We need to access registers directly to properly test
// that secrets are erased and go test doesn't like to conditionally
// include assembly files.
// These functions defined in the package proper and we
// rely on the linker to prune these away in regular builds

#include "go_asm.h"
#include "funcdata.h"

TEXT ·loadRegisters(SB),0,$0-8
	MOVQ	p+0(FP), AX

	MOVQ	(AX), R10
	MOVQ	(AX), R11
	MOVQ	(AX), R12
	MOVQ	(AX), R13

	MOVOU	(AX), X1
	MOVOU	(AX), X2
	MOVOU	(AX), X3
	MOVOU	(AX), X4

	CMPB	internal∕cpu·X86+const_offsetX86HasAVX(SB), $1
	JNE	return

	VMOVDQU	(AX), Y5
	VMOVDQU	(AX), Y6
	VMOVDQU	(AX), Y7
	VMOVDQU	(AX), Y8

	CMPB	internal∕cpu·X86+const_offsetX86HasAVX512(SB), $1
	JNE	return

	VMOVUPD	(AX), Z14
	VMOVUPD	(AX), Z15
	VMOVUPD	(AX), Z16
	VMOVUPD	(AX), Z17

	KMOVQ	(AX), K2
	KMOVQ	(AX), K3
	KMOVQ	(AX), K4
	KMOVQ	(AX), K5

return:
	RET

TEXT ·spillRegisters(SB),0,$0-16
	MOVQ	p+0(FP), AX
	MOVQ	AX, BX

	MOVQ	R10, (AX)
	MOVQ	R11, 8(AX)
	MOVQ	R12, 16(AX)
	MOVQ	R13, 24(AX)
	ADDQ	$32, AX

	MOVOU	X1, (AX)
	MOVOU	X2, 16(AX)
	MOVOU	X3, 32(AX)
	MOVOU	X4, 48(AX)
	ADDQ	$64, AX

	CMPB	internal∕cpu·X86+const_offsetX86HasAVX(SB), $1
	JNE	return

	VMOVDQU	Y5, (AX)
	VMOVDQU	Y6, 32(AX)
	VMOVDQU	Y7, 64(AX)
	VMOVDQU	Y8, 96(AX)
	ADDQ	$128, AX

	CMPB	internal∕cpu·X86+const_offsetX86HasAVX512(SB), $1
	JNE	return

	VMOVUPD	Z14, (AX)
	ADDQ	$64, AX
	VMOVUPD	Z15, (AX)
	ADDQ	$64, AX
	VMOVUPD	Z16, (AX)
	ADDQ	$64, AX
	VMOVUPD	Z17, (AX)
	ADDQ	$64, AX

	KMOVQ	K2, (AX)
	ADDQ	$8, AX
	KMOVQ	K3, (AX)
	ADDQ	$8, AX
	KMOVQ	K4, (AX)
	ADDQ	$8, AX
	KMOVQ	K5, (AX)
	ADDQ	$8, AX

return:
	SUBQ	BX, AX
	MOVQ	AX, ret+8(FP)
	RET

TEXT ·useSecret(SB),0,$64-24
	NO_LOCAL_POINTERS

	// Load secret into AX
	MOVQ	secret_base+0(FP), AX
	MOVQ	(AX), AX

	// Scatter secret all across registers.
	// Increment low byte so we can tell which register
	// a leaking secret came from.
	ADDQ	$2, AX // add 2 so Rn has secret #n.
	MOVQ	AX, BX
	INCQ	AX
	MOVQ	AX, CX
	INCQ	AX
	MOVQ	AX, DX
	INCQ	AX
	MOVQ	AX, SI
	INCQ	AX
	MOVQ	AX, DI
	INCQ	AX
	MOVQ	AX, BP
	INCQ	AX
	MOVQ	AX, R8
	INCQ	AX
	MOVQ	AX, R9
	INCQ	AX
	MOVQ	AX, R10
	INCQ	AX
	MOVQ	AX, R11
	INCQ	AX
	MOVQ	AX, R12
	INCQ	AX
	MOVQ	AX, R13
	INCQ	AX
	MOVQ	AX, R14
	INCQ	AX
	MOVQ	AX, R15

	CMPB	internal∕cpu·X86+const_offsetX86HasAVX512(SB), $1
	JNE	noavx512
	VMOVUPD	(SP), Z0
	VMOVUPD	(SP), Z1
	VMOVUPD	(SP), Z2
	VMOVUPD	(SP), Z3
	VMOVUPD	(SP), Z4
	VMOVUPD	(SP), Z5
	VMOVUPD	(SP), Z6
	VMOVUPD	(SP), Z7
	VMOVUPD	(SP), Z8
	VMOVUPD	(SP), Z9
	VMOVUPD	(SP), Z10
	VMOVUPD	(SP), Z11
	VMOVUPD	(SP), Z12
	VMOVUPD	(SP), Z13
	VMOVUPD	(SP), Z14
	VMOVUPD	(SP), Z15
	VMOVUPD	(SP), Z16
	VMOVUPD	(SP), Z17
	VMOVUPD	(SP), Z18
	VMOVUPD	(SP), Z19
	VMOVUPD	(SP), Z20
	VMOVUPD	(SP), Z21
	VMOVUPD	(SP), Z22
	VMOVUPD	(SP), Z23
	VMOVUPD	(SP), Z24
	VMOVUPD	(SP), Z25
	VMOVUPD	(SP), Z26
	VMOVUPD	(SP), Z27
	VMOVUPD	(SP), Z28
	VMOVUPD	(SP), Z29
	VMOVUPD	(SP), Z30
	VMOVUPD	(SP), Z31

noavx512:
	MOVOU	(SP), X0
	MOVOU	(SP), X1
	MOVOU	(SP), X2
	MOVOU	(SP), X3
	MOVOU	(SP), X4
	MOVOU	(SP), X5
	MOVOU	(SP), X6
	MOVOU	(SP), X7
	MOVOU	(SP), X8
	MOVOU	(SP), X9
	MOVOU	(SP), X10
	MOVOU	(SP), X11
	MOVOU	(SP), X12
	MOVOU	(SP), X13
	MOVOU	(SP), X14
	MOVOU	(SP), X15

	// Put secret on the stack.
	INCQ	AX
	MOVQ	AX, (SP)
	MOVQ	AX, 8(SP)
	MOVQ	AX, 16(SP)
	MOVQ	AX, 24(SP)
	MOVQ	AX, 32(SP)
	MOVQ	AX, 40(SP)
	MOVQ	AX, 48(SP)
	MOVQ	AX, 56(SP)

	// Delay a bit.  This makes it more likely that
	// we will be the target of a signal while
	// registers contain secrets.
	// It also tests the path from G stack to M stack
	// to scheduler and back.
	CALL	·delay(SB)

	RET
