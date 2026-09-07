// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimesecret

// Note: this assembly file is used for testing only.
// We need to access registers directly to properly test
// that secrets are erased and go test doesn't like to conditionally
// include assembly files.
// These functions defined in the package proper and we
// rely on the linker to prune these away in regular builds

#include "go_asm.h"
#include "funcdata.h"

#define REGTMP1	R27
#define REGTMP2	R28

#define dirtyReg(dstReg)		\
	ADDV	$0x1, REGTMP1		\
	MOVV	REGTMP1, dstReg		\


#define dirtyVReg(dstReg)		\
	ADDV	$0x1, REGTMP1		\
	VMOVQ	REGTMP1, dstReg.V2	\


#define dirtyXReg(dstReg)		\
	ADDV	$0x1, REGTMP1		\
	XVMOVQ	REGTMP1, dstReg.V4	\


// func loadRegisters(p unsafe.Pointer)
TEXT ·loadRegisters(SB),0,$0-8
	MOVV	p+0(FP), R4

	MOVV	(R4), R16
	MOVV	(R4), R17
	MOVV	(R4), R18
	MOVV	(R4), R19

	MOVD	(R4), F20
	MOVD	(R4), F21
	MOVD	(R4), F22
	MOVD	(R4), F23

	MOVBU	internal∕cpu·Loong64+const_offsetLOONG64HasLSX(SB), R5
	BEQ	R5, return
	VMOVQ	(R4), V24
	VMOVQ	(R4), V25
	VMOVQ	(R4), V26
	VMOVQ	(R4), V27

	MOVBU	internal∕cpu·Loong64+const_offsetLOONG64HasLASX(SB), R5
	BEQ	R5, return
	XVMOVQ	(R4), X28
	XVMOVQ	(R4), X29
	XVMOVQ	(R4), X30
	XVMOVQ	(R4), X31
return:
	RET

// func spillRegisters(p unsafe.Pointer) uintptr
TEXT ·spillRegisters(SB),0,$0-16
	MOVV	p+0(FP), R4
	MOVV	$0, R5

	MOVV	R16, 0(R4)
	MOVV	R17, 8(R4)
	MOVV	R18, 16(R4)
	MOVV	R19, 24(R4)

	MOVD	F20, 32(R4)
	MOVD	F21, 40(R4)
	MOVD	F22, 48(R4)
	MOVD	F23, 56(R4)
	ADDV	$64, R4
	ADDV	$64, R5

	MOVBU	internal∕cpu·Loong64+const_offsetLOONG64HasLSX(SB), R6
	BEQ	R6, return
	VMOVQ	V24, 0(R4)
	VMOVQ	V25, 16(R4)
	VMOVQ	V26, 32(R4)
	VMOVQ	V27, 48(R4)
	ADDV	$64, R4
	ADDV	$64, R5

	MOVBU	internal∕cpu·Loong64+const_offsetLOONG64HasLASX(SB), R6
	BEQ	R6, return
	XVMOVQ	X28, 0(R4)
	XVMOVQ	X29, 32(R4)
	XVMOVQ	X30, 64(R4)
	XVMOVQ	X31, 96(R4)
	ADDV	$128, R4
	ADDV	$128, R5
return:
	MOVV	R5, ret+8(FP)
	RET

TEXT ·useSecret(SB),0,$0-24
	NO_LOCAL_POINTERS

	// Load secret into R27
	MOVV	secret_base+0(FP), REGTMP2
	MOVV	(REGTMP2), REGTMP1

	MOVBU	internal∕cpu·Loong64+const_offsetLOONG64HasLASX(SB), REGTMP2
	BNE	REGTMP2, dirtyLASX

	MOVBU	internal∕cpu·Loong64+const_offsetLOONG64HasLSX(SB), REGTMP2
	BNE	REGTMP2, dirtyLSX

	// Dirty the floating point registers
	dirtyReg(F0)
	dirtyReg(F1)
	dirtyReg(F2)
	dirtyReg(F3)
	dirtyReg(F4)
	dirtyReg(F5)
	dirtyReg(F6)
	dirtyReg(F7)
	dirtyReg(F8)
	dirtyReg(F9)
	dirtyReg(F10)
	dirtyReg(F11)
	dirtyReg(F12)
	dirtyReg(F13)
	dirtyReg(F14)
	dirtyReg(F15)
	dirtyReg(F16)
	dirtyReg(F17)
	dirtyReg(F18)
	dirtyReg(F19)
	dirtyReg(F20)
	dirtyReg(F21)
	dirtyReg(F22)
	dirtyReg(F23)
	dirtyReg(F24)
	dirtyReg(F25)
	dirtyReg(F26)
	dirtyReg(F27)
	dirtyReg(F28)
	dirtyReg(F29)
	dirtyReg(F30)
	dirtyReg(F31)
	JMP	dirtyInt

	// Dirty the lsx registers
dirtyLSX:
	dirtyVReg(V0)
	dirtyVReg(V1)
	dirtyVReg(V2)
	dirtyVReg(V3)
	dirtyVReg(V4)
	dirtyVReg(V5)
	dirtyVReg(V6)
	dirtyVReg(V7)
	dirtyVReg(V8)
	dirtyVReg(V9)
	dirtyVReg(V10)
	dirtyVReg(V11)
	dirtyVReg(V12)
	dirtyVReg(V13)
	dirtyVReg(V14)
	dirtyVReg(V15)
	dirtyVReg(V16)
	dirtyVReg(V17)
	dirtyVReg(V18)
	dirtyVReg(V19)
	dirtyVReg(V20)
	dirtyVReg(V21)
	dirtyVReg(V22)
	dirtyVReg(V23)
	dirtyVReg(V24)
	dirtyVReg(V25)
	dirtyVReg(V26)
	dirtyVReg(V27)
	dirtyVReg(V28)
	dirtyVReg(V29)
	dirtyVReg(V30)
	dirtyVReg(V31)
	JMP	dirtyInt

	// Dirty the lasx registers
dirtyLASX:
	dirtyXReg(X0)
	dirtyXReg(X1)
	dirtyXReg(X2)
	dirtyXReg(X3)
	dirtyXReg(X4)
	dirtyXReg(X5)
	dirtyXReg(X6)
	dirtyXReg(X7)
	dirtyXReg(X8)
	dirtyXReg(X9)
	dirtyXReg(X10)
	dirtyXReg(X11)
	dirtyXReg(X12)
	dirtyXReg(X13)
	dirtyXReg(X14)
	dirtyXReg(X15)
	dirtyXReg(X16)
	dirtyXReg(X17)
	dirtyXReg(X18)
	dirtyXReg(X19)
	dirtyXReg(X20)
	dirtyXReg(X21)
	dirtyXReg(X22)
	dirtyXReg(X23)
	dirtyXReg(X24)
	dirtyXReg(X25)
	dirtyXReg(X26)
	dirtyXReg(X27)
	dirtyXReg(X28)
	dirtyXReg(X29)
	dirtyXReg(X30)
	dirtyXReg(X31)

dirtyInt:
	// Scatter secret across registers.
	// Increment low byte so we can tell which register
	// a leaking secret came from.
	dirtyReg(R4)
	dirtyReg(R5)
	dirtyReg(R6)
	dirtyReg(R7)
	dirtyReg(R8)
	dirtyReg(R9)
	dirtyReg(R10)
	dirtyReg(R11)
	dirtyReg(R12)
	dirtyReg(R13)
	dirtyReg(R14)
	dirtyReg(R15)
	dirtyReg(R16)
	dirtyReg(R17)
	dirtyReg(R18)
	dirtyReg(R19)
	RET
