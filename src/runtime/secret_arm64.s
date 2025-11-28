// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"
#include "funcdata.h"

TEXT ·secretEraseRegisters(SB),NOFRAME|NOSPLIT,$0-0
	MOVD	ZR, R0
	MOVD	ZR, R26
	JMP ·secretEraseRegistersMcall(SB)

// Mcall requires an argument in R0 and does not have a
// stack frame to spill into. Additionally, there is no stack
// to spill the link register into. This function deliberately
// doesn't clear R0 and R26, and Mcall uses R26 as a link register.
TEXT ·secretEraseRegistersMcall(SB),NOFRAME|NOSPLIT,$0-0
	// integer registers
	MOVD	ZR, R1
	MOVD	ZR, R2
	MOVD	ZR, R3
	MOVD	ZR, R4
	MOVD	ZR, R5
	MOVD	ZR, R6
	MOVD	ZR, R7
	MOVD	ZR, R8
	MOVD	ZR, R9
	MOVD	ZR, R10
	MOVD	ZR, R11
	MOVD	ZR, R12
	MOVD	ZR, R13
	MOVD	ZR, R14
	MOVD	ZR, R15
	MOVD	ZR, R16
	MOVD	ZR, R17
	// R18 = platform register
	MOVD	ZR, R19
	MOVD	ZR, R20
	MOVD	ZR, R21
	MOVD	ZR, R22
	MOVD	ZR, R23
	MOVD	ZR, R24
	MOVD	ZR, R25
	// R26 used for extra link register in mcall where we can't spill
	MOVD	ZR, R27
	// R28 = g
	// R29 = frame pointer
	// R30 = link pointer (return address)
	// R31 = stack pointer

	// floating point registers
	// (also clears simd registers)
	FMOVD	ZR, F0
	FMOVD	ZR, F1
	FMOVD	ZR, F2
	FMOVD	ZR, F3
	FMOVD	ZR, F4
	FMOVD	ZR, F5
	FMOVD	ZR, F6
	FMOVD	ZR, F7
	FMOVD	ZR, F8
	FMOVD	ZR, F9
	FMOVD	ZR, F10
	FMOVD	ZR, F11
	FMOVD	ZR, F12
	FMOVD	ZR, F13
	FMOVD	ZR, F14
	FMOVD	ZR, F15
	FMOVD	ZR, F16
	FMOVD	ZR, F17
	FMOVD	ZR, F18
	FMOVD	ZR, F19
	FMOVD	ZR, F20
	FMOVD	ZR, F21
	FMOVD	ZR, F22
	FMOVD	ZR, F23
	FMOVD	ZR, F24
	FMOVD	ZR, F25
	FMOVD	ZR, F26
	FMOVD	ZR, F27
	FMOVD	ZR, F28
	FMOVD	ZR, F29
	FMOVD	ZR, F30
	FMOVD	ZR, F31

	// misc registers
	CMP	ZR, ZR // N,Z,C,V flags

	RET
