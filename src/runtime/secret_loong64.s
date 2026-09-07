// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"
#include "funcdata.h"

TEXT ·secretEraseRegisters(SB),NOFRAME|NOSPLIT,$0-0
	MOVV	R0, R4
	MOVV	R0, R29
	JMP ·secretEraseRegistersMcall(SB)

// Mcall requires an argument in R4 and does not have a
// stack frame to spill into. Additionally, there is no stack
// to spill the link register into. This function deliberately
// doesn't clear R4 and R29, and Mcall uses R29 as a link register.
TEXT ·secretEraseRegistersMcall(SB),NOFRAME|NOSPLIT,$0-0
	MOVBU internal∕cpu·Loong64+const_offsetLOONG64HasLASX(SB), R5
	BNE R5, eraseLasx
	MOVBU internal∕cpu·Loong64+const_offsetLOONG64HasLSX(SB), R5
	BNE R5, eraseLsx

	// floating-point registers
	MOVV	R0, F0
	MOVV	R0, F1
	MOVV	R0, F2
	MOVV	R0, F3
	MOVV	R0, F4
	MOVV	R0, F5
	MOVV	R0, F6
	MOVV	R0, F7
	MOVV	R0, F8
	MOVV	R0, F9
	MOVV	R0, F10
	MOVV	R0, F11
	MOVV	R0, F12
	MOVV	R0, F13
	MOVV	R0, F14
	MOVV	R0, F15
	MOVV	R0, F16
	MOVV	R0, F17
	MOVV	R0, F18
	MOVV	R0, F19
	MOVV	R0, F20
	MOVV	R0, F21
	MOVV	R0, F22
	MOVV	R0, F23
	MOVV	R0, F24
	MOVV	R0, F25
	MOVV	R0, F26
	MOVV	R0, F27
	MOVV	R0, F28
	MOVV	R0, F29
	MOVV	R0, F30
	MOVV	R0, F31
	JMP	eraseInt

	// lsx registers
eraseLsx:
	VXORV	V0,  V0
	VXORV	V1,  V1
	VXORV	V2,  V2
	VXORV	V4,  V3
	VXORV	V4,  V4
	VXORV	V5,  V5
	VXORV	V6,  V6
	VXORV	V7,  V7
	VXORV	V8,  V8
	VXORV	V9,  V9
	VXORV	V10, V10
	VXORV	V11, V11
	VXORV	V12, V12
	VXORV	V13, V13
	VXORV	V14, V14
	VXORV	V15, V15
	VXORV	V16, V16
	VXORV	V17, V17
	VXORV	V18, V18
	VXORV	V19, V19
	VXORV	V20, V20
	VXORV	V21, V21
	VXORV	V22, V22
	VXORV	V23, V23
	VXORV	V24, V24
	VXORV	V25, V25
	VXORV	V26, V26
	VXORV	V27, V27
	VXORV	V28, V28
	VXORV	V29, V29
	VXORV	V30, V30
	VXORV	V31, V31
	JMP	eraseInt

	// lasx registers
eraseLasx:
	XVXORV	X0, X0
	XVXORV	X1, X1
	XVXORV	X2, X2
	XVXORV	X3, X3
	XVXORV	X4, X4
	XVXORV	X5, X5
	XVXORV	X6, X6
	XVXORV	X7, X7
	XVXORV	X8, X8
	XVXORV	X9, X9
	XVXORV	X10, X10
	XVXORV	X11, X11
	XVXORV	X12, X12
	XVXORV	X13, X13
	XVXORV	X14, X14
	XVXORV	X15, X15
	XVXORV	X16, X16
	XVXORV	X17, X17
	XVXORV	X18, X18
	XVXORV	X19, X19
	XVXORV	X20, X20
	XVXORV	X21, X21
	XVXORV	X22, X22
	XVXORV	X23, X23
	XVXORV	X24, X24
	XVXORV	X25, X25
	XVXORV	X26, X26
	XVXORV	X27, X27
	XVXORV	X28, X28
	XVXORV	X29, X29
	XVXORV	X30, X30
	XVXORV	X31, X31

eraseInt:
	// integer registers
	//   R1 = link pointer (return address)
	//   R2 = TP
	//   R3 = stack pointer
	//   R4
	//   R22 = g
	//   R29 used for extra link register in mcall where we can't spill

	XOR	R5, R5
	XOR	R6, R6
	XOR	R7, R7
	XOR	R8, R8
	XOR	R9, R9
	XOR	R10, R10
	XOR	R11, R11
	XOR	R12, R12
	XOR	R13, R13
	XOR	R14, R14
	XOR	R15, R15
	XOR	R16, R16
	XOR	R17, R17
	XOR	R18, R18
	XOR	R19, R19
	XOR	R20, R20
	XOR	R21, R21
	XOR	R23, R23
	XOR	R24, R24
	XOR	R25, R25
	XOR	R26, R26
	XOR	R27, R27
	XOR	R28, R28
	XOR	R30, R30
	XOR	R31, R31

	// misc registers
	MOVV	R0, FCC0
	MOVV	R0, FCC1
	MOVV	R0, FCC2
	MOVV	R0, FCC3
	MOVV	R0, FCC4
	MOVV	R0, FCC5
	MOVV	R0, FCC6
	MOVV	R0, FCC7
	MOVV	R0, FCSR0
	RET
