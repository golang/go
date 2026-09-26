// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"
#include "funcdata.h"

// TODO(dmo): generate these with mkpreempt.go, the register sets
// are tightly coupled and this will ensure that we keep them
// all synchronized

// secretEraseRegisters erases any register that may
// have been used with user code within a secret.Do function.
// This is roughly the general purpose and floating point
// registers, barring any reserved registers and registers generally
// considered architectural (amd64 segment registers, arm64 exception registers)
TEXT ·secretEraseRegisters(SB),NOFRAME|NOSPLIT,$0-0
	XORL	AX, AX
	JMP ·secretEraseRegistersMcall(SB)

// Mcall requires an argument in AX. This function
// excludes that register from being cleared
TEXT ·secretEraseRegistersMcall(SB),NOSPLIT|NOFRAME,$0-0
	// integer registers
	XORL	BX, BX
	XORL	CX, CX
	XORL	DX, DX
	XORL	DI, DI
	XORL	SI, SI
	// BP = frame pointer
	// SP = stack pointer
	XORL	R8, R8
	XORL	R9, R9
	XORL	R10, R10
	XORL	R11, R11
	XORL	R12, R12
	XORL	R13, R13
	// R14 = G register
	XORL	R15, R15

	// floating-point registers
	CMPB	internal∕cpu·X86+const_offsetX86HasAVX(SB), $1
	JEQ	avx

	PXOR	X0, X0
	PXOR	X1, X1
	PXOR	X2, X2
	PXOR	X3, X3
	PXOR	X4, X4
	PXOR	X5, X5
	PXOR	X6, X6
	PXOR	X7, X7
	PXOR	X8, X8
	PXOR	X9, X9
	PXOR	X10, X10
	PXOR	X11, X11
	PXOR	X12, X12
	PXOR	X13, X13
	PXOR	X14, X14
	PXOR	X15, X15
	JMP	noavx512

avx:
	// VZEROALL zeroes all of the X0-X15 registers, no matter how wide.
	// That includes Y0-Y15 (256-bit avx) and Z0-Z15 (512-bit avx512).
	VZEROALL

	// Clear all the avx512 state.
	CMPB	internal∕cpu·X86+const_offsetX86HasAVX512(SB), $1
	JNE	noavx512

	// Zero X16-X31
	// VPXORQ r, r, r is a zeroing idiom according to section
	// 3.5.1.7 "Clearing Registers and Dependency Breaking Idioms" in
	// "Intel® 64 and IA-32 Architectures Optimization Reference Manual: Volume 1"
	// (April 2024)
	VPXORQ  Z16, Z16, Z16
	VPXORQ  Z17, Z17, Z17
	VPXORQ  Z18, Z18, Z18
	VPXORQ  Z19, Z19, Z19
	VPXORQ  Z20, Z20, Z20
	VPXORQ  Z21, Z21, Z21
	VPXORQ  Z22, Z22, Z22
	VPXORQ  Z23, Z23, Z23
	VPXORQ  Z24, Z24, Z24
	VPXORQ  Z25, Z25, Z25
	VPXORQ  Z26, Z26, Z26
	VPXORQ  Z27, Z27, Z27
	VPXORQ  Z28, Z28, Z28
	VPXORQ  Z29, Z29, Z29
	VPXORQ  Z30, Z30, Z30
	VPXORQ  Z31, Z31, Z31

	// Zero k0-k7
	// While these are not categorized as zeroing idioms, having them
	// operate on a single register per instruction makes it easy to
	// understand what each instruction does.
	// Note: for wider compatibility these could equally also be KXORW.
	KXORQ	K0, K0, K0
	KXORQ	K1, K1, K1
	KXORQ	K2, K2, K2
	KXORQ	K3, K3, K3
	KXORQ	K4, K4, K4
	KXORQ	K5, K5, K5
	KXORQ	K6, K6, K6
	KXORQ	K7, K7, K7

noavx512:
	// misc registers
	CMPL	BX, BX	//eflags
	// segment registers? Direction flag? Both seem overkill.

	RET
