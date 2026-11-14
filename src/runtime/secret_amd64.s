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
	// Default to all vector/mask XINUSE bits set (0xE6 = XMM 0x02 | YMM_Hi128 0x04 |
	// OpMask 0x20 | ZMM_Hi256 0x40 | Hi16_ZMM 0x80) if XGETBV(ECX=1) is unavailable.
	MOVL $(const_xcr0_AX_SSE|const_xcr0_AX_YMM|const_xcr0_AX_OpMask|const_xcr0_AX_ZMM_Hi256|const_xcr0_AX_Hi16_ZMM), BX
	CMPB	internal∕cpu·X86+const_offsetX86HasXGETBV1(SB), $1
	JNE	have_xinuse
	// Preserve AX (which holds mcall's fn argument) across XGETBV (which writes EAX:EDX),
	// then re-zero R8, CX, and DX so no state leaks in GP registers.
	MOVQ	AX, R8
	MOVL	$1, CX
	XGETBV
	MOVL	AX, BX
	MOVQ	R8, AX
	XORL	R8, R8 // Not strictly necessary, since it is a copy of AX which is leaked on purpose.
	XORL	CX, CX // Not strictly necessary, since "1" set above means OS/HW provides XGETBV
	XORL	DX, DX // Don't leak XINUSE state.

	// BX now holds the low 32 bits of (XCR0 & XINUSE)

have_xinuse:
	// Test whether any part of Z0-Z15 (XMM 0x02 | YMM_Hi128 0x04 | ZMM_Hi256 0x40)
	// is non-INIT before executing VZEROALL.
	TESTL	$(const_xcr0_AX_SSE|const_xcr0_AX_YMM|const_xcr0_AX_ZMM_Hi256), BX
	JZ	clear_z16_31

	// VZEROALL zeroes all of the X0-X15 registers, no matter how wide.
	// That includes Y0-Y15 (256-bit avx) and Z0-Z15 (512-bit avx512),
	// and resets their XINUSE bits (1, 2, 6) to 0.
	VZEROALL

clear_z16_31:
	// Clear the avx512 state (Z16-Z31 and K0-K7) only if in use,
	// to avoid setting their XINUSE bits when they are already in INIT state.
	CMPB	internal∕cpu·X86+const_offsetX86HasAVX512(SB), $1
	JNE	noavx512

	TESTL	$const_xcr0_AX_Hi16_ZMM, BX
	JZ	clear_k

	// Zero Z16-Z31 via 128-bit EVEX zeroing idioms without touching Z0-Z15
	// so XINUSE[1,2,6] stay 0 and without triggering 512-bit execution
	// license transitions.
	VPXORQ	X16, X16, X16
	VPXORQ	X17, X17, X17
	VPXORQ	X18, X18, X18
	VPXORQ	X19, X19, X19
	VPXORQ	X20, X20, X20
	VPXORQ	X21, X21, X21
	VPXORQ	X22, X22, X22
	VPXORQ	X23, X23, X23
	VPXORQ	X24, X24, X24
	VPXORQ	X25, X25, X25
	VPXORQ	X26, X26, X26
	VPXORQ	X27, X27, X27
	VPXORQ	X28, X28, X28
	VPXORQ	X29, X29, X29
	VPXORQ	X30, X30, X30
	VPXORQ	X31, X31, X31

clear_k:
	TESTL	$const_xcr0_AX_OpMask, BX
	JZ	noavx512

	// Zero k0-k7 via self-XOR zeroing idioms so hardware resets XINUSE[5] (0x20)
	// to INIT state (0).
	KXORQ	K0, K0, K0
	KXORQ	K1, K1, K1
	KXORQ	K2, K2, K2
	KXORQ	K3, K3, K3
	KXORQ	K4, K4, K4
	KXORQ	K5, K5, K5
	KXORQ	K6, K6, K6
	KXORQ	K7, K7, K7

noavx512:
	// Clear (XCR0 & XINUSE) copy in BX. It leaks state about what registers were in use.
	XORL	BX, BX
	// Clear eflags (unlike XOR, CMP puts AF as well as CF/OF/SF/PF into known states)
	CMPL	BX, BX
	// segment registers? Direction flag? Both seem overkill.

	RET
