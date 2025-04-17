// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !faketime

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

#define SYS_clock_gettime	228

// func now() (sec int64, nsec int32, mono int64)
TEXT time·now<ABIInternal>(SB),NOSPLIT,$16-24
	MOVQ	SP, R12 // Save old SP; R12 unchanged by C code.

	MOVQ	g_m(R14), BX // BX unchanged by C code.

	// Set vdsoPC and vdsoSP for SIGPROF traceback.
	// Save the old values on stack and restore them on exit,
	// so this function is reentrant.
	MOVQ	m_vdsoPC(BX), CX
	MOVQ	m_vdsoSP(BX), DX
	MOVQ	CX, 0(SP)
	MOVQ	DX, 8(SP)

	LEAQ	sec+0(FP), DX
	MOVQ	-8(DX), CX	// Sets CX to function return address.
	MOVQ	CX, m_vdsoPC(BX)
	MOVQ	DX, m_vdsoSP(BX)

	CMPQ	R14, m_curg(BX)	// Only switch if on curg.
	JNE	noswitch

	MOVQ	m_g0(BX), DX
	MOVQ	(g_sched+gobuf_sp)(DX), SP	// Set SP to g0 stack

noswitch:
	SUBQ	$32, SP		// Space for two time results
	ANDQ	$~15, SP	// Align for C code

	MOVL	$0, DI // CLOCK_REALTIME
	LEAQ	16(SP), SI
	MOVQ	runtime·vdsoClockgettimeSym(SB), AX
	CMPQ	AX, $0
	JEQ	fallback
	CALL	AX

	MOVL	$1, DI // CLOCK_MONOTONIC
	LEAQ	0(SP), SI
	MOVQ	runtime·vdsoClockgettimeSym(SB), AX
	CALL	AX

ret:
	MOVQ	16(SP), AX	// realtime sec
	MOVQ	24(SP), DI	// realtime nsec (moved to BX below)
	MOVQ	0(SP), CX	// monotonic sec
	IMULQ	$1000000000, CX
	MOVQ	8(SP), DX	// monotonic nsec

	MOVQ	R12, SP		// Restore real SP

	// Restore vdsoPC, vdsoSP
	// We don't worry about being signaled between the two stores.
	// If we are not in a signal handler, we'll restore vdsoSP to 0,
	// and no one will care about vdsoPC. If we are in a signal handler,
	// we cannot receive another signal.
	MOVQ	8(SP), SI
	MOVQ	SI, m_vdsoSP(BX)
	MOVQ	0(SP), SI
	MOVQ	SI, m_vdsoPC(BX)

	// set result registers; AX is already correct
	MOVQ	DI, BX
	ADDQ	DX, CX
	RET

fallback:
	MOVQ	$SYS_clock_gettime, AX
	SYSCALL

	MOVL	$1, DI // CLOCK_MONOTONIC
	LEAQ	0(SP), SI
	MOVQ	$SYS_clock_gettime, AX
	SYSCALL

	JMP	ret
