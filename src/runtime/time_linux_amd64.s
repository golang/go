// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !faketime
// +build !faketime

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

#define SYS_clock_gettime	228

// func time.now() (sec int64, nsec int32, mono int64)
TEXT time·now(SB),NOSPLIT,$16-24
	MOVQ	SP, R12 // Save old SP; R12 unchanged by C code.

#ifdef GOEXPERIMENT_regabig
	MOVQ	g_m(R14), BX // BX unchanged by C code.
#else
	get_tls(CX)
	MOVQ	g(CX), AX
	MOVQ	g_m(AX), BX // BX unchanged by C code.
#endif

	// Store CLOCK_REALTIME results directly to return space.
	LEAQ	sec+0(FP), SI

	// Set vdsoPC and vdsoSP for SIGPROF traceback.
	// Save the old values on stack and restore them on exit,
	// so this function is reentrant.
	MOVQ	m_vdsoPC(BX), CX
	MOVQ	m_vdsoSP(BX), DX
	MOVQ	CX, 0(SP)
	MOVQ	DX, 8(SP)

	MOVQ	-8(SI), CX	// Sets CX to function return address.
	MOVQ	CX, m_vdsoPC(BX)
	MOVQ	SI, m_vdsoSP(BX)

#ifdef GOEXPERIMENT_regabig
	CMPQ	R14, m_curg(BX)	// Only switch if on curg.
#else
	CMPQ	AX, m_curg(BX)	// Only switch if on curg.
#endif
	JNE	noswitch

	MOVQ	m_g0(BX), DX
	MOVQ	(g_sched+gobuf_sp)(DX), SP	// Set SP to g0 stack

noswitch:
	SUBQ	$16, SP		// Space for monotonic time results
	ANDQ	$~15, SP	// Align for C code

	MOVL	$0, DI // CLOCK_REALTIME
	MOVQ	runtime·vdsoClockgettimeSym(SB), AX
	CMPQ	AX, $0
	JEQ	fallback
	CALL	AX

	MOVL	$1, DI // CLOCK_MONOTONIC
	LEAQ	0(SP), SI
	MOVQ	runtime·vdsoClockgettimeSym(SB), AX
	CALL	AX

ret:
	MOVQ	0(SP), AX	// sec
	MOVQ	8(SP), DX	// nsec

	MOVQ	R12, SP		// Restore real SP
	// Restore vdsoPC, vdsoSP
	// We don't worry about being signaled between the two stores.
	// If we are not in a signal handler, we'll restore vdsoSP to 0,
	// and no one will care about vdsoPC. If we are in a signal handler,
	// we cannot receive another signal.
	MOVQ	8(SP), CX
	MOVQ	CX, m_vdsoSP(BX)
	MOVQ	0(SP), CX
	MOVQ	CX, m_vdsoPC(BX)

	// sec is in AX, nsec in DX
	// return nsec in AX
	IMULQ	$1000000000, AX
	ADDQ	DX, AX
	MOVQ	AX, mono+16(FP)
	RET

fallback:
	MOVQ	$SYS_clock_gettime, AX
	SYSCALL

	MOVL	$1, DI // CLOCK_MONOTONIC
	LEAQ	0(SP), SI
	MOVQ	$SYS_clock_gettime, AX
	SYSCALL

	JMP	ret
