// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && !msan && !tsan && windows && arm64

#include "go_asm.h"
#include "textflag.h"
#include "abi_arm64.h"

// cgoCallTrampoline is called on the system stack by runtime.asmcgocall with
// a *cgoCallFrame in R0 using the Windows ARM64 C ABI. It returns
// TEB.LastErrorValue in R0, so that runtime.cgocall returns the error
// before runtime.exitsyscall can clobber it (see cgocallErr).
TEXT ·cgoCallTrampoline(SB),NOSPLIT|NOFRAME,$0-0
	// Go unwinders expect the saved LR at SP as soon as SP changes.
	// The C frame pointer needs a separate (FP, LR) record.
	STP.W	(R30, R19), -32(RSP)
	STP	(R29, R30), 16(RSP)
	ADD	$16, RSP, R29
	MOVD	R0, R19
	MOVD	cgoCallFrame_fn(R0), R9
	MOVD	cgoCallFrame_a1(R0), R1
	MOVD	cgoCallFrame_a2(R0), R2
	MOVD	cgoCallFrame_a3(R0), R3
	MOVD	cgoCallFrame_a4(R0), R4
	MOVD	cgoCallFrame_a5(R0), R5
	MOVD	cgoCallFrame_a0(R0), R0
	BL	(R9)
	MOVD	R0, cgoCallFrame_r1(R19)
	MOVWU	0x68(R18_PLATFORM), R0	// TEB.LastErrorValue
	MOVD	16(RSP), R29
	LDP.P	32(RSP), (R30, R19)
	RET

// CGO_PROLOGUE leaves 48 bytes at the bottom of the frame for the register
// argument spill area that Go ABIInternal callees use at 8(RSP) and up.
#define CGO_PROLOGUE \
	SUB	$208, RSP; \
	SAVE_R19_TO_R28(8*6); \
	SAVE_F8_TO_F15(8*16); \
	STP	(R29, R30), (8*24)(RSP); \
	ADD	$(8*24), RSP, R29; \
	MOVD	ZR, g

#define CGO_EPILOGUE \
	RESTORE_R19_TO_R28(8*6); \
	RESTORE_F8_TO_F15(8*16); \
	LDP	(8*24)(RSP), (R29, R30); \
	ADD	$208, RSP

TEXT _cgo_wait_runtime_init_done(SB),NOSPLIT|NOFRAME,$0-0
	// Save R27 in caller-saved R10: global accesses below clobber R27
	// (callee-saved in the C ABI) under -dynlink.
	MOVD	R27, R10
	MOVD	$·runtime_init_done(SB), R9
	LDARW	(R9), R9
	CMPW	$2, R9
	BNE	slow
	MOVD	$·cgo_context_function(SB), R9
	LDAR	(R9), R9
	CBNZ	R9, slow
	MOVD	R10, R27
	MOVD	$0, R0
	RET
slow:
	MOVD	R10, R27
	CGO_PROLOGUE
	BL	·cgoWaitRuntimeInitDone<ABIInternal>(SB)
	CGO_EPILOGUE
	RET

TEXT _cgo_release_context(SB),NOSPLIT|NOFRAME,$0-0
	CBZ	R0, done
	CGO_PROLOGUE
	BL	·cgoReleaseContext<ABIInternal>(SB)
	CGO_EPILOGUE
done:
	RET
