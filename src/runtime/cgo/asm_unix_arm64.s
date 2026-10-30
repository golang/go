// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.cgobindings && (darwin || (linux && !android)) && arm64

#include "go_asm.h"
#include "textflag.h"
#include "abi_arm64.h"

// cgoCallTrampoline is called by runtime.asmcgocall (or runtime.cgocall)
// using the AAPCS64 C ABI with R0 pointing to a cgoCallFrame.
TEXT ·cgoCallTrampoline(SB),NOSPLIT|NOFRAME,$0-0
	SUB	$32, RSP
	STP	(R29, R30), 0(RSP)
	MOVD	RSP, R29
	MOVD	R19, 16(RSP)
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
	MOVD	$0, R0
	MOVD	16(RSP), R19
	LDP	0(RSP), (R29, R30)
	ADD	$32, RSP
	RET

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

// Indirectly call Go targets via R9 to avoid the linker's static NOSPLIT
// stack overflow check: these entrypoints are only invoked from C on a
// full-size C/pthread stack after saving host registers in CGO_PROLOGUE.
#define CALL_GO(fn) \
	MOVD	$fn(SB), R9; \
	BL	(R9)

TEXT ·x_cgo_threadentry(SB),NOSPLIT|NOFRAME,$0-0
	CGO_PROLOGUE
	MOVD	R0, 8(RSP)
	CALL_GO(·threadentry)
	CGO_EPILOGUE
	MOVD	$0, R0
	RET

TEXT x_cgo_init<>(SB),NOSPLIT|NOFRAME,$0-0
	CGO_PROLOGUE
	STP	(R0, R1), 8(RSP)
	CALL_GO(·cgoInit)
	CGO_EPILOGUE
	RET

GLOBL _cgo_init(SB), NOPTR, $8
DATA _cgo_init+0(SB)/8, $x_cgo_init<>(SB)
