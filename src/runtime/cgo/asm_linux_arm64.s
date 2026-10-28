// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.cgobindings && linux && !android && arm64

#include "go_asm.h"
#include "textflag.h"

// cgoSyscallTrampoline is called by runtime.cgocall using the AAPCS64 C ABI
// with R0 pointing to a cgoSyscallFrame. It calls fn(a0, a1, a2) and, if fn
// returns -1, fetches errno via errnoFn before returning so both calls
// execute on the same OS thread.
TEXT ·cgoSyscallTrampoline(SB),NOSPLIT|NOFRAME,$0-0
	SUB	$32, RSP
	STP	(R29, R30), 0(RSP)
	MOVD	RSP, R29
	MOVD	R19, 16(RSP)
	MOVD	R0, R19
	MOVD	cgoSyscallFrame_fn(R0), R9
	MOVD	cgoSyscallFrame_a1(R0), R1
	MOVD	cgoSyscallFrame_a2(R0), R2
	MOVD	cgoSyscallFrame_a0(R0), R0
	BL	(R9)
	MOVW	R0, R0
	MOVD	$-1, R1
	CMP	R1, R0
	BNE	ok
	CALL	libc_errno_location(SB)
	MOVW	(R0), R0
ok:
	MOVD	R0, cgoSyscallFrame_r1(R19)
	MOVD	$0, R0
	MOVD	16(RSP), R19
	LDP	0(RSP), (R29, R30)
	ADD	$32, RSP
	RET
