// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.arm64e

#include "textflag.h"

// As in asm_noarm64e_arm64.s, but C calls x_crosscall2_ptr through an
// authenticating branch, so the pointer is signed with key IA and a zero
// modifier.
TEXT ·set_crosscall2(SB),NOSPLIT,$0-0
	MOVD	_crosscall2_ptr(SB), R1
	MOVD	$crosscall2_trampoline<>(SB), R2
	PACIZA	R2
	MOVD	R2, (R1)
	RET

TEXT crosscall2_trampoline<>(SB),NOSPLIT,$0-0
	JMP	crosscall2(SB)
