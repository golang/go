// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !goexperiment.arm64e

#include "textflag.h"

// Set the x_crosscall2_ptr C function pointer variable point to crosscall2.
// It's such a pointer chain: _crosscall2_ptr -> x_crosscall2_ptr -> crosscall2
// Use a local trampoline, to avoid taking the address of a dynamically exported
// function.
TEXT ·set_crosscall2(SB),NOSPLIT,$0-0
	MOVD	_crosscall2_ptr(SB), R1
	MOVD	$crosscall2_trampoline<>(SB), R2
	MOVD	R2, (R1)
	RET

TEXT crosscall2_trampoline<>(SB),NOSPLIT,$0-0
	JMP	crosscall2(SB)
