// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build valgrind && linux

#include "textflag.h"

// See valgrind_amd64.s for notes about this assembly.

// func valgrindClientRequest(uintptr, uintptr, uintptr, uintptr, uintptr, uintptr) (ret uintptr)
TEXT runtime·valgrindClientRequest(SB), NOSPLIT, $0-56
	// Load the address of the first of the (contiguous) arguments into x4.
	MOVD $args+0(FP), R4

	// Zero x3, since some requests may not populate it.
	MOVD ZR, R3

	// Emit the special preamble.
	ROR $3, R12; ROR $13, R12
	ROR $51, R12; ROR $61, R12

	// "Execute" the client request.
	ORR R10, R10

	// Copy the result out of x3.
	MOVD R3, ret+48(FP)

	RET
