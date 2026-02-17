// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build valgrind && linux

#include "textflag.h"

// Instead of using cgo and using the Valgrind macros, we just emit the special client request
// assembly ourselves. The client request mechanism is basically the same across all architectures,
// with the notable difference being the special preamble that lets Valgrind know we want to do
// a client request.
//
// The form of the VALGRIND_DO_CLIENT_REQUEST macro assembly can be found in the valgrind/valgrind.h
// header file [0].
//
// [0] https://sourceware.org/git/?p=valgrind.git;a=blob;f=include/valgrind.h.in;h=f1710924aa7372e7b7e2abfbf7366a2286e33d2d;hb=HEAD

// func valgrindClientRequest(uintptr, uintptr, uintptr, uintptr, uintptr, uintptr) (ret uintptr)
TEXT runtime·valgrindClientRequest(SB), NOSPLIT, $0-56
	// Load the address of the first of the (contiguous) arguments into AX.
	LEAQ args+0(FP), AX

	// Zero DX, since some requests may not populate it.
	XORL DX, DX

	// Emit the special preabmle.
	ROLQ $3, DI; ROLQ $13, DI
	ROLQ $61, DI; ROLQ $51, DI

	// "Execute" the client request.
	XCHGQ BX, BX

	// Copy the result out of DX.
	MOVQ DX, ret+48(FP)

	RET
