// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.cgobindings && race && !cgo && linux && amd64 && !amd64.v3

#include "textflag.h"

TEXT __popcountdi2(SB), NOSPLIT|NOFRAME, $0-0
	MOVQ	DI, AX
	SHRQ	$1, AX
	MOVQ	$0x5555555555555555, CX
	ANDQ	CX, AX
	SUBQ	AX, DI
	MOVQ	DI, AX
	SHRQ	$2, AX
	MOVQ	$0x3333333333333333, CX
	ANDQ	CX, AX
	ANDQ	CX, DI
	ADDQ	DI, AX
	MOVQ	AX, CX
	SHRQ	$4, CX
	ADDQ	CX, AX
	MOVQ	$0x0f0f0f0f0f0f0f0f, CX
	ANDQ	CX, AX
	MOVQ	$0x0101010101010101, CX
	IMULQ	CX, AX
	SHRQ	$56, AX
	RET
