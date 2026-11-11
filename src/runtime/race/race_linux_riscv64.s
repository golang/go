// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.cgobindings && race && !cgo && linux && riscv64

#include "textflag.h"

TEXT __popcountdi2(SB), NOSPLIT|NOFRAME, $0-0
	MOV	X10, X11
	SRL	$1, X11, X11
	MOV	$0x5555555555555555, X12
	AND	X12, X11, X11
	SUB	X11, X10, X10
	SRL	$2, X10, X11
	MOV	$0x3333333333333333, X12
	AND	X12, X11, X11
	AND	X12, X10, X10
	ADD	X11, X10, X10
	SRL	$4, X10, X11
	ADD	X11, X10, X10
	MOV	$0x0f0f0f0f0f0f0f0f, X12
	AND	X12, X10, X10
	SRL	$8, X10, X11
	ADD	X11, X10, X10
	SRL	$16, X10, X11
	ADD	X11, X10, X10
	SRL	$32, X10, X11
	ADD	X11, X10, X10
	AND	$0x7f, X10, X10
	RET

TEXT __clzdi2(SB), NOSPLIT|NOFRAME, $0-0
	MOV	$64, X11
	BEQ	X10, X0, clz_done
	MOV	$0, X11
	SRL	$32, X10, X12
	BNE	X12, X0, clz_16
	ADD	$32, X11, X11
	SLL	$32, X10, X10
clz_16:
	SRL	$48, X10, X12
	BNE	X12, X0, clz_8
	ADD	$16, X11, X11
	SLL	$16, X10, X10
clz_8:
	SRL	$56, X10, X12
	BNE	X12, X0, clz_4
	ADD	$8, X11, X11
	SLL	$8, X10, X10
clz_4:
	SRL	$60, X10, X12
	BNE	X12, X0, clz_2
	ADD	$4, X11, X11
	SLL	$4, X10, X10
clz_2:
	SRL	$62, X10, X12
	BNE	X12, X0, clz_1
	ADD	$2, X11, X11
	SLL	$2, X10, X10
clz_1:
	SRL	$63, X10, X12
	BNE	X12, X0, clz_done
	ADD	$1, X11, X11
clz_done:
	MOV	X11, X10
	RET
