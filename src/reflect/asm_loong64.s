// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"
#include "funcdata.h"

#define	REGCTXT	R29

// makeFuncStub is the code half of the function returned by MakeFunc.
// See the comment on the declaration of makeFuncStub in makefunc.go
// for more details.
// No arg size here, runtime pulls arg map out of the func value.
TEXT ·makeFuncStub(SB),(NOSPLIT|WRAPPER),$40
	NO_LOCAL_POINTERS
	MOVV	REGCTXT, 8(R3)
	MOVV	$argframe+0(FP), R19
	MOVV	R19, 16(R3)
	MOVB	R0, 40(R3)
	ADDV	$40, R3, R19
	MOVV	R19, 24(R3)
	MOVV	R0, 32(R3)
	JAL	·callReflect(SB)
	RET

// methodValueCall is the code half of the function returned by makeMethodValue.
// See the comment on the declaration of methodValueCall in makefunc.go
// for more details.
// No arg size here; runtime pulls arg map out of the func value.
TEXT ·methodValueCall(SB),(NOSPLIT|WRAPPER),$40
	NO_LOCAL_POINTERS
	MOVV	REGCTXT, 8(R3)
	MOVV	$argframe+0(FP), R19
	MOVV	R19, 16(R3)
	MOVB	R0, 40(R3)
	ADDV	$40, R3, R19
	MOVV	R19, 24(R3)
	MOVV	R0, 32(R3)
	JAL	·callMethod(SB)
	RET
