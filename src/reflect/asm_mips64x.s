// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build mips64 mips64le

#include "textflag.h"
#include "funcdata.h"

#define	REGCTXT	R22

// makeFuncStub is the code half of the function returned by MakeFunc.
// See the comment on the declaration of makeFuncStub in makefunc.go
// for more details.
// No arg size here, runtime pulls arg map out of the func value.
TEXT ·makeFuncStub(SB),(NOSPLIT|WRAPPER),$16
	NO_LOCAL_POINTERS
	MOVV	REGCTXT, 8(R29)
	MOVV	$argframe+0(FP), R1
	MOVV	R1, 16(R29)
	JAL	·callReflect(SB)
	RET

// methodValueCall is the code half of the function returned by makeMethodValue.
// See the comment on the declaration of methodValueCall in makefunc.go
// for more details.
// No arg size here; runtime pulls arg map out of the func value.
TEXT ·methodValueCall(SB),(NOSPLIT|WRAPPER),$16
	NO_LOCAL_POINTERS
	MOVV	REGCTXT, 8(R29)
	MOVV	$argframe+0(FP), R1
	MOVV	R1, 16(R29)
	JAL	·callMethod(SB)
	RET
