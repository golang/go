// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"
#include "funcdata.h"

// makeFuncStub is the code half of the function returned by MakeFunc.
// See the comment on the declaration of makeFuncStub in makefunc.go
// for more details.
// No arg size here, runtime pulls arg map out of the func value.
TEXT ·makeFuncStub(SB),(NOSPLIT|WRAPPER),$32
	NO_LOCAL_POINTERS
	MOVD	R12, 8(R15)
	MOVD	$argframe+0(FP), R3
	MOVD	R3, 16(R15)
	MOVB	$0, 32(R15)
	ADD	$32, R15, R3
	MOVD	R3, 24(R15)
	BL	·callReflect(SB)
	RET

// methodValueCall is the code half of the function returned by makeMethodValue.
// See the comment on the declaration of methodValueCall in makefunc.go
// for more details.
// No arg size here; runtime pulls arg map out of the func value.
TEXT ·methodValueCall(SB),(NOSPLIT|WRAPPER),$32
	NO_LOCAL_POINTERS
	MOVD	R12, 8(R15)
	MOVD	$argframe+0(FP), R3
	MOVD	R3, 16(R15)
	MOVB	$0, 32(R15)
	ADD	$32, R15, R3
	MOVD	R3, 24(R15)
	BL	·callMethod(SB)
	RET
