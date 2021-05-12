// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"
#include "funcdata.h"

// makeFuncStub is the code half of the function returned by MakeFunc.
// See the comment on the declaration of makeFuncStub in makefunc.go
// for more details.
// No arg size here, runtime pulls arg map out of the func value.
TEXT ·makeFuncStub(SB),(NOSPLIT|WRAPPER),$40
	NO_LOCAL_POINTERS
	MOV	CTXT, 8(SP)
	MOV	$argframe+0(FP), T0
	MOV	T0, 16(SP)
	ADD	$40, SP, T1
	MOV	T1, 24(SP)
	MOV	ZERO, 32(SP)
	MOVB	ZERO, 40(SP)
	CALL	·callReflect(SB)
	RET

// methodValueCall is the code half of the function returned by makeMethodValue.
// See the comment on the declaration of methodValueCall in makefunc.go
// for more details.
// No arg size here; runtime pulls arg map out of the func value.
TEXT ·methodValueCall(SB),(NOSPLIT|WRAPPER),$40
	NO_LOCAL_POINTERS
	MOV	CTXT, 8(SP)
	MOV	$argframe+0(FP), T0
	MOV	T0, 16(SP)
	ADD	$40, SP, T1
	MOV	T1, 24(SP)
	MOV	ZERO, 32(SP)
	MOVB	ZERO, 40(SP)
	CALL	·callMethod(SB)
	RET
