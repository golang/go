// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"
#include "funcdata.h"

// makeFuncStub is the code half of the function returned by MakeFunc.
// See the comment on the declaration of makeFuncStub in makefunc.go
// for more details.
// No arg size here; runtime pulls arg map out of the func value.
TEXT ·makeFuncStub(SB),(NOSPLIT|WRAPPER),$32
	NO_LOCAL_POINTERS

	MOVD CTXT, 0(SP)

	Get SP
	Get SP
	I64ExtendUI32
	I64Const $argframe+0(FP)
	I64Add
	I64Store $8

	MOVB $0, 24(SP)
	MOVD $24(SP), 16(SP)

	CALL ·callReflect(SB)
	RET

// methodValueCall is the code half of the function returned by makeMethodValue.
// See the comment on the declaration of methodValueCall in makefunc.go
// for more details.
// No arg size here; runtime pulls arg map out of the func value.
TEXT ·methodValueCall(SB),(NOSPLIT|WRAPPER),$32
	NO_LOCAL_POINTERS

	MOVD CTXT, 0(SP)

	Get SP
	Get SP
	I64ExtendUI32
	I64Const $argframe+0(FP)
	I64Add
	I64Store $8

	MOVB $0, 24(SP)
	MOVD $24(SP), 16(SP)

	CALL ·callMethod(SB)
	RET
