// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"
#include "funcdata.h"

// makeFuncStub is the code half of the function returned by MakeFunc.
// See the comment on the declaration of makeFuncStub in makefunc.go
// for more details.
// No arg size here; runtime pulls arg map out of the func value.
// makeFuncStub must be ABIInternal because it is placed directly
// in function values.
TEXT ·makeFuncStub<ABIInternal>(SB),(NOSPLIT|WRAPPER),$32
	NO_LOCAL_POINTERS
	MOVQ	DX, 0(SP)
	LEAQ	argframe+0(FP), CX
	MOVQ	CX, 8(SP)
	MOVB	$0, 24(SP)
	LEAQ	24(SP), AX
	MOVQ	AX, 16(SP)
	CALL	·callReflect<ABIInternal>(SB)
	RET

// methodValueCall is the code half of the function returned by makeMethodValue.
// See the comment on the declaration of methodValueCall in makefunc.go
// for more details.
// No arg size here; runtime pulls arg map out of the func value.
// methodValueCall must be ABIInternal because it is placed directly
// in function values.
TEXT ·methodValueCall<ABIInternal>(SB),(NOSPLIT|WRAPPER),$32
	NO_LOCAL_POINTERS
	MOVQ	DX, 0(SP)
	LEAQ	argframe+0(FP), CX
	MOVQ	CX, 8(SP)
	MOVB	$0, 24(SP)
	LEAQ	24(SP), AX
	MOVQ	AX, 16(SP)
	CALL	·callMethod<ABIInternal>(SB)
	RET
