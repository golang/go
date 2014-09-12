// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"
#include "funcdata.h"

// makeFuncStub is the code half of the function returned by MakeFunc.
// See the comment on the declaration of makeFuncStub in makefunc.go
// for more details.
// No argsize here, gc generates argsize info at call site.
TEXT ·makeFuncStub(SB),(NOSPLIT|WRAPPER),$8
	NO_LOCAL_POINTERS
	MOVL	DX, 0(SP)
	LEAL	argframe+0(FP), CX
	MOVL	CX, 4(SP)
	CALL	·callReflect(SB)
	RET

// methodValueCall is the code half of the function returned by makeMethodValue.
// See the comment on the declaration of methodValueCall in makefunc.go
// for more details.
// No argsize here, gc generates argsize info at call site.
TEXT ·methodValueCall(SB),(NOSPLIT|WRAPPER),$8
	NO_LOCAL_POINTERS
	MOVL	DX, 0(SP)
	LEAL	argframe+0(FP), CX
	MOVL	CX, 4(SP)
	CALL	·callMethod(SB)
	RET
