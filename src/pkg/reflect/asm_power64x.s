// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build power64 power64le

#include "textflag.h"

// makeFuncStub is the code half of the function returned by MakeFunc.
// See the comment on the declaration of makeFuncStub in makefunc.go
// for more details.
// No argsize here, gc generates argsize info at call site.
TEXT ·makeFuncStub(SB),(NOSPLIT|WRAPPER),$16
	MOVD	R11, 8(R1)
	MOVD	$argframe+0(FP), R3
	MOVD	R3, 16(R1)
	BL	·callReflect(SB)
	RETURN

// methodValueCall is the code half of the function returned by makeMethodValue.
// See the comment on the declaration of methodValueCall in makefunc.go
// for more details.
// No argsize here, gc generates argsize info at call site.
TEXT ·methodValueCall(SB),(NOSPLIT|WRAPPER),$16
	MOVD	R11, 8(R1)
	MOVD	$argframe+0(FP), R3
	MOVD	R3, 16(R1)
	BL	·callMethod(SB)
	RETURN
