// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ppc64 ppc64le

#include "textflag.h"
#include "funcdata.h"

// makeFuncStub is the code half of the function returned by MakeFunc.
// See the comment on the declaration of makeFuncStub in makefunc.go
// for more details.
// No arg size here, runtime pulls arg map out of the func value.
TEXT ·makeFuncStub(SB),(NOSPLIT|WRAPPER),$16
	NO_LOCAL_POINTERS
	MOVD	R11, 8(R1)
	MOVD	$argframe+0(FP), R3
	MOVD	R3, 16(R1)
	BL	·callReflect(SB)
	RETURN

// methodValueCall is the code half of the function returned by makeMethodValue.
// See the comment on the declaration of methodValueCall in makefunc.go
// for more details.
// No arg size here; runtime pulls arg map out of the func value.
TEXT ·methodValueCall(SB),(NOSPLIT|WRAPPER),$16
	NO_LOCAL_POINTERS
	MOVD	R11, 8(R1)
	MOVD	$argframe+0(FP), R3
	MOVD	R3, 16(R1)
	BL	·callMethod(SB)
	RETURN
