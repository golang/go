// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build power64 power64le

#include "../../cmd/ld/textflag.h"

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

// Stubs to give reflect package access to runtime services
// TODO: should probably be done another way.
TEXT ·makemap(SB),NOSPLIT,$0-0
	BR	runtime·reflect_makemap(SB)
TEXT ·mapaccess(SB),NOSPLIT,$0-0
	BR	runtime·reflect_mapaccess(SB)
TEXT ·mapassign(SB),NOSPLIT,$0-0
	BR	runtime·reflect_mapassign(SB)
TEXT ·mapdelete(SB),NOSPLIT,$0-0
	BR	runtime·reflect_mapdelete(SB)
TEXT ·mapiterinit(SB),NOSPLIT,$0-0
	BR	runtime·reflect_mapiterinit(SB)
TEXT ·mapiterkey(SB),NOSPLIT,$0-0
	BR	runtime·reflect_mapiterkey(SB)
TEXT ·mapiternext(SB),NOSPLIT,$0-0
	BR	runtime·reflect_mapiternext(SB)
TEXT ·maplen(SB),NOSPLIT,$0-0
	BR	runtime·reflect_maplen(SB)
TEXT ·ismapkey(SB),NOSPLIT,$0-0
	BR	runtime·reflect_ismapkey(SB)
