// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// makeFuncStub is the code half of the function returned by MakeFunc.
// See the comment on the declaration of makeFuncStub in makefunc.go
// for more details.
// No argsize here, gc generates argsize info at call site.
TEXT ·makeFuncStub(SB),(NOSPLIT|WRAPPER),$8
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
	MOVL	DX, 0(SP)
	LEAL	argframe+0(FP), CX
	MOVL	CX, 4(SP)
	CALL	·callMethod(SB)
	RET

// Stubs to give reflect package access to runtime services
// TODO: should probably be done another way.
TEXT ·makemap(SB),NOSPLIT,$0-0
	JMP	runtime·reflect_makemap(SB)
TEXT ·mapaccess(SB),NOSPLIT,$0-0
	JMP	runtime·reflect_mapaccess(SB)
TEXT ·mapassign(SB),NOSPLIT,$0-0
	JMP	runtime·reflect_mapassign(SB)
TEXT ·mapdelete(SB),NOSPLIT,$0-0
	JMP	runtime·reflect_mapdelete(SB)
TEXT ·mapiterinit(SB),NOSPLIT,$0-0
	JMP	runtime·reflect_mapiterinit(SB)
TEXT ·mapiterkey(SB),NOSPLIT,$0-0
	JMP	runtime·reflect_mapiterkey(SB)
TEXT ·mapiternext(SB),NOSPLIT,$0-0
	JMP	runtime·reflect_mapiternext(SB)
TEXT ·maplen(SB),NOSPLIT,$0-0
	JMP	runtime·reflect_maplen(SB)
TEXT ·ismapkey(SB),NOSPLIT,$0-0
	JMP	runtime·reflect_ismapkey(SB)
TEXT ·ifaceE2I(SB),NOSPLIT,$0-0
	JMP	runtime·reflect_ifaceE2I(SB)
TEXT ·unsafe_New(SB),NOSPLIT,$0-0
	JMP	runtime·newobject(SB)
TEXT ·unsafe_NewArray(SB),NOSPLIT,$0-0
	JMP	runtime·newarray(SB)
