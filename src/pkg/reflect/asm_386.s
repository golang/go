// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// makeFuncStub is the code half of the function returned by MakeFunc.
// See the comment on the declaration of makeFuncStub in value.go
// for more details.
TEXT ·makeFuncStub(SB),7,$8
	MOVL	DX, 0(SP)
	LEAL	arg+0(FP), CX
	MOVL	CX, 4(SP)
	CALL	·callReflect(SB)
	RET
