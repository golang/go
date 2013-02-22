// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// makeFuncStub is the code half of the function returned by MakeFunc.
// See the comment on the declaration of makeFuncStub in value.go
// for more details.
TEXT ·makeFuncStub(SB),7,$16
	MOVQ	DX, 0(SP)
	LEAQ	arg+0(FP), CX
	MOVQ	CX, 8(SP)
	CALL	·callReflect(SB)
	RET
