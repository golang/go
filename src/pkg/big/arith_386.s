// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file provides fast assembly versions for the elementary
// arithmetic operations on vectors implemented in arith.go.

TEXT big·useAsm(SB),7,$0
	MOVB $0, 4(SP)	// assembly routines disabled
	RET


// TODO(gri) Implement these routines and enable them.
TEXT big·addVV_s(SB),7,$0
TEXT big·subVV_s(SB),7,$0
TEXT big·addVW_s(SB),7,$0
TEXT big·subVW_s(SB),7,$0
TEXT big·mulAddVWW_s(SB),7,$0
TEXT big·addMulVVW_s(SB),7,$0
TEXT big·divWVW_s(SB),7,$0
	RET


// func divWWW_s(x1, x0, y Word) (q, r Word)
// TODO(gri) Implement this routine completely in Go.
//           At the moment we need this assembly version.
TEXT big·divWWW_s(SB),7,$0
	MOVL x1+0(FP), DX
	MOVL x0+4(FP), AX
	DIVL y+8(FP)
	MOVL AX, q+12(FP)
	MOVL DX, r+16(FP)
	RET
