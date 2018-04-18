// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

TEXT errors(SB),$0
	MOVL	foo<>(SB)(AX), AX	// ERROR "invalid instruction"
	MOVL	(AX)(SP*1), AX		// ERROR "invalid instruction"
	EXTRACTPS $4, X2, (BX)          // ERROR "invalid instruction"
	EXTRACTPS $-1, X2, (BX)         // ERROR "invalid instruction"
	// VSIB addressing does not permit non-vector (X/Y)
	// scaled index register.
	VPGATHERDQ X12,(R13)(AX*2), X11 // ERROR "invalid instruction"
	VPGATHERDQ X2, 664(BX*1), X1    // ERROR "invalid instruction"
	VPGATHERDQ Y2, (BP)(AX*2), Y1   // ERROR "invalid instruction"
	VPGATHERDQ Y5, 664(DX*8), Y6    // ERROR "invalid instruction"
	VPGATHERDQ Y5, (DX), Y0         // ERROR "invalid instruction"
	// VM/X rejects Y index register.
	VPGATHERDQ Y5, 664(Y14*8), Y6   // ERROR "invalid instruction"
	VPGATHERQQ X2, (BP)(Y7*2), X1   // ERROR "invalid instruction"
	// VM/Y rejects X index register.
	VPGATHERQQ Y2, (BP)(X7*2), Y1   // ERROR "invalid instruction"
	VPGATHERDD Y5, -8(X14*8), Y6    // ERROR "invalid instruction"
	// No VSIB for legacy instructions.
	MOVL (AX)(X0*1), AX             // ERROR "invalid instruction"
	MOVL (AX)(Y0*1), AX             // ERROR "invalid instruction"
	// AVX2GATHER mask/index/dest #UD cases.
	VPGATHERQQ Y2, (BP)(X2*2), Y2   // ERROR "mask, index, and destination registers should be distinct"
	VPGATHERQQ Y2, (BP)(X2*2), Y7   // ERROR "mask, index, and destination registers should be distinct"
	VPGATHERQQ Y2, (BP)(X7*2), Y2   // ERROR "mask, index, and destination registers should be distinct"
	VPGATHERQQ Y7, (BP)(X2*2), Y2   // ERROR "mask, index, and destination registers should be distinct"
	VPGATHERDQ X2, 664(X2*8), X2    // ERROR "mask, index, and destination registers should be distinct"
	VPGATHERDQ X2, 664(X2*8), X7    // ERROR "mask, index, and destination registers should be distinct"
	VPGATHERDQ X2, 664(X7*8), X2    // ERROR "mask, index, and destination registers should be distinct"
	VPGATHERDQ X7, 664(X2*8), X2    // ERROR "mask, index, and destination registers should be distinct"
	// Non-X0 for Yxr0 should produce an error
	BLENDVPD X1, (BX), X2           // ERROR "invalid instruction"
	// Check offset overflow. Must fit in int32.
	MOVQ 2147483647+1(AX), AX       // ERROR "offset too large"
	MOVQ 3395469782(R10), R8        // ERROR "offset too large"
	LEAQ 3395469782(AX), AX         // ERROR "offset too large"
	ADDQ 3395469782(AX), AX         // ERROR "offset too large"
	ADDL 3395469782(AX), AX         // ERROR "offset too large"
	ADDW 3395469782(AX), AX         // ERROR "offset too large"
	LEAQ 433954697820(AX), AX       // ERROR "offset too large"
	ADDQ 433954697820(AX), AX       // ERROR "offset too large"
	ADDL 433954697820(AX), AX       // ERROR "offset too large"
	ADDW 433954697820(AX), AX       // ERROR "offset too large"
	// Pseudo-registers should not be used as scaled index.
	CALL (AX)(PC*1)                 // ERROR "invalid instruction"
	CALL (AX)(SB*1)                 // ERROR "invalid instruction"
	CALL (AX)(FP*1)                 // ERROR "invalid instruction"
	RET
