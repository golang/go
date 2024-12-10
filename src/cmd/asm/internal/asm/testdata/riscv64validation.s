// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file is for validation errors only, i.e., errors reported by the validate function.
// Negative test cases for errors generated earlier in the assembler's preprocess stage
// should be added to riscv64error.s.  If they are added to this file, they will prevent
// the validate function from being run and TestRISCVValidation will report missing
// errors.

TEXT validation(SB),$0
	SRLI	$1, X5, F1			// ERROR "expected integer register in rd position but got non-integer register F1"
	SRLI	$1, F1, X5			// ERROR "expected integer register in rs1 position but got non-integer register F1"
	VSETVLI	$32, E16, M1, TU, MU, X12	// ERROR "must be in range [0, 31] (5 bits)"
	VSETVLI	$-1, E32, M2, TA, MA, X12	// ERROR "must be in range [0, 31] (5 bits)"
	VSETVL	X10, X11			// ERROR "expected integer register in rs1 position"
	VLE8V	(X10), X10			// ERROR "expected vector register in rd position"
	VLE8V	(V1), V3			// ERROR "expected integer register in rs1 position"
	VSE8V	X10, (X10)			// ERROR "expected vector register in rs1 position"
	VSE8V	V3, (V1)			// ERROR "expected integer register in rd position"
	VLSE8V	(X10), V3			// ERROR "expected integer register in rs2 position"
	VLSE8V	(X10), X10, X11			// ERROR "expected vector register in rd position"
	VLSE8V	(V1), X10, V3			// ERROR "expected integer register in rs1 position"
	VLSE8V	(X10), V1, V0, V3		// ERROR "expected integer register in rs2 position"
	VSSE8V	V3, (X10)			// ERROR "expected integer register in rs2 position"
	VSSE8V	X10, X11, (X10)			// ERROR "expected vector register in rd position"
	VSSE8V	V3, X11, (V1)			// ERROR "expected integer register in rs1 position"
	VSSE8V	V3, V1, V0, (X10)		// ERROR "expected integer register in rs2 position"
	VLUXEI8V (X10), V2, X11			// ERROR "expected vector register in rd position"
	VLUXEI8V (X10), V2, X11			// ERROR "expected vector register in rd position"
	VLUXEI8V (V1), V2, V3			// ERROR "expected integer register in rs1 position"
	VLUXEI8V (X10), X11, V0, V3		// ERROR "expected vector register in rs2 position"
	VSUXEI8V X10, V2, (X10)			// ERROR "expected vector register in rd position"
	VSUXEI8V V3, V2, (V1)			// ERROR "expected integer register in rs1 position"
	VSUXEI8V V3, X11, V0, (X10)		// ERROR "expected vector register in rs2 position"
	VLOXEI8V (X10), V2, X11			// ERROR "expected vector register in rd position"
	VLOXEI8V (V1), V2, V3			// ERROR "expected integer register in rs1 position"
	VLOXEI8V (X10), X11, V0, V3		// ERROR "expected vector register in rs2 position"
	VSOXEI8V X10, V2, (X10)			// ERROR "expected vector register in rd position"
	VSOXEI8V V3, V2, (V1)			// ERROR "expected integer register in rs1 position"
	VSOXEI8V V3, X11, V0, (X10)		// ERROR "expected vector register in rs2 position"
	VL1RV	(X10), X10			// ERROR "expected vector register in rd position"
	VL1RV	(V1), V3			// ERROR "expected integer register in rs1 position"
	VS1RV	X11, (X11)			// ERROR "expected vector register in rs1 position"
	VS1RV	V3, (V1)			// ERROR "expected integer register in rd position"
	RET
