// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

TEXT errors(SB),$0
	VSHUF4IV	$16, V1, V2	// ERROR "operand out of range 0 to 15"
	XVSHUF4IV	$16, X1, X2	// ERROR "operand out of range 0 to 15"
	ADDV16		$1, R4, R5	// ERROR "the constant must be a multiple of 65536."
	ADDV16		$65535, R4, R5	// ERROR "the constant must be a multiple of 65536."
	SC		R4, 1(R5)	// ERROR "offset must be a multiple of 4."
	SCV		R4, 1(R5)	// ERROR "offset must be a multiple of 4."
	LL		1(R5), R4	// ERROR "offset must be a multiple of 4."
	LLV		1(R5), R4	// ERROR "offset must be a multiple of 4."

