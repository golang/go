// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

TEXT errors(SB),$0
	VSHUF4IV	$16, V1, V2	// ERROR "operand out of range 0 to 15"
	XVSHUF4IV	$16, X1, X2	// ERROR "operand out of range 0 to 15"
	VMOVQ		V1.B[0], 128(R4)	// ERROR "offset out of range [-128, 127]"
	VMOVQ		V1.B[0], -129(R4)	// ERROR "offset out of range [-128, 127]"
	VMOVQ		V1.H[0], 256(R4)	// ERROR "offset out of range [-256, 254]"
	VMOVQ		V1.H[0], -258(R4)	// ERROR "offset out of range [-256, 254]"
	VMOVQ		V1.W[0], 512(R4)	// ERROR "offset out of range [-512, 508]"
	VMOVQ		V1.W[0], -516(R4)	// ERROR "offset out of range [-512, 508]"
	VMOVQ		V1.V[0], 1024(R4)	// ERROR "offset out of range [-1024, 1016]"
	VMOVQ		V1.V[0], -1032(R4)	// ERROR "offset out of range [-1024, 1016]"
	XVMOVQ		X1.V[0], 1024(R4)	// ERROR "offset out of range [-1024, 1016]"
	XVMOVQ		X1.V[0], -1032(R4)	// ERROR "offset out of range [-1024, 1016]"
	ADDV16		$1, R4, R5	// ERROR "the constant must be a multiple of 65536."
	ADDV16		$65535, R4, R5	// ERROR "the constant must be a multiple of 65536."
	SC		R4, 1(R5)	// ERROR "offset must be a multiple of 4."
	SCW		R4, 1(R5)	// ERROR "offset must be a multiple of 4."
	SCV		R4, 1(R5)	// ERROR "offset must be a multiple of 4."
	LL		1(R5), R4	// ERROR "offset must be a multiple of 4."
	LLW		1(R5), R4	// ERROR "offset must be a multiple of 4."
	LLV		1(R5), R4	// ERROR "offset must be a multiple of 4."
