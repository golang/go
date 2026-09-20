// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build gc

#include "textflag.h"

// Read the vector register length in bytes from the vlenb CSR.
// May only be called when the vector extension is present.
// func readVLENB() uint
TEXT ·readVLENB(SB), NOSPLIT|NOFRAME, $0-8
	// Go 1.25's assembler does not recognize VLENB, so use its raw CSRR encoding.
	// Replace WORD with CSRR VLENB, X10 once go.mod requires Go 1.27.
	WORD	$0xc2202573
	MOV	X10, ret+0(FP)
	RET
