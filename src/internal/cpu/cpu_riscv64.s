// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// Read the vector register length in bytes from the vlenb CSR.
// May only be called when the vector extension is present.
// func readVLENB() uint
TEXT ·readVLENB(SB), NOSPLIT|NOFRAME, $0-8
	CSRR	VLENB, X10
	MOV	X10, ret+0(FP)
	RET
