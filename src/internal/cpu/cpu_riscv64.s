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

// func getGORISCV64level() int32
TEXT ·getGORISCV64level(SB), NOSPLIT|NOFRAME, $0-4
#ifdef GORISCV64_rva20u64
	MOV	$20, X10
#endif
#ifdef GORISCV64_rva22u64
	MOV	$22, X10
#endif
#ifdef GORISCV64_rva23u64
	MOV	$23, X10
#endif
	MOVW	X10, ret+0(FP)
	RET
