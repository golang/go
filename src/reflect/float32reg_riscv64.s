// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// riscv64 allows 32-bit floats to live in the bottom
// part of the register, it expects them to be NaN-boxed.
// These functions are needed to ensure correct conversions
// on riscv64.

// Convert float32->uint64
TEXT ·archFloat32ToReg(SB),NOSPLIT,$0-16
	MOVF	val+0(FP), F1
	MOVD	F1, ret+8(FP)
	RET

// Convert uint64->float32
TEXT ·archFloat32FromReg(SB),NOSPLIT,$0-12
	// Normally a float64->float32 conversion
	// would need rounding, but riscv64 store valid
	// float32 in the lower 32 bits, thus we only need to
	// unboxed the NaN-box by store a float32.
	MOVD	reg+0(FP), F1
	MOVF	F1, ret+8(FP)
	RET

