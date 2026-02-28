// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func get_cpucfg(reg uint32) uint32
TEXT ·get_cpucfg(SB), NOSPLIT|NOFRAME, $0-12
	MOVW	reg+0(FP), R5
	CPUCFG	R5, R4
	MOVW	R4, ret+8(FP)
	RET
