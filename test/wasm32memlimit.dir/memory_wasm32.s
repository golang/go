// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT ·currentMemory(SB), NOSPLIT, $0-4
	Get SP
	CurrentMemory
	I32Store ret+0(FP)
	RET

TEXT ·growMemory(SB), NOSPLIT, $0-8
	Get SP
	I32Load pages+0(FP)
	GrowMemory
	I32Store ret+4(FP)
	RET
