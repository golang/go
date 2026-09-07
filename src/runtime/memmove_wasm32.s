// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// See memmove Go doc for important implementation constraints.

// func memmove(to, from unsafe.Pointer, n uintptr)
TEXT runtime·memmove(SB), NOSPLIT, $0-12
	MOVW to+0(FP), R0
	MOVW from+4(FP), R1
	MOVW n+8(FP), R2

	Get R0
	I32WrapI64
	Get R1
	I32WrapI64
	Get R2
	I32WrapI64
	MemoryCopy
	RET
