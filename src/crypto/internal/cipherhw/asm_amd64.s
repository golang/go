// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build amd64,!gccgo,!appengine

#include "textflag.h"

// func hasAESNI() bool
TEXT ·hasAESNI(SB),NOSPLIT,$0
	XORQ AX, AX
	INCL AX
	CPUID
	SHRQ $25, CX
	ANDQ $1, CX
	MOVB CX, ret+0(FP)
	RET
