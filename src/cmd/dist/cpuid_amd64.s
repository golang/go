// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !gccgo

TEXT ·cpuid(SB),$0-12
	MOVL ax+8(FP), AX
	CPUID
	MOVQ info+0(FP), DI
	MOVL AX, 0(DI)
	MOVL BX, 4(DI)
	MOVL CX, 8(DI)
	MOVL DX, 12(DI)
	RET

