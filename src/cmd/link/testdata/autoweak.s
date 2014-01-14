// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test of go.weak symbols.

TEXT start(SB),7,$0
	MOVQ $autotab(SB),AX
	MOVQ $autoptr(SB),AX
	RET

// go.weak.sym should resolve to sym, because sym is in the binary.
DATA autotab+0(SB)/8, $go·weak·sym(SB)
DATA autotab+8(SB)/8, $sym(SB)

// go.weak.missingsym should resolve to 0, because missingsym is not in the binary.
DATA autotab+16(SB)/8, $go·weak·missingsym(SB)
DATA autotab+24(SB)/8, $0

// go.weak.deadsym should resolve to 0, because deadsym is discarded during dead code removal
DATA autotab+32(SB)/8, $go·weak·deadsym(SB)
DATA autotab+40(SB)/8, $0

GLOBL autotab(SB), $48

GLOBL sym(SB), $1

GLOBL deadsym(SB), $1

GLOBL autoptr(SB), $0
