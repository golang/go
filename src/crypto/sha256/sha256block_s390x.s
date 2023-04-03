// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func doBlock(dig *digest, p *byte, n int)
TEXT ·doBlock(SB), NOSPLIT|NOFRAME, $0-24
	MOVBZ  ·useAsm(SB), R4
	LMG    dig+0(FP), R1, R3            // R2 = p, R3 = n
	MOVBZ  $2, R0                       // SHA-256 function code
	CMPBEQ R4, $0, generic

loop:
	WORD $0xB93E0002 // KIMD R2
	BVS  loop        // continue if interrupted
	RET

generic:
	BR ·doBlockGeneric(SB)
