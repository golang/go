// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func block(dig *digest, p []byte)
TEXT ·block(SB), NOSPLIT|NOFRAME, $0-32
	MOVBZ  ·useAsm(SB), R4
	LMG    dig+0(FP), R1, R3            // R2 = &p[0], R3 = len(p)
	MOVBZ  $2, R0                       // SHA-256 function code
	CMPBEQ R4, $0, generic

loop:
	KIMD R0, R2      // compute intermediate message digest (KIMD)
	BVS  loop        // continue if interrupted
	RET

generic:
	BR ·blockGeneric(SB)
