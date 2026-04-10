// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func r28DirtyLoop(tid, pid int, sig, iters int32)
TEXT ·r28DirtyLoop(SB), NOSPLIT, $16-24
	MOVD R19, 16(RSP)
	MOVD g, R19                // save g
	MOVD $0xDEADBEEF, g        // clobber R28
loop:
	// tgkill(pid, tid, sig)
	MOVD pid+8(FP), R0
	MOVD tid+0(FP), R1
	MOVW sig+16(FP), R2
	MOVD $131, R8
	SVC $0
	WORD $0xD5033FDF            // ISB

	MOVW iters+20(FP), R7
	SUBW $1, R7
	MOVW R7, iters+20(FP)

	WORD $0xD5033FDF            // ISB

	CBNZW R7, loop
	MOVD R19, g                // restore g
	MOVD 16(RSP), R19
	RET
