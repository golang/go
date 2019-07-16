// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build mips mipsle

#include "textflag.h"

TEXT ·spinLock(SB),NOSPLIT,$0-4
	MOVW	state+0(FP), R1
	MOVW	$1, R2
	SYNC
try_lock:
	MOVW	R2, R3
check_again:
	LL	(R1), R4
	BNE	R4, check_again
	SC	R3, (R1)
	BEQ	R3, try_lock
	SYNC
	RET

TEXT ·spinUnlock(SB),NOSPLIT,$0-4
	MOVW	state+0(FP), R1
	SYNC
	MOVW	R0, (R1)
	SYNC
	RET
