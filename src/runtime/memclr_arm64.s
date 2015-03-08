// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// void runtime·memclr(void*, uintptr)
TEXT runtime·memclr(SB),NOSPLIT,$0-16
	MOVD	ptr+0(FP), R3
	MOVD	n+8(FP), R4
	CMP	$0, R4
	BEQ	done
	ADD	R3, R4, R4
	MOVBU.P	$0, 1(R3)
	CMP	R3, R4
	BNE	-2(PC)
done:
	RET
