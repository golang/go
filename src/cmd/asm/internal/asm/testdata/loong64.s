// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "../../../../../runtime/textflag.h"
// TODO: cover more instruction

TEXT foo(SB),DUPOK|NOSPLIT,$0
	JAL	1(PC)		// CALL 1(PC)		// 00040054
	JAL	(R4)		// CALL (R4)		// 8100004c
	JAL	0(R4), R5	// CALL (R4), R5	// 8500004c
	JAL	8(R4), R6	// CALL 8(R4), R6	// 8608004c
	JAL	131069(R4), R6	// CALL 131069(R4), R6	// 86fcff4d
	JAL	-131070(R4), R6	// CALL -131070(R4), R6	// 8600004e

	// relocation in play so the assembled offset should be 0
	JAL	foo(SB)		// CALL foo(SB)		// 00000054
	JMP     foo(SB)		// JMP  foo(SB)		// 00000050
