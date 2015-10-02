// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

/*
 * void crosscall2(void (*fn)(void*, int32), void*, int32)
 * Save registers and call fn with two arguments.
 */
TEXT crosscall2(SB),NOSPLIT,$-8
	/*
	 * We still need to save all callee save register as before, and then
	 *  push 2 args for fn (R1 and R2).
	 * Also note that at procedure entry in gc world, 8(RSP) will be the
	 *  first arg.
	 * TODO(minux): use LDP/STP here if it matters.
	 */
	SUB	$128, RSP
	MOVD	R1, (8*1)(RSP)
	MOVD	R2, (8*2)(RSP)
	MOVD	R19, (8*3)(RSP)
	MOVD	R20, (8*4)(RSP)
	MOVD	R21, (8*5)(RSP)
	MOVD	R22, (8*6)(RSP)
	MOVD	R23, (8*7)(RSP)
	MOVD	R24, (8*8)(RSP)
	MOVD	R25, (8*9)(RSP)
	MOVD	R26, (8*10)(RSP)
	MOVD	R27, (8*11)(RSP)
	MOVD	g, (8*12)(RSP)
	MOVD	R29, (8*13)(RSP)
	MOVD	R30, (8*14)(RSP)

	MOVD	R0, R19

	// Initialize Go ABI environment
	BL      runtime·reginit(SB)
	BL	runtime·load_g(SB)
	BL	(R19)

	MOVD	(8*1)(RSP), R1
	MOVD	(8*2)(RSP), R2
	MOVD	(8*3)(RSP), R19
	MOVD	(8*4)(RSP), R20
	MOVD	(8*5)(RSP), R21
	MOVD	(8*6)(RSP), R22
	MOVD	(8*7)(RSP), R23
	MOVD	(8*8)(RSP), R24
	MOVD	(8*9)(RSP), R25
	MOVD	(8*10)(RSP), R26
	MOVD	(8*11)(RSP), R27
	MOVD	(8*12)(RSP), g
	MOVD	(8*13)(RSP), R29
	MOVD	(8*14)(RSP), R30
	ADD	$128, RSP
	RET
