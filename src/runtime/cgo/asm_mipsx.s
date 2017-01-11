// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build mips mipsle

#include "textflag.h"

/*
 * void crosscall2(void (*fn)(void*, int32, uintptr), void*, int32, uintptr)
 * Save registers and call fn with two arguments.
 */
TEXT crosscall2(SB),NOSPLIT,$-4
	/*
	 * We still need to save all callee save register as before, and then
	 *  push 3 args for fn (R5, R6, R7).
	 * Also note that at procedure entry in gc world, 4(R29) will be the
	 *  first arg.
	 */

	// Space for 9 caller-saved GPR + LR + 6 caller-saved FPR.
	// O32 ABI allows us to smash 16 bytes argument area of caller frame.
	SUBU	$(4*14+8*6-16), R29
	MOVW	R5, (4*1)(R29)
	MOVW	R6, (4*2)(R29)
	MOVW	R7, (4*3)(R29)
	MOVW	R16, (4*4)(R29)
	MOVW	R17, (4*5)(R29)
	MOVW	R18, (4*6)(R29)
	MOVW	R19, (4*7)(R29)
	MOVW	R20, (4*8)(R29)
	MOVW	R21, (4*9)(R29)
	MOVW	R22, (4*10)(R29)
	MOVW	R23, (4*11)(R29)
	MOVW	g, (4*12)(R29)
	MOVW	R31, (4*13)(R29)

	MOVD	F20, (4*14)(R29)
	MOVD	F22, (4*14+8*1)(R29)
	MOVD	F24, (4*14+8*2)(R29)
	MOVD	F26, (4*14+8*3)(R29)
	MOVD	F28, (4*14+8*4)(R29)
	MOVD	F30, (4*14+8*5)(R29)

	JAL	runtime·load_g(SB)
	JAL	(R4)

	MOVW	(4*4)(R29), R16
	MOVW	(4*5)(R29), R17
	MOVW	(4*6)(R29), R18
	MOVW	(4*7)(R29), R19
	MOVW	(4*8)(R29), R20
	MOVW	(4*9)(R29), R21
	MOVW	(4*10)(R29), R22
	MOVW	(4*11)(R29), R23
	MOVW	(4*12)(R29), g
	MOVW	(4*13)(R29), R31

	MOVD	(4*14)(R29), F20
	MOVD	(4*14+8*1)(R29), F22
	MOVD	(4*14+8*2)(R29), F24
	MOVD	(4*14+8*3)(R29), F26
	MOVD	(4*14+8*4)(R29), F28
	MOVD	(4*14+8*5)(R29), F30

	ADDU	$(4*14+8*6-16), R29
	RET
