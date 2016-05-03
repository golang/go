// Copyright 2016 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build mips64 mips64le

#include "textflag.h"

/*
 * void crosscall2(void (*fn)(void*, int32, uintptr), void*, int32, uintptr)
 * Save registers and call fn with two arguments.
 */
TEXT crosscall2(SB),NOSPLIT,$-8
	/*
	 * We still need to save all callee save register as before, and then
	 *  push 3 args for fn (R5, R6, R7).
	 * Also note that at procedure entry in gc world, 8(R29) will be the
	 *  first arg.
	 */
	ADDV	$(-8*23), R29
	MOVV	R5, (8*1)(R29)
	MOVV	R6, (8*2)(R29)
	MOVV	R7, (8*3)(R29)
	MOVV	R16, (8*4)(R29)
	MOVV	R17, (8*5)(R29)
	MOVV	R18, (8*6)(R29)
	MOVV	R19, (8*7)(R29)
	MOVV	R20, (8*8)(R29)
	MOVV	R21, (8*9)(R29)
	MOVV	R22, (8*10)(R29)
	MOVV	R23, (8*11)(R29)
	MOVV	RSB, (8*12)(R29)
	MOVV	g, (8*13)(R29)
	MOVV	R31, (8*14)(R29)
	MOVD	F24, (8*15)(R29)
	MOVD	F25, (8*16)(R29)
	MOVD	F26, (8*17)(R29)
	MOVD	F27, (8*18)(R29)
	MOVD	F28, (8*19)(R29)
	MOVD	F29, (8*20)(R29)
	MOVD	F30, (8*21)(R29)
	MOVD	F31, (8*22)(R29)

	// Initialize Go ABI environment
	// prepare SB register = PC & 0xffffffff00000000
	BGEZAL	R0, 1(PC)
	SRLV	$32, R31, RSB
	SLLV	$32, RSB
	JAL	runtime·reginit(SB)
	JAL	runtime·load_g(SB)
	JAL	(R4)

	MOVV	(8*1)(R29), R5
	MOVV	(8*2)(R29), R6
	MOVV	(8*3)(R29), R7
	MOVV	(8*4)(R29), R16
	MOVV	(8*5)(R29), R17
	MOVV	(8*6)(R29), R18
	MOVV	(8*7)(R29), R19
	MOVV	(8*8)(R29), R20
	MOVV	(8*9)(R29), R21
	MOVV	(8*10)(R29), R22
	MOVV	(8*11)(R29), R23
	MOVV	(8*12)(R29), RSB
	MOVV	(8*13)(R29), g
	MOVV	(8*14)(R29), R31
	MOVD	(8*15)(R29), F24
	MOVD	(8*16)(R29), F25
	MOVD	(8*17)(R29), F26
	MOVD	(8*18)(R29), F27
	MOVD	(8*19)(R29), F28
	MOVD	(8*20)(R29), F29
	MOVD	(8*21)(R29), F30
	MOVD	(8*22)(R29), F31
	ADDV	$(8*23), R29
	RET
