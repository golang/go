// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"
#include "funcdata.h"

#define	REGCTXT	R29

// The frames of each of the two functions below contain two locals, at offsets
// that are known to the runtime.
//
// The first local is a bool called retValid with a whole pointer-word reserved
// for it on the stack. The purpose of this word is so that the runtime knows
// whether the stack-allocated return space contains valid values for stack
// scanning.
//
// The second local is an abi.RegArgs value whose offset is also known to the
// runtime, so that a stack map for it can be constructed, since it contains
// pointers visible to the GC.
#define LOCAL_RETVALID 40
#define LOCAL_REGARGS 48

// The frame size of the functions below is
// 32 (args of callReflect) + 8 (bool + padding) + 392 (abi.RegArgs) = 432.

// makeFuncStub is the code half of the function returned by MakeFunc.
// See the comment on the declaration of makeFuncStub in makefunc.go
// for more details.
// No arg size here, runtime pulls arg map out of the func value.
TEXT ·makeFuncStub(SB),(NOSPLIT|WRAPPER),$432
	NO_LOCAL_POINTERS
	ADDV	$LOCAL_REGARGS, R3, R25 // spillArgs using R25
	JAL	runtime·spillArgs(SB)
	MOVV	REGCTXT, 32(R3) // save REGCTXT > args of moveMakeFuncArgPtrs < LOCAL_REGARGS

	MOVV	REGCTXT, R4
	MOVV	R25, R5
	JAL	·moveMakeFuncArgPtrs<ABIInternal>(SB)
	MOVV	32(R3), REGCTXT // restore REGCTXT

	MOVV	REGCTXT, 8(R3)
	MOVV	$argframe+0(FP), R20
	MOVV	R20, 16(R3)
	MOVV	R0, LOCAL_RETVALID(R3)
	ADDV	$LOCAL_RETVALID, R3, R20
	MOVV	R20, 24(R3)
	ADDV	$LOCAL_REGARGS, R3, R20
	MOVV	R20, 32(R3)
	JAL	·callReflect(SB)
	ADDV	$LOCAL_REGARGS, R3, R25	//unspillArgs using R25
	JAL	runtime·unspillArgs(SB)
	RET

// methodValueCall is the code half of the function returned by makeMethodValue.
// See the comment on the declaration of methodValueCall in makefunc.go
// for more details.
// No arg size here; runtime pulls arg map out of the func value.
TEXT ·methodValueCall(SB),(NOSPLIT|WRAPPER),$432
	NO_LOCAL_POINTERS
	ADDV	$LOCAL_REGARGS, R3, R25 // spillArgs using R25
	JAL	runtime·spillArgs(SB)
	MOVV	REGCTXT, 32(R3) // save REGCTXT > args of moveMakeFuncArgPtrs < LOCAL_REGARGS
	MOVV	REGCTXT, R4
	MOVV	R25, R5
	JAL	·moveMakeFuncArgPtrs<ABIInternal>(SB)
	MOVV	32(R3), REGCTXT // restore REGCTXT
	MOVV	REGCTXT, 8(R3)
	MOVV	$argframe+0(FP), R20
	MOVV	R20, 16(R3)
	MOVB	R0, LOCAL_RETVALID(R3)
	ADDV	$LOCAL_RETVALID, R3, R20
	MOVV	R20, 24(R3)
	ADDV	$LOCAL_REGARGS, R3, R20
	MOVV	R20, 32(R3) // frame size to 32+SP as callreflect args)
	JAL	·callMethod(SB)
	ADDV	$LOCAL_REGARGS, R3, R25 // unspillArgs using R25
	JAL	runtime·unspillArgs(SB)
	RET
