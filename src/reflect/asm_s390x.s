// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"
#include "funcdata.h"

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
// 32 (args of callReflect/callMethod) + 8 (bool + padding) + 264 (abi.RegArgs) = 304.

// makeFuncStub is the code half of the function returned by MakeFunc.
// See the comment on the declaration of makeFuncStub in makefunc.go
// for more details.
// No arg size here, runtime pulls arg map out of the func value.
TEXT ·makeFuncStub(SB),(NOSPLIT|WRAPPER),$304
	NO_LOCAL_POINTERS
	ADD	$LOCAL_REGARGS, R15, R10 // spillArgs using R10
	BL	runtime·spillArgs(SB)
	MOVD	R12, 32(R15) // save context reg R12 > args of moveMakeFuncArgPtrs < LOCAL_REGARGS
#ifdef GOEXPERIMENT_regabiargs
	MOVD	R12, R2
	MOVD	R10, R3
#else
	MOVD	R12, 8(R15)
	MOVD	R10, 16(R15)
#endif
	BL	·moveMakeFuncArgPtrs<ABIInternal>(SB)
	MOVD	32(R15), R12 // restore context reg R12
	MOVD	R12, 8(R15)
	MOVD	$argframe+0(FP), R1
	MOVD	R1, 16(R15)
	MOVB	$0, LOCAL_RETVALID(R15)
	ADD	$LOCAL_RETVALID, R15, R1
	MOVD	R1, 24(R15)
	ADD	$LOCAL_REGARGS, R15, R1
	MOVD	R1, 32(R15)
	BL	·callReflect(SB)
	ADD	$LOCAL_REGARGS, R15, R10 // unspillArgs using R10
	BL	runtime·unspillArgs(SB)
	RET

// methodValueCall is the code half of the function returned by makeMethodValue.
// See the comment on the declaration of methodValueCall in makefunc.go
// for more details.
// No arg size here; runtime pulls arg map out of the func value.
TEXT ·methodValueCall(SB),(NOSPLIT|WRAPPER),$304
	NO_LOCAL_POINTERS
	ADD	$LOCAL_REGARGS, R15, R10 // spillArgs using R10
	BL	runtime·spillArgs(SB)
	MOVD	R12, 32(R15) // save context reg R12 > args of moveMakeFuncArgPtrs < LOCAL_REGARGS
#ifdef GOEXPERIMENT_regabiargs
	MOVD	R12, R2
	MOVD	R10, R3
#else
	MOVD	R12, 8(R15)
	MOVD	R10, 16(R15)
#endif
	BL	·moveMakeFuncArgPtrs<ABIInternal>(SB)
	MOVD	32(R15), R12 // restore context reg R12
	MOVD	R12, 8(R15)
	MOVD	$argframe+0(FP), R1
	MOVD	R1, 16(R15)
	MOVB	$0, LOCAL_RETVALID(R15)
	ADD	$LOCAL_RETVALID, R15, R1
	MOVD	R1, 24(R15)
	ADD	$LOCAL_REGARGS, R15, R1
	MOVD	R1, 32(R15)
	BL	·callMethod(SB)
	ADD	$LOCAL_REGARGS, R15, R10 // unspillArgs using R10
	BL	runtime·unspillArgs(SB)
	RET
