// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ppc64 || ppc64le

#include "textflag.h"
#include "funcdata.h"
#include "asm_ppc64x.h"

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

#define LOCAL_RETVALID 32+FIXED_FRAME
#define LOCAL_REGARGS 40+FIXED_FRAME

// The frame size of the functions below is
// 32 (args of callReflect) + 8 (bool + padding) + 296 (abi.RegArgs) = 336.

// makeFuncStub is the code half of the function returned by MakeFunc.
// See the comment on the declaration of makeFuncStub in makefunc.go
// for more details.
// No arg size here, runtime pulls arg map out of the func value.
TEXT ·makeFuncStub(SB),(NOSPLIT|WRAPPER),$336
	NO_LOCAL_POINTERS
	// NO_LOCAL_POINTERS is a lie. The stack map for the two locals in this
	// frame is specially handled in the runtime. See the comment above LOCAL_RETVALID.
	ADD	$LOCAL_REGARGS, R1, R20
	CALL	runtime·spillArgs(SB)
	MOVD	R11, FIXED_FRAME+32(R1)	// save R11
	MOVD	R11, FIXED_FRAME+0(R1)	// arg for moveMakeFuncArgPtrs
	MOVD	R20, FIXED_FRAME+8(R1)	// arg for local args
	CALL	·moveMakeFuncArgPtrs(SB)
	MOVD	FIXED_FRAME+32(R1), R11	// restore R11 ctxt
	MOVD	R11, FIXED_FRAME+0(R1)	// ctxt (arg0)
	MOVD	$argframe+0(FP), R3	// save arg to callArg
	MOVD	R3, FIXED_FRAME+8(R1)	// frame (arg1)
	ADD	$LOCAL_RETVALID, R1, R3 // addr of return flag
	MOVB	R0, (R3)		// clear flag
	MOVD	R3, FIXED_FRAME+16(R1)	// addr retvalid (arg2)
	ADD     $LOCAL_REGARGS, R1, R3
	MOVD	R3, FIXED_FRAME+24(R1)	// abiregargs (arg3)
	BL	·callReflect(SB)
	ADD	$LOCAL_REGARGS, R1, R20	// set address of spill area
	CALL	runtime·unspillArgs(SB)
	RET

// methodValueCall is the code half of the function returned by makeMethodValue.
// See the comment on the declaration of methodValueCall in makefunc.go
// for more details.
// No arg size here; runtime pulls arg map out of the func value.
TEXT ·methodValueCall(SB),(NOSPLIT|WRAPPER),$336
	NO_LOCAL_POINTERS
	// NO_LOCAL_POINTERS is a lie. The stack map for the two locals in this
	// frame is specially handled in the runtime. See the comment above LOCAL_RETVALID.
	ADD	$LOCAL_REGARGS, R1, R20
	CALL	runtime·spillArgs(SB)
	MOVD	R11, FIXED_FRAME+0(R1) // arg0 ctxt
	MOVD	R11, FIXED_FRAME+32(R1) // save for later
	MOVD	R20, FIXED_FRAME+8(R1) // arg1 abiregargs
	CALL	·moveMakeFuncArgPtrs(SB)
	MOVD	FIXED_FRAME+32(R1), R11 // restore ctxt
	MOVD	R11, FIXED_FRAME+0(R1) // set as arg0
	MOVD	$argframe+0(FP), R3	// frame pointer
	MOVD	R3, FIXED_FRAME+8(R1)	// set as arg1
	ADD	$LOCAL_RETVALID, R1, R3
	MOVB	$0, (R3)		// clear ret flag
	MOVD	R3, FIXED_FRAME+16(R1)	// addr of return flag
	ADD	$LOCAL_REGARGS, R1, R3	// addr of abiregargs
	MOVD	R3, FIXED_FRAME+24(R1)	// set as arg3
	BL	·callMethod(SB)
	ADD     $LOCAL_REGARGS, R1, R20
	CALL	runtime·unspillArgs(SB)
	RET
