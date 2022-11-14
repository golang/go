// Copyright 2019 The Go Authors. All rights reserved.
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
// 32 (args of callReflect/callMethod) + (8 bool with padding) + 392 (abi.RegArgs) = 432.

// makeFuncStub is the code half of the function returned by MakeFunc.
// See the comment on the declaration of makeFuncStub in makefunc.go
// for more details.
// No arg size here, runtime pulls arg map out of the func value.
TEXT ·makeFuncStub(SB),(NOSPLIT|WRAPPER),$432
	NO_LOCAL_POINTERS
	ADD	$LOCAL_REGARGS, SP, X25 // spillArgs using X25
	CALL	runtime·spillArgs(SB)
	MOV	CTXT, 32(SP) // save CTXT > args of moveMakeFuncArgPtrs < LOCAL_REGARGS
	MOV	CTXT, 8(SP)
	MOV	X25, 16(SP)
	CALL	·moveMakeFuncArgPtrs(SB)
	MOV	32(SP), CTXT // restore CTXT

	MOV	CTXT, 8(SP)
	MOV	$argframe+0(FP), T0
	MOV	T0, 16(SP)
	MOV	ZERO, LOCAL_RETVALID(SP)
	ADD	$LOCAL_RETVALID, SP, T1
	MOV	T1, 24(SP)
	ADD	$LOCAL_REGARGS, SP, T1
	MOV	T1, 32(SP)
	CALL	·callReflect(SB)
	ADD	$LOCAL_REGARGS, SP, X25 // unspillArgs using X25
	CALL	runtime·unspillArgs(SB)
	RET

// methodValueCall is the code half of the function returned by makeMethodValue.
// See the comment on the declaration of methodValueCall in makefunc.go
// for more details.
// No arg size here; runtime pulls arg map out of the func value.
TEXT ·methodValueCall(SB),(NOSPLIT|WRAPPER),$432
	NO_LOCAL_POINTERS
	ADD	$LOCAL_REGARGS, SP, X25 // spillArgs using X25
	CALL	runtime·spillArgs(SB)
	MOV	CTXT, 32(SP) // save CTXT
	MOV	CTXT, 8(SP)
	MOV	X25, 16(SP)
	CALL	·moveMakeFuncArgPtrs(SB)
	MOV	32(SP), CTXT // restore CTXT
	MOV	CTXT, 8(SP)
	MOV	$argframe+0(FP), T0
	MOV	T0, 16(SP)
	MOV	ZERO, LOCAL_RETVALID(SP)
	ADD	$LOCAL_RETVALID, SP, T1
	MOV	T1, 24(SP)
	ADD	$LOCAL_REGARGS, SP, T1
	MOV	T1, 32(SP) // frame size to 32+SP as callreflect args
	CALL	·callMethod(SB)
	ADD	$LOCAL_REGARGS, SP, X25 // unspillArgs using X25
	CALL	runtime·unspillArgs(SB)
	RET
