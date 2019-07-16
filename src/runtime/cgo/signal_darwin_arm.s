// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// xx_cgo_panicmem is the entrypoint for SIGSEGV as intercepted via a
// mach thread port as EXC_BAD_ACCESS. As the segfault may have happened
// in C code, we first need to load_g then call xx_cgo_panicmem.
//
//	R1 - LR at moment of fault
//	R2 - PC at moment of fault
TEXT xx_cgo_panicmem(SB),NOSPLIT|NOFRAME,$0
	// If in external C code, we need to load the g register.
	BL  runtime·load_g(SB)
	CMP $0, g
	BNE ongothread

	// On a foreign thread. We call badsignal, which will, if all
	// goes according to plan, not return.
	SUB  $4, R13
	MOVW $11, R1
	MOVW $11, R2
	MOVM.DB.W [R1,R2], (R13)
	// TODO: badsignal should not return, but it does. Issue #10139.
	//BL runtime·badsignal(SB)
	MOVW $139, R1
	MOVW R1, 4(R13)
	B    runtime·exit(SB)

ongothread:
	// Trigger a SIGSEGV panic.
	//
	// The goal is to arrange the stack so it looks like the runtime
	// function sigpanic was called from the PC that faulted. It has
	// to be sigpanic, as the stack unwinding code in traceback.go
	// looks explicitly for it.
	//
	// To do this we call into runtime·setsigsegv, which sets the
	// appropriate state inside the g object. We give it the faulting
	// PC on the stack, then put it in the LR before calling sigpanic.
	MOVM.DB.W [R1,R2], (R13)
	BL runtime·setsigsegv(SB)
	MOVM.IA.W (R13), [R1,R2]

	SUB $4, R13
	MOVW R1, 0(R13)
	MOVW R2, R14
	B runtime·sigpanic(SB)
