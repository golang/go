// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// panicmem is the entrypoint for SIGSEGV as intercepted via a
// mach thread port as EXC_BAD_ACCESS. As the segfault may have happened
// in C code, we first need to load_g then call panicmem.
//
//	R1 - LR at moment of fault
//	R2 - PC at moment of fault
TEXT ·panicmem(SB),NOSPLIT,$-8
	// If in external C code, we need to load the g register.
	BL  runtime·load_g(SB)
	CMP $0, g
	BNE ongothread

	// On a foreign thread.
	// TODO(crawshaw): call badsignal
	MOVD.W $0, -16(RSP)
	MOVW $139, R1
	MOVW R1, 8(RSP)
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

	// Build a 32-byte stack frame for us for this call.
	// Saved LR (none available) is at the bottom,
	// then the PC argument for setsigsegv, 
	// then a copy of the LR for us to restore.
	MOVD.W $0, -32(RSP)
	MOVD R1, 8(RSP)
	MOVD R2, 16(RSP)
	BL runtime·setsigsegv(SB)
	MOVD 8(RSP), R1
	MOVD 16(RSP), R2

	// Build a 16-byte stack frame for the simulated
	// call to sigpanic, by taking 16 bytes away from the
	// 32-byte stack frame above.
	// The saved LR in this frame is the LR at time of fault,
	// and the LR on entry to sigpanic is the PC at time of fault.
	MOVD.W R1, 16(RSP)
	MOVD R2, R30
	B runtime·sigpanic(SB)
