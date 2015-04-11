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
	MOVW $139, R1
	MOVW R1, (RSP)
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
	STP.W (R1, R2), -16(RSP)
	BL runtime·setsigsegv(SB)
	LDP.P 16(RSP), (R1, R2)

	MOVD R1, 8(RSP)
	MOVD R2, R30 // link register
	B runtime·sigpanic(SB)
