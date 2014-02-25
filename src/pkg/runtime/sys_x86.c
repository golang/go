// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build amd64 amd64p32 386

#include "runtime.h"

// adjust Gobuf as it if executed a call to fn with context ctxt
// and then did an immediate gosave.
void
runtime·gostartcall(Gobuf *gobuf, void (*fn)(void), void *ctxt)
{
	uintptr *sp;
	
	sp = (uintptr*)gobuf->sp;
	if(sizeof(uintreg) > sizeof(uintptr))
		*--sp = 0;
	*--sp = (uintptr)gobuf->pc;
	gobuf->sp = (uintptr)sp;
	gobuf->pc = (uintptr)fn;
	gobuf->ctxt = ctxt;
}

// Called to rewind context saved during morestack back to beginning of function.
// To help us, the linker emits a jmp back to the beginning right after the
// call to morestack. We just have to decode and apply that jump.
void
runtime·rewindmorestack(Gobuf *gobuf)
{
	byte *pc;

	pc = (byte*)gobuf->pc;
	if(pc[0] == 0xe9) { // jmp 4-byte offset
		gobuf->pc = gobuf->pc + 5 + *(int32*)(pc+1);
		return;
	}
	if(pc[0] == 0xeb) { // jmp 1-byte offset
		gobuf->pc = gobuf->pc + 2 + *(int8*)(pc+1);
		return;
	}
	if(pc[0] == 0xcc) {
		// This is a breakpoint inserted by gdb.  We could use
		// runtime·findfunc to find the function.  But if we
		// do that, then we will continue execution at the
		// function entry point, and we will not hit the gdb
		// breakpoint.  So for this case we don't change
		// gobuf->pc, so that when we return we will execute
		// the jump instruction and carry on.  This means that
		// stack unwinding may not work entirely correctly
		// (http://golang.org/issue/5723) but the user is
		// running under gdb anyhow.
		return;
	}
	runtime·printf("runtime: pc=%p %x %x %x %x %x\n", pc, pc[0], pc[1], pc[2], pc[3], pc[4]);
	runtime·throw("runtime: misuse of rewindmorestack");
}
