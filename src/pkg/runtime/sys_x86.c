// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build amd64 386

#include "runtime.h"

// adjust Gobuf as it if executed a call to fn with context ctxt
// and then did an immediate gosave.
void
runtime·gostartcall(Gobuf *gobuf, void (*fn)(void), void *ctxt)
{
	uintptr *sp;
	
	sp = (uintptr*)gobuf->sp;
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
	Func *f;

	pc = (byte*)gobuf->pc;
	if(pc[0] == 0xe9) { // jmp 4-byte offset
		gobuf->pc = gobuf->pc + 5 + *(int32*)(pc+1);
		return;
	}
	if(pc[0] == 0xeb) { // jmp 1-byte offset
		gobuf->pc = gobuf->pc + 2 + *(int8*)(pc+1);
		return;
	}
	if(pc[0] == 0xcc) { // breakpoint inserted by gdb
		f = runtime·findfunc(gobuf->pc);
		if(f != nil) {
			gobuf->pc = f->entry;
			return;
		}
	}
	runtime·printf("runtime: pc=%p %x %x %x %x %x\n", pc, pc[0], pc[1], pc[2], pc[3], pc[4]);
	runtime·throw("runtime: misuse of rewindmorestack");
}
