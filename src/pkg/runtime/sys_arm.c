// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

// adjust Gobuf as if it executed a call to fn with context ctxt
// and then did an immediate Gosave.
void
runtime·gostartcall(Gobuf *gobuf, void (*fn)(void), void *ctxt)
{
	if(gobuf->lr != 0)
		runtime·throw("invalid use of gostartcall");
	gobuf->lr = gobuf->pc;
	gobuf->pc = (uintptr)fn;
	gobuf->ctxt = ctxt;
}

// Called to rewind context saved during morestack back to beginning of function.
// To help us, the linker emits a jmp back to the beginning right after the
// call to morestack. We just have to decode and apply that jump.
void
runtime·rewindmorestack(Gobuf *gobuf)
{
	uint32 inst;

	inst = *(uint32*)gobuf->pc;
	if((gobuf->pc&3) == 0 && (inst>>24) == 0x9a) {
		//runtime·printf("runtime: rewind pc=%p to pc=%p\n", gobuf->pc, gobuf->pc + ((int32)(inst<<8)>>6) + 8);
		gobuf->pc += ((int32)(inst<<8)>>6) + 8;
		return;
	}
	runtime·printf("runtime: pc=%p %x\n", gobuf->pc, inst);
	runtime·throw("runtime: misuse of rewindmorestack");
}
