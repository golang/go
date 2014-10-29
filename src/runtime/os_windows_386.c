// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "defs_GOOS_GOARCH.h"
#include "os_GOOS.h"

void
runtime·dumpregs(Context *r)
{
	runtime·printf("eax     %x\n", r->Eax);
	runtime·printf("ebx     %x\n", r->Ebx);
	runtime·printf("ecx     %x\n", r->Ecx);
	runtime·printf("edx     %x\n", r->Edx);
	runtime·printf("edi     %x\n", r->Edi);
	runtime·printf("esi     %x\n", r->Esi);
	runtime·printf("ebp     %x\n", r->Ebp);
	runtime·printf("esp     %x\n", r->Esp);
	runtime·printf("eip     %x\n", r->Eip);
	runtime·printf("eflags  %x\n", r->EFlags);
	runtime·printf("cs      %x\n", r->SegCs);
	runtime·printf("fs      %x\n", r->SegFs);
	runtime·printf("gs      %x\n", r->SegGs);
}

bool
runtime·isgoexception(ExceptionRecord *info, Context *r)
{
	extern byte runtime·text[], runtime·etext[];

	// Only handle exception if executing instructions in Go binary
	// (not Windows library code). 
	if(r->Eip < (uint32)runtime·text || (uint32)runtime·etext < r->Eip)
		return false;

	if(!runtime·issigpanic(info->ExceptionCode))
		return false;

	return true;
}

// Called by sigtramp from Windows VEH handler.
// Return value signals whether the exception has been handled (EXCEPTION_CONTINUE_EXECUTION)
// or should be made available to other handlers in the chain (EXCEPTION_CONTINUE_SEARCH).
uint32
runtime·exceptionhandler(ExceptionRecord *info, Context *r, G *gp)
{
	uintptr *sp;

	if(!runtime·isgoexception(info, r))
		return EXCEPTION_CONTINUE_SEARCH;

	// Make it look like a call to the signal func.
	// Have to pass arguments out of band since
	// augmenting the stack frame would break
	// the unwinding code.
	gp->sig = info->ExceptionCode;
	gp->sigcode0 = info->ExceptionInformation[0];
	gp->sigcode1 = info->ExceptionInformation[1];
	gp->sigpc = r->Eip;

	// Only push runtime·sigpanic if r->eip != 0.
	// If r->eip == 0, probably panicked because of a
	// call to a nil func.  Not pushing that onto sp will
	// make the trace look like a call to runtime·sigpanic instead.
	// (Otherwise the trace will end at runtime·sigpanic and we
	// won't get to see who faulted.)
	if(r->Eip != 0) {
		sp = (uintptr*)r->Esp;
		*--sp = r->Eip;
		r->Esp = (uintptr)sp;
	}
	r->Eip = (uintptr)runtime·sigpanic;
	return EXCEPTION_CONTINUE_EXECUTION;
}

// lastcontinuehandler is reached, because runtime cannot handle
// current exception. lastcontinuehandler will print crash info and exit.
uint32
runtime·lastcontinuehandler(ExceptionRecord *info, Context *r, G *gp)
{
	bool crash;

	if(runtime·panicking)	// traceback already printed
		runtime·exit(2);
	runtime·panicking = 1;

	runtime·printf("Exception %x %p %p %p\n", info->ExceptionCode,
		(uintptr)info->ExceptionInformation[0], (uintptr)info->ExceptionInformation[1], (uintptr)r->Eip);

	runtime·printf("PC=%x\n", r->Eip);
	if(g->m->lockedg != nil && g->m->ncgo > 0 && gp == g->m->g0) {
		runtime·printf("signal arrived during cgo execution\n");
		gp = g->m->lockedg;
	}
	runtime·printf("\n");

	if(runtime·gotraceback(&crash)){
		runtime·tracebacktrap(r->Eip, r->Esp, 0, gp);
		runtime·tracebackothers(gp);
		runtime·dumpregs(r);
	}
	
	if(crash)
		runtime·crash();

	runtime·exit(2);
	return 0; // not reached
}

void
runtime·sigenable(uint32 sig)
{
	USED(sig);
}

void
runtime·sigdisable(uint32 sig)
{
	USED(sig);
}

void
runtime·dosigprof(Context *r, G *gp, M *mp)
{
	runtime·sigprof((uint8*)r->Eip, (uint8*)r->Esp, nil, gp, mp);
}
