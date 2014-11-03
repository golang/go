// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "defs_GOOS_GOARCH.h"
#include "os_GOOS.h"

void
runtime·dumpregs(Context *r)
{
	runtime·printf("rax     %X\n", r->Rax);
	runtime·printf("rbx     %X\n", r->Rbx);
	runtime·printf("rcx     %X\n", r->Rcx);
	runtime·printf("rdx     %X\n", r->Rdx);
	runtime·printf("rdi     %X\n", r->Rdi);
	runtime·printf("rsi     %X\n", r->Rsi);
	runtime·printf("rbp     %X\n", r->Rbp);
	runtime·printf("rsp     %X\n", r->Rsp);
	runtime·printf("r8      %X\n", r->R8 );
	runtime·printf("r9      %X\n", r->R9 );
	runtime·printf("r10     %X\n", r->R10);
	runtime·printf("r11     %X\n", r->R11);
	runtime·printf("r12     %X\n", r->R12);
	runtime·printf("r13     %X\n", r->R13);
	runtime·printf("r14     %X\n", r->R14);
	runtime·printf("r15     %X\n", r->R15);
	runtime·printf("rip     %X\n", r->Rip);
	runtime·printf("rflags  %X\n", r->EFlags);
	runtime·printf("cs      %X\n", (uint64)r->SegCs);
	runtime·printf("fs      %X\n", (uint64)r->SegFs);
	runtime·printf("gs      %X\n", (uint64)r->SegGs);
}

bool
runtime·isgoexception(ExceptionRecord *info, Context *r)
{
	extern byte runtime·text[], runtime·etext[];

	// Only handle exception if executing instructions in Go binary
	// (not Windows library code). 
	if(r->Rip < (uint64)runtime·text || (uint64)runtime·etext < r->Rip)
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
	gp->sigpc = r->Rip;

	// Only push runtime·sigpanic if r->rip != 0.
	// If r->rip == 0, probably panicked because of a
	// call to a nil func.  Not pushing that onto sp will
	// make the trace look like a call to runtime·sigpanic instead.
	// (Otherwise the trace will end at runtime·sigpanic and we
	// won't get to see who faulted.)
	if(r->Rip != 0) {
		sp = (uintptr*)r->Rsp;
		*--sp = r->Rip;
		r->Rsp = (uintptr)sp;
	}
	r->Rip = (uintptr)runtime·sigpanic;
	return EXCEPTION_CONTINUE_EXECUTION;
}

// It seems Windows searches ContinueHandler's list even
// if ExceptionHandler returns EXCEPTION_CONTINUE_EXECUTION.
// firstcontinuehandler will stop that search,
// if exceptionhandler did the same earlier.
uint32
runtime·firstcontinuehandler(ExceptionRecord *info, Context *r, G *gp)
{
	USED(gp);
	if(!runtime·isgoexception(info, r))
		return EXCEPTION_CONTINUE_SEARCH;
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
		info->ExceptionInformation[0], info->ExceptionInformation[1], r->Rip);


	runtime·printf("PC=%X\n", r->Rip);
	if(g->m->lockedg != nil && g->m->ncgo > 0 && gp == g->m->g0) {
		runtime·printf("signal arrived during cgo execution\n");
		gp = g->m->lockedg;
	}
	runtime·printf("\n");

	if(runtime·gotraceback(&crash)){
		runtime·tracebacktrap(r->Rip, r->Rsp, 0, gp);
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
	runtime·sigprof((uint8*)r->Rip, (uint8*)r->Rsp, nil, gp, mp);
}
