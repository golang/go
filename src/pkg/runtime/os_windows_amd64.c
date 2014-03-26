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

#define DBG_PRINTEXCEPTION_C 0x40010006

// Called by sigtramp from Windows VEH handler.
// Return value signals whether the exception has been handled (-1)
// or should be made available to other handlers in the chain (0).
uint32
runtime·sighandler(ExceptionRecord *info, Context *r, G *gp)
{
	bool crash;
	uintptr *sp;

	switch(info->ExceptionCode) {
	case DBG_PRINTEXCEPTION_C:
		// This exception is intended to be caught by debuggers.
		// There is a not-very-informational message like
		// "Invalid parameter passed to C runtime function"
		// sitting at info->ExceptionInformation[0] (a wchar_t*),
		// with length info->ExceptionInformation[1].
		// The default behavior is to ignore this exception,
		// but somehow returning 0 here (meaning keep going)
		// makes the program crash instead. Maybe Windows has no
		// other handler registered? In any event, ignore it.
		return -1;

	case EXCEPTION_BREAKPOINT:
		// It is unclear whether this is needed, unclear whether it
		// would work, and unclear how to test it. Leave out for now.
		// This only handles breakpoint instructions written in the
		// assembly sources, not breakpoints set by a debugger, and
		// there are very few of the former.
		break;
	}

	if(gp != nil && runtime·issigpanic(info->ExceptionCode)) {
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
		return -1;
	}

	if(runtime·panicking)	// traceback already printed
		runtime·exit(2);
	runtime·panicking = 1;

	runtime·printf("Exception %x %p %p %p\n", info->ExceptionCode,
		info->ExceptionInformation[0], info->ExceptionInformation[1], r->Rip);


	runtime·printf("PC=%X\n", r->Rip);
	if(m->lockedg != nil && m->ncgo > 0 && gp == m->g0) {
		runtime·printf("signal arrived during cgo execution\n");
		gp = m->lockedg;
	}
	runtime·printf("\n");

	if(runtime·gotraceback(&crash)){
		runtime·traceback(r->Rip, r->Rsp, 0, gp);
		runtime·tracebackothers(gp);
		runtime·dumpregs(r);
	}
	
	if(crash)
		runtime·crash();

	runtime·exit(2);
	return -1; // not reached
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
