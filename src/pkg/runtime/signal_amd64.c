// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux netbsd openbsd

#include "runtime.h"
#include "defs_GOOS_GOARCH.h"
#include "os_GOOS.h"
#include "signal_GOOS_GOARCH.h"
#include "signals_GOOS.h"

void
runtime·dumpregs(Siginfo *info, void *ctxt)
{
	USED(info);
	USED(ctxt);
	
	runtime·printf("rax     %X\n", SIG_RAX(info, ctxt));
	runtime·printf("rbx     %X\n", SIG_RBX(info, ctxt));
	runtime·printf("rcx     %X\n", SIG_RCX(info, ctxt));
	runtime·printf("rdx     %X\n", SIG_RDX(info, ctxt));
	runtime·printf("rdi     %X\n", SIG_RDI(info, ctxt));
	runtime·printf("rsi     %X\n", SIG_RSI(info, ctxt));
	runtime·printf("rbp     %X\n", SIG_RBP(info, ctxt));
	runtime·printf("rsp     %X\n", SIG_RSP(info, ctxt));
	runtime·printf("r8      %X\n", SIG_R8(info, ctxt) );
	runtime·printf("r9      %X\n", SIG_R9(info, ctxt) );
	runtime·printf("r10     %X\n", SIG_R10(info, ctxt));
	runtime·printf("r11     %X\n", SIG_R11(info, ctxt));
	runtime·printf("r12     %X\n", SIG_R12(info, ctxt));
	runtime·printf("r13     %X\n", SIG_R13(info, ctxt));
	runtime·printf("r14     %X\n", SIG_R14(info, ctxt));
	runtime·printf("r15     %X\n", SIG_R15(info, ctxt));
	runtime·printf("rip     %X\n", SIG_RIP(info, ctxt));
	runtime·printf("rflags  %X\n", SIG_RFLAGS(info, ctxt));
	runtime·printf("cs      %X\n", SIG_CS(info, ctxt));
	runtime·printf("fs      %X\n", SIG_FS(info, ctxt));
	runtime·printf("gs      %X\n", SIG_GS(info, ctxt));
}

void
runtime·sighandler(int32 sig, Siginfo *info, void *ctxt, G *gp)
{
	uintptr *sp;
	SigTab *t;
	bool crash;

	if(sig == SIGPROF) {
		if(gp != m->g0 && gp != m->gsignal)
			runtime·sigprof((byte*)SIG_RIP(info, ctxt), (byte*)SIG_RSP(info, ctxt), nil, gp);
		return;
	}

	t = &runtime·sigtab[sig];
	if(SIG_CODE0(info, ctxt) != SI_USER && (t->flags & SigPanic)) {
		if(gp == nil || gp == m->g0)
			goto Throw;

		// Make it look like a call to the signal func.
		// Have to pass arguments out of band since
		// augmenting the stack frame would break
		// the unwinding code.
		gp->sig = sig;
		gp->sigcode0 = SIG_CODE0(info, ctxt);
		gp->sigcode1 = SIG_CODE1(info, ctxt);
		gp->sigpc = SIG_RIP(info, ctxt);

#ifdef GOOS_darwin
		// Work around Leopard bug that doesn't set FPE_INTDIV.
		// Look at instruction to see if it is a divide.
		// Not necessary in Snow Leopard (si_code will be != 0).
		if(sig == SIGFPE && gp->sigcode0 == 0) {
			byte *pc;
			pc = (byte*)gp->sigpc;
			if((pc[0]&0xF0) == 0x40)	// 64-bit REX prefix
				pc++;
			else if(pc[0] == 0x66)	// 16-bit instruction prefix
				pc++;
			if(pc[0] == 0xF6 || pc[0] == 0xF7)
				gp->sigcode0 = FPE_INTDIV;
		}
#endif

		// Only push runtime·sigpanic if rip != 0.
		// If rip == 0, probably panicked because of a
		// call to a nil func.  Not pushing that onto sp will
		// make the trace look like a call to runtime·sigpanic instead.
		// (Otherwise the trace will end at runtime·sigpanic and we
		// won't get to see who faulted.)
		if(SIG_RIP(info, ctxt) != 0) {
			sp = (uintptr*)SIG_RSP(info, ctxt);
			*--sp = SIG_RIP(info, ctxt);
			SIG_RSP(info, ctxt) = (uintptr)sp;
		}
		SIG_RIP(info, ctxt) = (uintptr)runtime·sigpanic;
		return;
	}

	if(SIG_CODE0(info, ctxt) == SI_USER || (t->flags & SigNotify))
		if(runtime·sigsend(sig))
			return;
	if(t->flags & SigKill)
		runtime·exit(2);
	if(!(t->flags & SigThrow))
		return;

Throw:
	runtime·startpanic();

	if(sig < 0 || sig >= NSIG)
		runtime·printf("Signal %d\n", sig);
	else
		runtime·printf("%s\n", runtime·sigtab[sig].name);

	runtime·printf("PC=%X\n", SIG_RIP(info, ctxt));
	if(m->lockedg != nil && m->ncgo > 0 && gp == m->g0) {
		runtime·printf("signal arrived during cgo execution\n");
		gp = m->lockedg;
	}
	runtime·printf("\n");

	if(runtime·gotraceback(&crash)){
		runtime·traceback((void*)SIG_RIP(info, ctxt), (void*)SIG_RSP(info, ctxt), 0, gp);
		runtime·tracebackothers(gp);
		runtime·dumpregs(info, ctxt);
	}
	
	if(crash)
		runtime·crash();

	runtime·exit(2);
}
