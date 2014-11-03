// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux nacl netbsd openbsd

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
	
	runtime·printf("eax     %x\n", SIG_EAX(info, ctxt));
	runtime·printf("ebx     %x\n", SIG_EBX(info, ctxt));
	runtime·printf("ecx     %x\n", SIG_ECX(info, ctxt));
	runtime·printf("edx     %x\n", SIG_EDX(info, ctxt));
	runtime·printf("edi     %x\n", SIG_EDI(info, ctxt));
	runtime·printf("esi     %x\n", SIG_ESI(info, ctxt));
	runtime·printf("ebp     %x\n", SIG_EBP(info, ctxt));
	runtime·printf("esp     %x\n", SIG_ESP(info, ctxt));
	runtime·printf("eip     %x\n", SIG_EIP(info, ctxt));
	runtime·printf("eflags  %x\n", SIG_EFLAGS(info, ctxt));
	runtime·printf("cs      %x\n", SIG_CS(info, ctxt));
	runtime·printf("fs      %x\n", SIG_FS(info, ctxt));
	runtime·printf("gs      %x\n", SIG_GS(info, ctxt));
}

void
runtime·sighandler(int32 sig, Siginfo *info, void *ctxt, G *gp)
{
	uintptr *sp;
	SigTab *t;
	bool crash;

	if(sig == SIGPROF) {
		runtime·sigprof((byte*)SIG_EIP(info, ctxt), (byte*)SIG_ESP(info, ctxt), nil, gp, g->m);
		return;
	}

	t = &runtime·sigtab[sig];
	if(SIG_CODE0(info, ctxt) != SI_USER && (t->flags & SigPanic)) {
		// Make it look like a call to the signal func.
		// Have to pass arguments out of band since
		// augmenting the stack frame would break
		// the unwinding code.
		gp->sig = sig;
		gp->sigcode0 = SIG_CODE0(info, ctxt);
		gp->sigcode1 = SIG_CODE1(info, ctxt);
		gp->sigpc = SIG_EIP(info, ctxt);

#ifdef GOOS_darwin
		// Work around Leopard bug that doesn't set FPE_INTDIV.
		// Look at instruction to see if it is a divide.
		// Not necessary in Snow Leopard (si_code will be != 0).
		if(sig == SIGFPE && gp->sigcode0 == 0) {
			byte *pc;
			pc = (byte*)gp->sigpc;
			if(pc[0] == 0x66)	// 16-bit instruction prefix
				pc++;
			if(pc[0] == 0xF6 || pc[0] == 0xF7)
				gp->sigcode0 = FPE_INTDIV;
		}
#endif

		// Only push runtime·sigpanic if eip != 0.
		// If eip == 0, probably panicked because of a
		// call to a nil func.  Not pushing that onto sp will
		// make the trace look like a call to runtime·sigpanic instead.
		// (Otherwise the trace will end at runtime·sigpanic and we
		// won't get to see who faulted.)
		if(SIG_EIP(info, ctxt) != 0) {
			sp = (uintptr*)SIG_ESP(info, ctxt);
			*--sp = SIG_EIP(info, ctxt);
			SIG_ESP(info, ctxt) = (uintptr)sp;
		}
		SIG_EIP(info, ctxt) = (uintptr)runtime·sigpanic;
		return;
	}

	if(SIG_CODE0(info, ctxt) == SI_USER || (t->flags & SigNotify))
		if(runtime·sigsend(sig))
			return;
	if(t->flags & SigKill)
		runtime·exit(2);
	if(!(t->flags & SigThrow))
		return;

	g->m->throwing = 1;
	g->m->caughtsig = gp;
	runtime·startpanic();

	if(sig < 0 || sig >= NSIG)
		runtime·printf("Signal %d\n", sig);
	else
		runtime·printf("%s\n", runtime·sigtab[sig].name);

	runtime·printf("PC=%x\n", SIG_EIP(info, ctxt));
	if(g->m->lockedg != nil && g->m->ncgo > 0 && gp == g->m->g0) {
		runtime·printf("signal arrived during cgo execution\n");
		gp = g->m->lockedg;
	}
	runtime·printf("\n");

	if(runtime·gotraceback(&crash)){
		runtime·goroutineheader(gp);
		runtime·tracebacktrap(SIG_EIP(info, ctxt), SIG_ESP(info, ctxt), 0, gp);
		runtime·tracebackothers(gp);
		runtime·printf("\n");
		runtime·dumpregs(info, ctxt);
	}
	
	if(crash)
		runtime·crash();

	runtime·exit(2);
}
