// Copyright 2009 The Go Authors. All rights reserved.
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

	runtime·printf("trap    %x\n", SIG_TRAP(info, ctxt));
	runtime·printf("error   %x\n", SIG_ERROR(info, ctxt));
	runtime·printf("oldmask %x\n", SIG_OLDMASK(info, ctxt));
	runtime·printf("r0      %x\n", SIG_R0(info, ctxt));
	runtime·printf("r1      %x\n", SIG_R1(info, ctxt));
	runtime·printf("r2      %x\n", SIG_R2(info, ctxt));
	runtime·printf("r3      %x\n", SIG_R3(info, ctxt));
	runtime·printf("r4      %x\n", SIG_R4(info, ctxt));
	runtime·printf("r5      %x\n", SIG_R5(info, ctxt));
	runtime·printf("r6      %x\n", SIG_R6(info, ctxt));
	runtime·printf("r7      %x\n", SIG_R7(info, ctxt));
	runtime·printf("r8      %x\n", SIG_R8(info, ctxt));
	runtime·printf("r9      %x\n", SIG_R9(info, ctxt));
	runtime·printf("r10     %x\n", SIG_R10(info, ctxt));
	runtime·printf("fp      %x\n", SIG_FP(info, ctxt));
	runtime·printf("ip      %x\n", SIG_IP(info, ctxt));
	runtime·printf("sp      %x\n", SIG_SP(info, ctxt));
	runtime·printf("lr      %x\n", SIG_LR(info, ctxt));
	runtime·printf("pc      %x\n", SIG_PC(info, ctxt));
	runtime·printf("cpsr    %x\n", SIG_CPSR(info, ctxt));
	runtime·printf("fault   %x\n", SIG_FAULT(info, ctxt));
}

void
runtime·sighandler(int32 sig, Siginfo *info, void *ctxt, G *gp)
{
	SigTab *t;
	bool crash;

	if(sig == SIGPROF) {
		runtime·sigprof((uint8*)SIG_PC(info, ctxt), (uint8*)SIG_SP(info, ctxt), (uint8*)SIG_LR(info, ctxt), gp, g->m);
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
		gp->sigcode1 = SIG_FAULT(info, ctxt);
		gp->sigpc = SIG_PC(info, ctxt);

		// We arrange lr, and pc to pretend the panicking
		// function calls sigpanic directly.
		// Always save LR to stack so that panics in leaf
		// functions are correctly handled. This smashes
		// the stack frame but we're not going back there
		// anyway.
		SIG_SP(info, ctxt) -= 4;
		*(uint32*)SIG_SP(info, ctxt) = SIG_LR(info, ctxt);
		// Don't bother saving PC if it's zero, which is
		// probably a call to a nil func: the old link register
		// is more useful in the stack trace.
		if(gp->sigpc != 0)
			SIG_LR(info, ctxt) = gp->sigpc;
		// In case we are panicking from external C code
		SIG_R10(info, ctxt) = (uintptr)gp;
		SIG_PC(info, ctxt) = (uintptr)runtime·sigpanic;
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
	if(runtime·panicking)	// traceback already printed
		runtime·exit(2);
	runtime·panicking = 1;

	if(sig < 0 || sig >= NSIG)
		runtime·printf("Signal %d\n", sig);
	else
		runtime·printf("%s\n", runtime·sigtab[sig].name);

	runtime·printf("PC=%x\n", SIG_PC(info, ctxt));
	if(g->m->lockedg != nil && g->m->ncgo > 0 && gp == g->m->g0) {
		runtime·printf("signal arrived during cgo execution\n");
		gp = g->m->lockedg;
	}
	runtime·printf("\n");

	if(runtime·gotraceback(&crash)){
		runtime·goroutineheader(gp);
		runtime·tracebacktrap(SIG_PC(info, ctxt), SIG_SP(info, ctxt), SIG_LR(info, ctxt), gp);
		runtime·tracebackothers(gp);
		runtime·printf("\n");
		runtime·dumpregs(info, ctxt);
	}
	
	if(crash)
		runtime·crash();

	runtime·exit(2);
}
