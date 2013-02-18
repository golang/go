// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "defs_GOOS_GOARCH.h"
#include "os_GOOS.h"
#include "signals_GOOS.h"

void
runtime·dumpregs(Regs32 *r)
{
	runtime·printf("eax     %x\n", r->eax);
	runtime·printf("ebx     %x\n", r->ebx);
	runtime·printf("ecx     %x\n", r->ecx);
	runtime·printf("edx     %x\n", r->edx);
	runtime·printf("edi     %x\n", r->edi);
	runtime·printf("esi     %x\n", r->esi);
	runtime·printf("ebp     %x\n", r->ebp);
	runtime·printf("esp     %x\n", r->esp);
	runtime·printf("eip     %x\n", r->eip);
	runtime·printf("eflags  %x\n", r->eflags);
	runtime·printf("cs      %x\n", r->cs);
	runtime·printf("fs      %x\n", r->fs);
	runtime·printf("gs      %x\n", r->gs);
}

void
runtime·sighandler(int32 sig, Siginfo *info, void *context, G *gp)
{
	Ucontext *uc;
	Mcontext32 *mc;
	Regs32 *r;
	uintptr *sp;
	byte *pc;
	SigTab *t;

	uc = context;
	mc = uc->uc_mcontext;
	r = &mc->ss;

	if(sig == SIGPROF) {
		if(gp != m->g0 && gp != m->gsignal)
			runtime·sigprof((uint8*)r->eip, (uint8*)r->esp, nil, gp);
		return;
	}

	t = &runtime·sigtab[sig];
	if(info->si_code != SI_USER && (t->flags & SigPanic)) {
		if(gp == nil || gp == m->g0)
			goto Throw;
		// Work around Leopard bug that doesn't set FPE_INTDIV.
		// Look at instruction to see if it is a divide.
		// Not necessary in Snow Leopard (si_code will be != 0).
		if(sig == SIGFPE && info->si_code == 0) {
			pc = (byte*)r->eip;
			if(pc[0] == 0x66)	// 16-bit instruction prefix
				pc++;
			if(pc[0] == 0xF6 || pc[0] == 0xF7)
				info->si_code = FPE_INTDIV;
		}

		// Make it look like a call to the signal func.
		// Have to pass arguments out of band since
		// augmenting the stack frame would break
		// the unwinding code.
		gp->sig = sig;
		gp->sigcode0 = info->si_code;
		gp->sigcode1 = (uintptr)info->si_addr;
		gp->sigpc = r->eip;

		// Only push runtime·sigpanic if r->eip != 0.
		// If r->eip == 0, probably panicked because of a
		// call to a nil func.  Not pushing that onto sp will
		// make the trace look like a call to runtime·sigpanic instead.
		// (Otherwise the trace will end at runtime·sigpanic and we
		// won't get to see who faulted.)
		if(r->eip != 0) {
			sp = (uintptr*)r->esp;
			*--sp = r->eip;
			r->esp = (uintptr)sp;
		}
		r->eip = (uintptr)runtime·sigpanic;
		return;
	}

	if(info->si_code == SI_USER || (t->flags & SigNotify))
		if(runtime·sigsend(sig))
			return;
	if(t->flags & SigKill)
		runtime·exit(2);
	if(!(t->flags & SigThrow))
		return;

Throw:
	runtime·startpanic();

	if(sig < 0 || sig >= NSIG){
		runtime·printf("Signal %d\n", sig);
	}else{
		runtime·printf("%s\n", runtime·sigtab[sig].name);
	}

	runtime·printf("PC=%x\n", r->eip);
	if(m->lockedg != nil && m->ncgo > 0 && gp == m->g0) {
		runtime·printf("signal arrived during cgo execution\n");
		gp = m->lockedg;
	}	
	runtime·printf("\n");

	if(runtime·gotraceback()){
		runtime·traceback((void*)r->eip, (void*)r->esp, 0, gp);
		runtime·tracebackothers(gp);
		runtime·dumpregs(r);
	}

	runtime·exit(2);
}

void
runtime·signalstack(byte *p, int32 n)
{
	StackT st;

	st.ss_sp = p;
	st.ss_size = n;
	st.ss_flags = 0;
	if(p == nil)
		st.ss_flags = SS_DISABLE;
	runtime·sigaltstack(&st, nil);
}

void
runtime·setsig(int32 i, void (*fn)(int32, Siginfo*, void*, G*), bool restart)
{
	Sigaction sa;

	// If SIGHUP handler is SIG_IGN, assume running
	// under nohup and do not set explicit handler.
	if(i == SIGHUP) {
		runtime·memclr((byte*)&sa, sizeof sa);
		runtime·sigaction(i, nil, &sa);
		if(*(void**)sa.__sigaction_u == SIG_IGN)
			return;
	}

	runtime·memclr((byte*)&sa, sizeof sa);
	sa.sa_flags = SA_SIGINFO|SA_ONSTACK;
	if(restart)
		sa.sa_flags |= SA_RESTART;
	sa.sa_mask = ~0U;
	sa.sa_tramp = (void*)runtime·sigtramp;	// runtime·sigtramp's job is to call into real handler
	*(uintptr*)sa.__sigaction_u = (uintptr)fn;
	runtime·sigaction(i, &sa, nil);
}
