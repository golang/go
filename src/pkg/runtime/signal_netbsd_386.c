// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "defs_GOOS_GOARCH.h"
#include "signals_GOOS.h"
#include "os_GOOS.h"

extern void runtime·sigtramp(void);

typedef struct sigaction {
	union {
		void    (*_sa_handler)(int32);
		void    (*_sa_sigaction)(int32, Siginfo*, void *);
	} _sa_u;			/* signal handler */
	uint32	sa_mask[4];		/* signal mask to apply */
	int32	sa_flags;		/* see signal options below */
} Sigaction;

void
runtime·dumpregs(Sigcontext *r)
{
	runtime·printf("eax     %x\n", r->sc_eax);
	runtime·printf("ebx     %x\n", r->sc_ebx);
	runtime·printf("ecx     %x\n", r->sc_ecx);
	runtime·printf("edx     %x\n", r->sc_edx);
	runtime·printf("edi     %x\n", r->sc_edi);
	runtime·printf("esi     %x\n", r->sc_esi);
	runtime·printf("ebp     %x\n", r->sc_ebp);
	runtime·printf("esp     %x\n", r->sc_esp);
	runtime·printf("eip     %x\n", r->sc_eip);
	runtime·printf("eflags  %x\n", r->sc_eflags);
	runtime·printf("cs      %x\n", r->sc_cs);
	runtime·printf("fs      %x\n", r->sc_fs);
	runtime·printf("gs      %x\n", r->sc_gs);
}

void
runtime·sighandler(int32 sig, Siginfo *info, void *context, G *gp)
{
	Sigcontext *r = context;
	uintptr *sp;
	SigTab *t;

	if(sig == SIGPROF) {
		runtime·sigprof((uint8*)r->sc_eip, (uint8*)r->sc_esp, nil, gp);
		return;
	}

	t = &runtime·sigtab[sig];
	if(info->si_code != SI_USER && (t->flags & SigPanic)) {
		if(gp == nil)
			goto Throw;
		// Make it look like a call to the signal func.
		// Have to pass arguments out of band since
		// augmenting the stack frame would break
		// the unwinding code.
		gp->sig = sig;
		gp->sigcode0 = info->si_code;
		gp->sigcode1 = *(uintptr*)((byte*)info + 12); /* si_addr */
		gp->sigpc = r->sc_eip;

		// Only push runtime·sigpanic if r->sc_eip != 0.
		// If r->sc_eip == 0, probably panicked because of a
		// call to a nil func.  Not pushing that onto sp will
		// make the trace look like a call to runtime·sigpanic instead.
		// (Otherwise the trace will end at runtime·sigpanic and we
		// won't get to see who faulted.)
		if(r->sc_eip != 0) {
			sp = (uintptr*)r->sc_esp;
			*--sp = r->sc_eip;
			r->sc_esp = (uintptr)sp;
		}
		r->sc_eip = (uintptr)runtime·sigpanic;
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

	if(sig < 0 || sig >= NSIG)
		runtime·printf("Signal %d\n", sig);
	else
		runtime·printf("%s\n", runtime·sigtab[sig].name);

	runtime·printf("PC=%X\n", r->sc_eip);
	runtime·printf("\n");

	if(runtime·gotraceback()){
		runtime·traceback((void*)r->sc_eip, (void*)r->sc_esp, 0, gp);
		runtime·tracebackothers(gp);
		runtime·dumpregs(r);
	}

	runtime·exit(2);
}

void
runtime·signalstack(byte *p, int32 n)
{
	Sigaltstack st;

	st.ss_sp = (int8*)p;
	st.ss_size = n;
	st.ss_flags = 0;
	runtime·sigaltstack(&st, nil);
}

void
runtime·setsig(int32 i, void (*fn)(int32, Siginfo*, void*, G*), bool restart)
{
	Sigaction sa;

	runtime·memclr((byte*)&sa, sizeof sa);
	sa.sa_flags = SA_SIGINFO|SA_ONSTACK;
	if(restart)
		sa.sa_flags |= SA_RESTART;
	sa.sa_mask[0] = ~0U;
	sa.sa_mask[1] = ~0U;
	sa.sa_mask[2] = ~0U;
	sa.sa_mask[3] = ~0U;
	if (fn == runtime·sighandler)
		fn = (void*)runtime·sigtramp;
	sa._sa_u._sa_sigaction = (void*)fn;
	runtime·sigaction(i, &sa, nil);
}
