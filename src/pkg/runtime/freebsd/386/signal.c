// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "defs.h"
#include "signals.h"
#include "os.h"

extern void runtime·sigtramp(void);

typedef struct sigaction {
	union {
		void    (*__sa_handler)(int32);
		void    (*__sa_sigaction)(int32, Siginfo*, void *);
	} __sigaction_u;		/* signal handler */
	int32	sa_flags;		/* see signal options below */
	int64	sa_mask;		/* signal mask to apply */
} Sigaction;

void
runtime·dumpregs(Mcontext *r)
{
	runtime·printf("eax     %x\n", r->mc_eax);
	runtime·printf("ebx     %x\n", r->mc_ebx);
	runtime·printf("ecx     %x\n", r->mc_ecx);
	runtime·printf("edx     %x\n", r->mc_edx);
	runtime·printf("edi     %x\n", r->mc_edi);
	runtime·printf("esi     %x\n", r->mc_esi);
	runtime·printf("ebp     %x\n", r->mc_ebp);
	runtime·printf("esp     %x\n", r->mc_esp);
	runtime·printf("eip     %x\n", r->mc_eip);
	runtime·printf("eflags  %x\n", r->mc_eflags);
	runtime·printf("cs      %x\n", r->mc_cs);
	runtime·printf("fs      %x\n", r->mc_fs);
	runtime·printf("gs      %x\n", r->mc_gs);
}

String
runtime·signame(int32 sig)
{
	if(sig < 0 || sig >= NSIG)
		return runtime·emptystring;
	return runtime·gostringnocopy((byte*)runtime·sigtab[sig].name);
}

void
runtime·sighandler(int32 sig, Siginfo* info, void* context)
{
	Ucontext *uc;
	Mcontext *r;
	G *gp;
	uintptr *sp;

	uc = context;
	r = &uc->uc_mcontext;

	if((gp = m->curg) != nil && (runtime·sigtab[sig].flags & SigPanic)) {
		// Make it look like a call to the signal func.
		// Have to pass arguments out of band since
		// augmenting the stack frame would break
		// the unwinding code.
		gp->sig = sig;
		gp->sigcode0 = info->si_code;
		gp->sigcode1 = (uintptr)info->si_addr;

		// Only push runtime·sigpanic if r->mc_eip != 0.
		// If r->mc_eip == 0, probably panicked because of a
		// call to a nil func.  Not pushing that onto sp will
		// make the trace look like a call to runtime·sigpanic instead.
		// (Otherwise the trace will end at runtime·sigpanic and we
		// won't get to see who faulted.)
		if(r->mc_eip != 0) {
			sp = (uintptr*)r->mc_esp;
			*--sp = r->mc_eip;
			r->mc_esp = (uintptr)sp;
		}
		r->mc_eip = (uintptr)runtime·sigpanic;
		return;
	}

	if(runtime·sigtab[sig].flags & SigQueue) {
		if(runtime·sigsend(sig) || (runtime·sigtab[sig].flags & SigIgnore))
			return;
		runtime·exit(2);	// SIGINT, SIGTERM, etc
	}

	if(runtime·panicking)	// traceback already printed
		runtime·exit(2);
	runtime·panicking = 1;

	if(sig < 0 || sig >= NSIG)
		runtime·printf("Signal %d\n", sig);
	else
		runtime·printf("%s\n", runtime·sigtab[sig].name);

	runtime·printf("PC=%X\n", r->mc_eip);
	runtime·printf("\n");

	if(runtime·gotraceback()){
		runtime·traceback((void*)r->mc_eip, (void*)r->mc_esp, 0, m->curg);
		runtime·tracebackothers(m->curg);
		runtime·dumpregs(r);
	}

	runtime·breakpoint();
	runtime·exit(2);
}

void
runtime·sigignore(void)
{
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
runtime·initsig(int32 queue)
{
	static Sigaction sa;

	runtime·siginit();

	int32 i;
	sa.sa_flags |= SA_ONSTACK | SA_SIGINFO;
	sa.sa_mask = ~0x0ull;
	
	for(i = 0; i < NSIG; i++) {
		if(runtime·sigtab[i].flags) {
			if((runtime·sigtab[i].flags & SigQueue) != queue)
				continue;
			if(runtime·sigtab[i].flags & (SigCatch | SigQueue))
				sa.__sigaction_u.__sa_sigaction = (void*) runtime·sigtramp;
			else
				sa.__sigaction_u.__sa_sigaction = (void*) runtime·sigignore;

			if(runtime·sigtab[i].flags & SigRestart)
				sa.sa_flags |= SA_RESTART;
			else
				sa.sa_flags &= ~SA_RESTART;

			runtime·sigaction(i, &sa, nil);
		}
	}
}
