// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "defs.h"
#include "signals.h"
#include "os.h"

extern void sigtramp(void);

typedef struct sigaction {
	union {
		void    (*__sa_handler)(int32);
		void    (*__sa_sigaction)(int32, Siginfo*, void *);
	} __sigaction_u;		/* signal handler */
	int32	sa_flags;		/* see signal options below */
	int64	sa_mask;		/* signal mask to apply */
} Sigaction;

void
dumpregs(Mcontext *r)
{
	printf("eax     %x\n", r->mc_eax);
	printf("ebx     %x\n", r->mc_ebx);
	printf("ecx     %x\n", r->mc_ecx);
	printf("edx     %x\n", r->mc_edx);
	printf("edi     %x\n", r->mc_edi);
	printf("esi     %x\n", r->mc_esi);
	printf("ebp     %x\n", r->mc_ebp);
	printf("esp     %x\n", r->mc_esp);
	printf("eip     %x\n", r->mc_eip);
	printf("eflags  %x\n", r->mc_eflags);
	printf("cs      %x\n", r->mc_cs);
	printf("fs      %x\n", r->mc_fs);
	printf("gs      %x\n", r->mc_gs);
}

String
signame(int32 sig)
{
	if(sig < 0 || sig >= NSIG)
		return emptystring;
	return gostring((byte*)sigtab[sig].name);
}

void
sighandler(int32 sig, Siginfo* info, void* context)
{
	Ucontext *uc;
	Mcontext *mc;

	if(sigtab[sig].flags & SigQueue) {
		if(sigsend(sig) || (sigtab[sig].flags & SigIgnore))
			return;
		exit(2);	// SIGINT, SIGTERM, etc
	}

	if(panicking)	// traceback already printed
		exit(2);
	panicking = 1;

	uc = context;
	mc = &uc->uc_mcontext;

	if(sig < 0 || sig >= NSIG)
		printf("Signal %d\n", sig);
	else
		printf("%s\n", sigtab[sig].name);

	printf("Faulting address: %p\n", info->si_addr);
	printf("PC=%X\n", mc->mc_eip);
	printf("\n");

	if(gotraceback()){
		traceback((void*)mc->mc_eip, (void*)mc->mc_esp, m->curg);
		tracebackothers(m->curg);
		dumpregs(mc);
	}

	breakpoint();
	exit(2);
}

void
sigignore(void)
{
}

void
signalstack(byte *p, int32 n)
{
	Sigaltstack st;

	st.ss_sp = (int8*)p;
	st.ss_size = n;
	st.ss_flags = 0;
	sigaltstack(&st, nil);
}

void
initsig(void)
{
	static Sigaction sa;

	siginit();

	int32 i;
	sa.sa_flags |= SA_ONSTACK | SA_SIGINFO;
	sa.sa_mask = ~0x0ull;
	
	for(i = 0; i < NSIG; i++) {
		if(sigtab[i].flags) {
			if(sigtab[i].flags & (SigCatch | SigQueue))
				sa.__sigaction_u.__sa_sigaction = (void*) sigtramp;
			else
				sa.__sigaction_u.__sa_sigaction = (void*) sigignore;

			if(sigtab[i].flags & SigRestart)
				sa.sa_flags |= SA_RESTART;
			else
				sa.sa_flags &= ~SA_RESTART;

			sigaction(i, &sa, nil);
		}
	}
}
