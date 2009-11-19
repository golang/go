// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "defs.h"
#include "os.h"
#include "signals.h"

void
dumpregs(Regs *r)
{
	printf("eax     %x\n", r->eax);
	printf("ebx     %x\n", r->ebx);
	printf("ecx     %x\n", r->ecx);
	printf("edx     %x\n", r->edx);
	printf("edi     %x\n", r->edi);
	printf("esi     %x\n", r->esi);
	printf("ebp     %x\n", r->ebp);
	printf("esp     %x\n", r->esp);
	printf("eip     %x\n", r->eip);
	printf("eflags  %x\n", r->eflags);
	printf("cs      %x\n", r->cs);
	printf("fs      %x\n", r->fs);
	printf("gs      %x\n", r->gs);
}

void
sighandler(int32 sig, Siginfo *info, void *context)
{
	Ucontext *uc;
	Mcontext *mc;
	Regs *r;

	if(panicking)	// traceback already printed
		exit(2);
	panicking = 1;

	if(sig < 0 || sig >= NSIG){
		printf("Signal %d\n", sig);
	}else{
		printf("%s\n", sigtab[sig].name);
	}

	uc = context;
	mc = uc->uc_mcontext;
	r = &mc->ss;

	printf("Faulting address: %p\n", info->si_addr);
	printf("pc: %x\n", r->eip);
	printf("\n");

	if(gotraceback()){
		traceback((void*)r->eip, (void*)r->esp, m->curg);
		tracebackothers(m->curg);
		dumpregs(r);
	}

	breakpoint();
	exit(2);
}

void
sigignore(int32, Siginfo*, void*)
{
}

void
signalstack(byte *p, int32 n)
{
	StackT st;

	st.ss_sp = p;
	st.ss_size = n;
	st.ss_flags = 0;
	sigaltstack(&st, nil);
}

void
initsig(void)
{
	int32 i;
	static Sigaction sa;

	sa.sa_flags |= SA_SIGINFO|SA_ONSTACK;
	sa.sa_mask = 0xFFFFFFFFU;
	sa.sa_tramp = sigtramp;	// sigtramp's job is to call into real handler
	for(i = 0; i<NSIG; i++) {
		if(sigtab[i].flags) {
			if(sigtab[i].flags & SigCatch) {
				sa.__sigaction_u.__sa_sigaction = sighandler;
			} else {
				sa.__sigaction_u.__sa_sigaction = sigignore;
			}
			if(sigtab[i].flags & SigRestart)
				sa.sa_flags |= SA_RESTART;
			else
				sa.sa_flags &= ~SA_RESTART;
			sigaction(i, &sa, nil);
		}
	}
}

