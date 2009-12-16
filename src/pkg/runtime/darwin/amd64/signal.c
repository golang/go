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
	printf("rax     %X\n", r->rax);
	printf("rbx     %X\n", r->rbx);
	printf("rcx     %X\n", r->rcx);
	printf("rdx     %X\n", r->rdx);
	printf("rdi     %X\n", r->rdi);
	printf("rsi     %X\n", r->rsi);
	printf("rbp     %X\n", r->rbp);
	printf("rsp     %X\n", r->rsp);
	printf("r8      %X\n", r->r8 );
	printf("r9      %X\n", r->r9 );
	printf("r10     %X\n", r->r10);
	printf("r11     %X\n", r->r11);
	printf("r12     %X\n", r->r12);
	printf("r13     %X\n", r->r13);
	printf("r14     %X\n", r->r14);
	printf("r15     %X\n", r->r15);
	printf("rip     %X\n", r->rip);
	printf("rflags  %X\n", r->rflags);
	printf("cs      %X\n", r->cs);
	printf("fs      %X\n", r->fs);
	printf("gs      %X\n", r->gs);
}

String
signame(int32 sig)
{
	if(sig < 0 || sig >= NSIG)
		return emptystring;
	return gostring((byte*)sigtab[sig].name);
}

void
sighandler(int32 sig, Siginfo *info, void *context)
{
	Ucontext *uc;
	Mcontext *mc;
	Regs *r;

	if(sigtab[sig].flags & SigQueue) {
		sigsend(sig);
		return;
	}

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
	printf("pc: %X\n", r->rip);
	printf("\n");

	if(gotraceback()){
		traceback((void*)r->rip, (void*)r->rsp, (void*)r->r15);
		tracebackothers((void*)r->r15);
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

	siginit();

	sa.sa_flags |= SA_SIGINFO|SA_ONSTACK;
	sa.sa_mask = 0xFFFFFFFFU;
	sa.sa_tramp = sigtramp;	// sigtramp's job is to call into real handler
	for(i = 0; i<NSIG; i++) {
		if(sigtab[i].flags) {
			if(sigtab[i].flags & (SigCatch | SigQueue)) {
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
