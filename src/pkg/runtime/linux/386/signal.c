// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "defs.h"
#include "signals.h"
#include "os.h"

void
dumpregs(Sigcontext *r)
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

/*
 * This assembler routine takes the args from registers, puts them on the stack,
 * and calls sighandler().
 */
extern void sigtramp(void);
extern void sigignore(void);	// just returns
extern void sigreturn(void);	// calls sigreturn

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
	Sigcontext *sc;

	if(sigtab[sig].flags & SigQueue) {
		sigsend(sig);
		return;
	}

	if(panicking)	// traceback already printed
		exit(2);
	panicking = 1;

	uc = context;
	sc = &uc->uc_mcontext;

	if(sig < 0 || sig >= NSIG)
		printf("Signal %d\n", sig);
	else
		printf("%s\n", sigtab[sig].name);

	printf("Faulting address: %p\n", *(void**)info->_sifields);
	printf("PC=%X\n", sc->eip);
	printf("\n");

	if(gotraceback()){
		traceback((void*)sc->eip, (void*)sc->esp, m->curg);
		tracebackothers(m->curg);
		dumpregs(sc);
	}

	breakpoint();
	exit(2);
}

void
signalstack(byte *p, int32 n)
{
	Sigaltstack st;

	st.ss_sp = p;
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
	sa.sa_flags = SA_ONSTACK | SA_SIGINFO | SA_RESTORER;
	sa.sa_mask = 0xFFFFFFFFFFFFFFFFULL;
	sa.sa_restorer = (void*)sigreturn;
	for(i = 0; i<NSIG; i++) {
		if(sigtab[i].flags) {
			if(sigtab[i].flags & (SigCatch | SigQueue))
				sa.k_sa_handler = (void*)sigtramp;
			else
				sa.k_sa_handler = (void*)sigignore;
			if(sigtab[i].flags & SigRestart)
				sa.sa_flags |= SA_RESTART;
			else
				sa.sa_flags &= ~SA_RESTART;
			rt_sigaction(i, &sa, nil, 8);
		}
	}
}
