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
	uintptr *sp;
	void (*fn)(void);
	G *gp;
	byte *pc;

	uc = context;
	mc = uc->uc_mcontext;
	r = &mc->ss;

	if((gp = m->curg) != nil && (sigtab[sig].flags & SigPanic)) {
		// Work around Leopard bug that doesn't set FPE_INTDIV.
		// Look at instruction to see if it is a divide.
		// Not necessary in Snow Leopard (si_code will be != 0).
		if(sig == SIGFPE && info->si_code == 0) {
			pc = (byte*)r->eip;
			if(pc[0] == 0xF7)
				info->si_code = FPE_INTDIV;
		}
		
		// Make it look like a call to the signal func.
		// Have to pass arguments out of band since
		// augmenting the stack frame would break
		// the unwinding code.
		gp->sig = sig;
		gp->sigcode0 = info->si_code;
		gp->sigcode1 = (uintptr)info->si_addr;

		sp = (uintptr*)r->esp;
		*--sp = r->eip;
		r->eip = (uintptr)sigpanic;
		r->esp = (uintptr)sp;
		return;
	}

	if(sigtab[sig].flags & SigQueue) {
		if(sigsend(sig) || (sigtab[sig].flags & SigIgnore))
			return;
		exit(2);	// SIGINT, SIGTERM, etc
	}

	if(panicking)	// traceback already printed
		exit(2);
	panicking = 1;

	if(sig < 0 || sig >= NSIG){
		printf("Signal %d\n", sig);
	}else{
		printf("%s\n", sigtab[sig].name);
	}

	printf("pc: %x\n", r->eip);
	printf("\n");

	if(gotraceback()){
		traceback((void*)r->eip, (void*)r->esp, 0, m->curg);
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
