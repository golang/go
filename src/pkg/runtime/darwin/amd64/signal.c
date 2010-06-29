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
	return gostringnocopy((byte*)sigtab[sig].name);
}

void
sighandler(int32 sig, Siginfo *info, void *context)
{
	Ucontext *uc;
	Mcontext *mc;
	Regs *r;
	G *gp;
	uintptr *sp;
	byte *pc;

	uc = context;
	mc = uc->uc_mcontext;
	r = &mc->ss;

	if((gp = m->curg) != nil && (sigtab[sig].flags & SigPanic)) {
		// Work around Leopard bug that doesn't set FPE_INTDIV.
		// Look at instruction to see if it is a divide.
		// Not necessary in Snow Leopard (si_code will be != 0).
		if(sig == SIGFPE && info->si_code == 0) {
			pc = (byte*)r->rip;
			if((pc[0]&0xF0) == 0x40)	// 64-bit REX prefix
				pc++;
			else if(pc[0] == 0x66)	// 16-bit instruction prefix
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
		
		// Only push sigpanic if r->rip != 0.
		// If r->rip == 0, probably panicked because of a
		// call to a nil func.  Not pushing that onto sp will
		// make the trace look like a call to sigpanic instead.
		// (Otherwise the trace will end at sigpanic and we
		// won't get to see who faulted.)
		if(r->rip != 0) {
			sp = (uintptr*)r->rsp;
			*--sp = r->rip;
			r->rsp = (uintptr)sp;
		}
		r->rip = (uintptr)sigpanic;
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

	printf("pc: %X\n", r->rip);
	printf("\n");

	if(gotraceback()){
		traceback((void*)r->rip, (void*)r->rsp, 0, (void*)r->r15);
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
initsig(int32 queue)
{
	int32 i;
	static Sigaction sa;

	siginit();

	sa.sa_flags |= SA_SIGINFO|SA_ONSTACK;
	sa.sa_mask = 0xFFFFFFFFU;
	sa.sa_tramp = sigtramp;	// sigtramp's job is to call into real handler
	for(i = 0; i<NSIG; i++) {
		if(sigtab[i].flags) {
			if((sigtab[i].flags & SigQueue) != queue)
				continue;
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
