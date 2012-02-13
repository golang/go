// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "defs_GOOS_GOARCH.h"
#include "os_GOOS.h"
#include "signals_GOOS.h"

void
runtime·dumpregs(Regs64 *r)
{
	runtime·printf("rax     %X\n", r->rax);
	runtime·printf("rbx     %X\n", r->rbx);
	runtime·printf("rcx     %X\n", r->rcx);
	runtime·printf("rdx     %X\n", r->rdx);
	runtime·printf("rdi     %X\n", r->rdi);
	runtime·printf("rsi     %X\n", r->rsi);
	runtime·printf("rbp     %X\n", r->rbp);
	runtime·printf("rsp     %X\n", r->rsp);
	runtime·printf("r8      %X\n", r->r8 );
	runtime·printf("r9      %X\n", r->r9 );
	runtime·printf("r10     %X\n", r->r10);
	runtime·printf("r11     %X\n", r->r11);
	runtime·printf("r12     %X\n", r->r12);
	runtime·printf("r13     %X\n", r->r13);
	runtime·printf("r14     %X\n", r->r14);
	runtime·printf("r15     %X\n", r->r15);
	runtime·printf("rip     %X\n", r->rip);
	runtime·printf("rflags  %X\n", r->rflags);
	runtime·printf("cs      %X\n", r->cs);
	runtime·printf("fs      %X\n", r->fs);
	runtime·printf("gs      %X\n", r->gs);
}

void
runtime·sighandler(int32 sig, Siginfo *info, void *context, G *gp)
{
	Ucontext *uc;
	Mcontext64 *mc;
	Regs64 *r;
	uintptr *sp;
	byte *pc;
	SigTab *t;

	uc = context;
	mc = uc->uc_mcontext;
	r = &mc->ss;

	if(sig == SIGPROF) {
		runtime·sigprof((uint8*)r->rip, (uint8*)r->rsp, nil, gp);
		return;
	}

	t = &runtime·sigtab[sig];
	if(info->si_code != SI_USER && (t->flags & SigPanic)) {
		if(gp == nil)
			goto Throw;
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
		gp->sigpc = r->rip;
		
		// Only push runtime·sigpanic if r->rip != 0.
		// If r->rip == 0, probably panicked because of a
		// call to a nil func.  Not pushing that onto sp will
		// make the trace look like a call to runtime·sigpanic instead.
		// (Otherwise the trace will end at runtime·sigpanic and we
		// won't get to see who faulted.)
		if(r->rip != 0) {
			sp = (uintptr*)r->rsp;
			*--sp = r->rip;
			r->rsp = (uintptr)sp;
		}
		r->rip = (uintptr)runtime·sigpanic;
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
	if(runtime·panicking)	// traceback already printed
		runtime·exit(2);
	runtime·panicking = 1;

	if(sig < 0 || sig >= NSIG){
		runtime·printf("Signal %d\n", sig);
	}else{
		runtime·printf("%s\n", runtime·sigtab[sig].name);
	}

	runtime·printf("pc: %X\n", r->rip);
	runtime·printf("\n");

	if(runtime·gotraceback()){
		runtime·traceback((void*)r->rip, (void*)r->rsp, 0, gp);
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
	sa.sa_mask = ~0ULL;
	sa.sa_tramp = runtime·sigtramp;	// runtime·sigtramp's job is to call into real handler
	*(uintptr*)sa.__sigaction_u = (uintptr)fn;
	runtime·sigaction(i, &sa, nil);
}
