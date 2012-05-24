// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "defs_GOOS_GOARCH.h"
#include "signals_GOOS.h"
#include "os_GOOS.h"

extern void runtime·lwp_tramp(void);
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
runtime·dumpregs(McontextT *mc)
{
	runtime·printf("rax     %X\n", mc->__gregs[REG_RAX]);
	runtime·printf("rbx     %X\n", mc->__gregs[REG_RBX]);
	runtime·printf("rcx     %X\n", mc->__gregs[REG_RCX]);
	runtime·printf("rdx     %X\n", mc->__gregs[REG_RDX]);
	runtime·printf("rdi     %X\n", mc->__gregs[REG_RDI]);
	runtime·printf("rsi     %X\n", mc->__gregs[REG_RSI]);
	runtime·printf("rbp     %X\n", mc->__gregs[REG_RBP]);
	runtime·printf("rsp     %X\n", mc->__gregs[REG_RSP]);
	runtime·printf("r8      %X\n", mc->__gregs[REG_R8]);
	runtime·printf("r9      %X\n", mc->__gregs[REG_R9]);
	runtime·printf("r10     %X\n", mc->__gregs[REG_R10]);
	runtime·printf("r11     %X\n", mc->__gregs[REG_R11]);
	runtime·printf("r12     %X\n", mc->__gregs[REG_R12]);
	runtime·printf("r13     %X\n", mc->__gregs[REG_R13]);
	runtime·printf("r14     %X\n", mc->__gregs[REG_R14]);
	runtime·printf("r15     %X\n", mc->__gregs[REG_R15]);
	runtime·printf("rip     %X\n", mc->__gregs[REG_RIP]);
	runtime·printf("rflags  %X\n", mc->__gregs[REG_RFLAGS]);
	runtime·printf("cs      %X\n", mc->__gregs[REG_CS]);
	runtime·printf("fs      %X\n", mc->__gregs[REG_FS]);
	runtime·printf("gs      %X\n", mc->__gregs[REG_GS]);
}

void
runtime·sighandler(int32 sig, Siginfo *info, void *context, G *gp)
{
	UcontextT *uc = context;
	McontextT *mc = &uc->uc_mcontext;
	uintptr *sp;
	SigTab *t;

	if(sig == SIGPROF) {
		runtime·sigprof((uint8*)mc->__gregs[REG_RIP],
			(uint8*)mc->__gregs[REG_RSP], nil, gp);
		return;
	}

	t = &runtime·sigtab[sig];
	if(info->_code != SI_USER && (t->flags & SigPanic)) {
		if(gp == nil)
			goto Throw;
		// Make it look like a call to the signal func.
		// We need to pass arguments out of band since augmenting the
		// stack frame would break the unwinding code.
		gp->sig = sig;
		gp->sigcode0 = info->_code;
		gp->sigcode1 = *(uintptr*)&info->_reason[0]; /* _addr */
		gp->sigpc = mc->__gregs[REG_RIP];

		// Only push runtime·sigpanic if __gregs[REG_RIP] != 0.
		// If __gregs[REG_RIP] == 0, probably panicked because of a
		// call to a nil func. Not pushing that onto sp will make the
		// trace look like a call to runtime·sigpanic instead.
		// (Otherwise the trace will end at runtime·sigpanic
		// and we won't get to see who faulted.)
		if(mc->__gregs[REG_RIP] != 0) {
			sp = (uintptr*)mc->__gregs[REG_RSP];
			*--sp = mc->__gregs[REG_RIP];
			mc->__gregs[REG_RSP] = (uintptr)sp;
		}
		mc->__gregs[REG_RIP] = (uintptr)runtime·sigpanic;
		return;
	}

	if(info->_code == SI_USER || (t->flags & SigNotify))
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

	runtime·printf("PC=%X\n", mc->__gregs[REG_RIP]);
	runtime·printf("\n");

	if(runtime·gotraceback()){
		runtime·traceback((void*)mc->__gregs[REG_RIP],
			(void*)mc->__gregs[REG_RSP], 0, gp);
		runtime·tracebackothers(gp);
		runtime·dumpregs(mc);
	}

	runtime·exit(2);
}

void
runtime·signalstack(byte *p, int32 n)
{
	Sigaltstack st;

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
	sa.sa_mask[0] = ~0U;
	sa.sa_mask[1] = ~0U;
	sa.sa_mask[2] = ~0U;
	sa.sa_mask[3] = ~0U;
	if (fn == runtime·sighandler)
		fn = (void*)runtime·sigtramp;
	sa._sa_u._sa_sigaction = (void*)fn;
	runtime·sigaction(i, &sa, nil);
}

void
runtime·lwp_mcontext_init(McontextT *mc, void *stack, M *m, G *g, void (*fn)(void))
{
	// Machine dependent mcontext initialisation for LWP.
	mc->__gregs[REG_RIP] = (uint64)runtime·lwp_tramp;
	mc->__gregs[REG_RSP] = (uint64)stack;
	mc->__gregs[REG_R8] = (uint64)m;
	mc->__gregs[REG_R9] = (uint64)g;
	mc->__gregs[REG_R12] = (uint64)fn;
}
