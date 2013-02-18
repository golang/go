// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "defs_GOOS_GOARCH.h"
#include "signals_GOOS.h"
#include "os_GOOS.h"

#define r0	__gregs[0]
#define r1	__gregs[1]
#define r2	__gregs[2]
#define r3	__gregs[3]
#define r4	__gregs[4]
#define r5	__gregs[5]
#define r6	__gregs[6]
#define r7	__gregs[7]
#define r8	__gregs[8]
#define r9	__gregs[9]
#define r10	__gregs[10]
#define r11	__gregs[11]
#define r12	__gregs[12]
#define r13	__gregs[13]
#define r14	__gregs[14]
#define r15	__gregs[15]
#define cpsr	__gregs[16]

void
runtime·dumpregs(Mcontext *r)
{
	runtime·printf("r0      %x\n", r->r0);
	runtime·printf("r1      %x\n", r->r1);
	runtime·printf("r2      %x\n", r->r2);
	runtime·printf("r3      %x\n", r->r3);
	runtime·printf("r4      %x\n", r->r4);
	runtime·printf("r5      %x\n", r->r5);
	runtime·printf("r6      %x\n", r->r6);
	runtime·printf("r7      %x\n", r->r7);
	runtime·printf("r8      %x\n", r->r8);
	runtime·printf("r9      %x\n", r->r9);
	runtime·printf("r10     %x\n", r->r10);
	runtime·printf("fp      %x\n", r->r11);
	runtime·printf("ip      %x\n", r->r12);
	runtime·printf("sp      %x\n", r->r13);
	runtime·printf("lr      %x\n", r->r14);
	runtime·printf("pc      %x\n", r->r15);
	runtime·printf("cpsr    %x\n", r->cpsr);
}

extern void runtime·sigtramp(void);

typedef struct sigaction {
	union {
		void    (*__sa_handler)(int32);
		void    (*__sa_sigaction)(int32, Siginfo*, void *);
	} __sigaction_u;		/* signal handler */
	int32	sa_flags;		/* see signal options below */
	Sigset	sa_mask;		/* signal mask to apply */
} Sigaction;

void
runtime·sighandler(int32 sig, Siginfo *info, void *context, G *gp)
{
	Ucontext *uc;
	Mcontext *r;
	SigTab *t;

	uc = context;
	r = &uc->uc_mcontext;

	if(sig == SIGPROF) {
		runtime·sigprof((uint8*)r->r15, (uint8*)r->r13, (uint8*)r->r14, gp);
		return;
	}

	t = &runtime·sigtab[sig];
	if(info->si_code != SI_USER && (t->flags & SigPanic)) {
		if(gp == nil || gp == m->g0)
			goto Throw;
		// Make it look like a call to the signal func.
		// Have to pass arguments out of band since
		// augmenting the stack frame would break
		// the unwinding code.
		gp->sig = sig;
		gp->sigcode0 = info->si_code;
		gp->sigcode1 = (uintptr)info->si_addr;
		gp->sigpc = r->r15;

		// Only push runtime·sigpanic if r->mc_rip != 0.
		// If r->mc_rip == 0, probably panicked because of a
		// call to a nil func.  Not pushing that onto sp will
		// make the trace look like a call to runtime·sigpanic instead.
		// (Otherwise the trace will end at runtime·sigpanic and we
		// won't get to see who faulted.)
		if(r->r15 != 0)
			r->r14 = r->r15;
		// In case we are panicking from external C code
		r->r10 = (uintptr)gp;
		r->r9 = (uintptr)m;
		r->r15 = (uintptr)runtime·sigpanic;
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

	runtime·printf("PC=%x\n", r->r15);
	if(m->lockedg != nil && m->ncgo > 0 && gp == m->g0) {
		runtime·printf("signal arrived during cgo execution\n");
		gp = m->lockedg;
	}
	runtime·printf("\n");

	if(runtime·gotraceback()){
		runtime·traceback((void*)r->r15, (void*)r->r13, (void*)r->r14, gp);
		runtime·tracebackothers(gp);
		runtime·printf("\n");
		runtime·dumpregs(r);
	}

//	breakpoint();
	runtime·exit(2);
}

void
runtime·signalstack(byte *p, int32 n)
{
	Sigaltstack st;

	st.ss_sp = (uint8*)p;
	st.ss_size = n;
	st.ss_flags = 0;
	if(p == nil)
		st.ss_flags = SS_DISABLE;
	runtime·sigaltstack(&st, nil);
}

void
runtime·setsig(int32 i, void (*fn)(int32, Siginfo*, void*, G*), bool restart)
{
	Sigaction sa;

	// If SIGHUP handler is SIG_IGN, assume running
	// under nohup and do not set explicit handler.
	if(i == SIGHUP) {
		runtime·memclr((byte*)&sa, sizeof sa);
		runtime·sigaction(i, nil, &sa);
		if(sa.__sigaction_u.__sa_sigaction == SIG_IGN)
			return;
	}

	runtime·memclr((byte*)&sa, sizeof sa);
	sa.sa_flags = SA_SIGINFO|SA_ONSTACK;
	if(restart)
		sa.sa_flags |= SA_RESTART;
	sa.sa_mask.__bits[0] = ~(uint32)0;
	sa.sa_mask.__bits[1] = ~(uint32)0;
	sa.sa_mask.__bits[2] = ~(uint32)0;
	sa.sa_mask.__bits[3] = ~(uint32)0;
	if (fn == runtime·sighandler)
		fn = (void*)runtime·sigtramp;
	sa.__sigaction_u.__sa_sigaction = (void*)fn;
	runtime·sigaction(i, &sa, nil);
}

void
runtime·checkgoarm(void)
{
	// TODO(minux)
}

#pragma textflag 7
int64
runtime·cputicks() {
	// Currently cputicks() is used in blocking profiler and to seed runtime·fastrand1().
	// runtime·nanotime() is a poor approximation of CPU ticks that is enough for the profiler.
	// TODO: need more entropy to better seed fastrand1.
	return runtime·nanotime();
}
