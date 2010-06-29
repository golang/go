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
	printf("trap    %x\n", r->trap_no);
	printf("error   %x\n", r->error_code);
	printf("oldmask %x\n", r->oldmask);
	printf("r0      %x\n", r->arm_r0);
	printf("r1      %x\n", r->arm_r1);
	printf("r2      %x\n", r->arm_r2);
	printf("r3      %x\n", r->arm_r3);
	printf("r4      %x\n", r->arm_r4);
	printf("r5      %x\n", r->arm_r5);
	printf("r6      %x\n", r->arm_r6);
	printf("r7      %x\n", r->arm_r7);
	printf("r8      %x\n", r->arm_r8);
	printf("r9      %x\n", r->arm_r9);
	printf("r10     %x\n", r->arm_r10);
	printf("fp      %x\n", r->arm_fp);
	printf("ip      %x\n", r->arm_ip);
	printf("sp      %x\n", r->arm_sp);
	printf("lr      %x\n", r->arm_lr);
	printf("pc      %x\n", r->arm_pc);
	printf("cpsr    %x\n", r->arm_cpsr);
	printf("fault   %x\n", r->fault_address);
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
	return gostringnocopy((byte*)sigtab[sig].name);
}

void
sighandler(int32 sig, Siginfo *info, void *context)
{
	Ucontext *uc;
	Sigcontext *r;
	G *gp;

	uc = context;
	r = &uc->uc_mcontext;

	if((gp = m->curg) != nil && (sigtab[sig].flags & SigPanic)) {
		// Make it look like a call to the signal func.
		// Have to pass arguments out of band since
		// augmenting the stack frame would break
		// the unwinding code.
		gp->sig = sig;
		gp->sigcode0 = info->si_code;
		gp->sigcode1 = r->fault_address;

		// If this is a leaf function, we do smash LR,
		// but we're not going back there anyway.
		// Don't bother smashing if r->arm_pc is 0,
		// which is probably a call to a nil func: the
		// old link register is more useful in the stack trace.
		if(r->arm_pc != 0)
			r->arm_lr = r->arm_pc;
		r->arm_pc = (uintptr)sigpanic;
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

	if(sig < 0 || sig >= NSIG)
		printf("Signal %d\n", sig);
	else
		printf("%s\n", sigtab[sig].name);

	printf("PC=%x\n", r->arm_pc);
	printf("\n");

	if(gotraceback()){
		traceback((void*)r->arm_pc, (void*)r->arm_sp, (void*)r->arm_lr, m->curg);
		tracebackothers(m->curg);
		printf("\n");
		dumpregs(r);
	}

//	breakpoint();
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
initsig(int32 queue)
{
	static Sigaction sa;

	siginit();

	int32 i;
	sa.sa_flags = SA_ONSTACK | SA_SIGINFO | SA_RESTORER;
	sa.sa_mask.sig[0] = 0xFFFFFFFF;
	sa.sa_mask.sig[1] = 0xFFFFFFFF;
	sa.sa_restorer = (void*)sigreturn;
	for(i = 0; i<NSIG; i++) {
		if(sigtab[i].flags) {
			if((sigtab[i].flags & SigQueue) != queue)
				continue;
			if(sigtab[i].flags & (SigCatch | SigQueue))
				sa.sa_handler = (void*)sigtramp;
			else
				sa.sa_handler = (void*)sigignore;
			if(sigtab[i].flags & SigRestart)
				sa.sa_flags |= SA_RESTART;
			else
				sa.sa_flags &= ~SA_RESTART;
			rt_sigaction(i, &sa, nil, 8);
		}
	}
}
