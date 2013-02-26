// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "defs_GOOS_GOARCH.h"
#include "signals_GOOS.h"
#include "os_GOOS.h"

void
runtime·dumpregs(Sigcontext *r)
{
	runtime·printf("trap    %x\n", r->trap_no);
	runtime·printf("error   %x\n", r->error_code);
	runtime·printf("oldmask %x\n", r->oldmask);
	runtime·printf("r0      %x\n", r->arm_r0);
	runtime·printf("r1      %x\n", r->arm_r1);
	runtime·printf("r2      %x\n", r->arm_r2);
	runtime·printf("r3      %x\n", r->arm_r3);
	runtime·printf("r4      %x\n", r->arm_r4);
	runtime·printf("r5      %x\n", r->arm_r5);
	runtime·printf("r6      %x\n", r->arm_r6);
	runtime·printf("r7      %x\n", r->arm_r7);
	runtime·printf("r8      %x\n", r->arm_r8);
	runtime·printf("r9      %x\n", r->arm_r9);
	runtime·printf("r10     %x\n", r->arm_r10);
	runtime·printf("fp      %x\n", r->arm_fp);
	runtime·printf("ip      %x\n", r->arm_ip);
	runtime·printf("sp      %x\n", r->arm_sp);
	runtime·printf("lr      %x\n", r->arm_lr);
	runtime·printf("pc      %x\n", r->arm_pc);
	runtime·printf("cpsr    %x\n", r->arm_cpsr);
	runtime·printf("fault   %x\n", r->fault_address);
}

/*
 * This assembler routine takes the args from registers, puts them on the stack,
 * and calls sighandler().
 */
extern void runtime·sigtramp(void);
extern void runtime·sigreturn(void);	// calls runtime·sigreturn

void
runtime·sighandler(int32 sig, Siginfo *info, void *context, G *gp)
{
	Ucontext *uc;
	Sigcontext *r;
	SigTab *t;

	uc = context;
	r = &uc->uc_mcontext;

	if(sig == SIGPROF) {
		runtime·sigprof((uint8*)r->arm_pc, (uint8*)r->arm_sp, (uint8*)r->arm_lr, gp);
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
		gp->sigcode1 = r->fault_address;
		gp->sigpc = r->arm_pc;

		// We arrange lr, and pc to pretend the panicking
		// function calls sigpanic directly.
		// Always save LR to stack so that panics in leaf
		// functions are correctly handled. This smashes
		// the stack frame but we're not going back there
		// anyway.
		r->arm_sp -= 4;
		*(uint32 *)r->arm_sp = r->arm_lr;
		// Don't bother saving PC if it's zero, which is
		// probably a call to a nil func: the old link register
		// is more useful in the stack trace.
		if(r->arm_pc != 0)
			r->arm_lr = r->arm_pc;
		// In case we are panicking from external C code
		r->arm_r10 = (uintptr)gp;
		r->arm_r9 = (uintptr)m;
		r->arm_pc = (uintptr)runtime·sigpanic;
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

	if(sig < 0 || sig >= NSIG)
		runtime·printf("Signal %d\n", sig);
	else
		runtime·printf("%s\n", runtime·sigtab[sig].name);

	runtime·printf("PC=%x\n", r->arm_pc);
	if(m->lockedg != nil && m->ncgo > 0 && gp == m->g0) {
		runtime·printf("signal arrived during cgo execution\n");
		gp = m->lockedg;
	}
	runtime·printf("\n");

	if(runtime·gotraceback()){
		runtime·traceback((void*)r->arm_pc, (void*)r->arm_sp, (void*)r->arm_lr, gp);
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

	st.ss_sp = p;
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
		if(runtime·rt_sigaction(i, nil, &sa, sizeof(sa.sa_mask)) != 0)
			runtime·throw("rt_sigaction read failure");
		if(sa.sa_handler == SIG_IGN)
			return;
	}

	runtime·memclr((byte*)&sa, sizeof sa);
	sa.sa_flags = SA_ONSTACK | SA_SIGINFO | SA_RESTORER;
	if(restart)
		sa.sa_flags |= SA_RESTART;
	sa.sa_mask = ~0ULL;
	sa.sa_restorer = (void*)runtime·sigreturn;
	if(fn == runtime·sighandler)
		fn = (void*)runtime·sigtramp;
	sa.sa_handler = fn;
	if(runtime·rt_sigaction(i, &sa, nil, sizeof(sa.sa_mask)) != 0)
		runtime·throw("rt_sigaction failure");
}

#define AT_NULL		0
#define AT_PLATFORM	15 // introduced in at least 2.6.11
#define AT_HWCAP	16 // introduced in at least 2.6.11
#define AT_RANDOM	25 // introduced in 2.6.29
#define HWCAP_VFP	(1 << 6) // introduced in at least 2.6.11
#define HWCAP_VFPv3	(1 << 13) // introduced in 2.6.30
static uint32 runtime·randomNumber;
uint8  runtime·armArch = 6;	// we default to ARMv6
uint32 runtime·hwcap;	// set by setup_auxv
uint8  runtime·goarm;	// set by 5l

void
runtime·checkgoarm(void)
{
	if(runtime·goarm > 5 && !(runtime·hwcap & HWCAP_VFP)) {
		runtime·printf("runtime: this CPU has no floating point hardware, so it cannot run\n");
		runtime·printf("this GOARM=%d binary. Recompile using GOARM=5.\n", runtime·goarm);
		runtime·exit(1);
	}
	if(runtime·goarm > 6 && !(runtime·hwcap & HWCAP_VFPv3)) {
		runtime·printf("runtime: this CPU has no VFPv3 floating point hardware, so it cannot run\n");
		runtime·printf("this GOARM=%d binary. Recompile using GOARM=6.\n", runtime·goarm);
		runtime·exit(1);
	}
}

#pragma textflag 7
void
runtime·setup_auxv(int32 argc, void *argv_list)
{
	byte **argv;
	byte **envp;
	byte *rnd;
	uint32 *auxv;
	uint32 t;

	argv = &argv_list;

	// skip envp to get to ELF auxiliary vector.
	for(envp = &argv[argc+1]; *envp != nil; envp++)
		;
	envp++;
	
	for(auxv=(uint32*)envp; auxv[0] != AT_NULL; auxv += 2) {
		switch(auxv[0]) {
		case AT_RANDOM: // kernel provided 16-byte worth of random data
			if(auxv[1]) {
				rnd = (byte*)auxv[1];
				runtime·randomNumber = rnd[4] | rnd[5]<<8 | rnd[6]<<16 | rnd[7]<<24;
			}
			break;
		case AT_PLATFORM: // v5l, v6l, v7l
			if(auxv[1]) {
				t = *(uint8*)(auxv[1]+1);
				if(t >= '5' && t <= '7')
					runtime·armArch = t - '0';
			}
			break;
		case AT_HWCAP: // CPU capability bit flags
			runtime·hwcap = auxv[1];
			break;
		}
	}
}

#pragma textflag 7
int64
runtime·cputicks(void)
{
	// Currently cputicks() is used in blocking profiler and to seed runtime·fastrand1().
	// runtime·nanotime() is a poor approximation of CPU ticks that is enough for the profiler.
	// runtime·randomNumber provides better seeding of fastrand1.
	return runtime·nanotime() + runtime·randomNumber;
}
