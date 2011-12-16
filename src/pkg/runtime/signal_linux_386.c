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
	runtime·printf("eax     %x\n", r->eax);
	runtime·printf("ebx     %x\n", r->ebx);
	runtime·printf("ecx     %x\n", r->ecx);
	runtime·printf("edx     %x\n", r->edx);
	runtime·printf("edi     %x\n", r->edi);
	runtime·printf("esi     %x\n", r->esi);
	runtime·printf("ebp     %x\n", r->ebp);
	runtime·printf("esp     %x\n", r->esp);
	runtime·printf("eip     %x\n", r->eip);
	runtime·printf("eflags  %x\n", r->eflags);
	runtime·printf("cs      %x\n", r->cs);
	runtime·printf("fs      %x\n", r->fs);
	runtime·printf("gs      %x\n", r->gs);
}

/*
 * This assembler routine takes the args from registers, puts them on the stack,
 * and calls sighandler().
 */
extern void runtime·sigtramp(void);
extern void runtime·sigignore(void);	// just returns
extern void runtime·sigreturn(void);	// calls runtime·sigreturn

String
runtime·signame(int32 sig)
{
	if(sig < 0 || sig >= NSIG)
		return runtime·emptystring;
	return runtime·gostringnocopy((byte*)runtime·sigtab[sig].name);
}

void
runtime·sighandler(int32 sig, Siginfo *info, void *context, G *gp)
{
	Ucontext *uc;
	Sigcontext *r;
	uintptr *sp;

	uc = context;
	r = &uc->uc_mcontext;

	if(sig == SIGPROF) {
		runtime·sigprof((uint8*)r->eip, (uint8*)r->esp, nil, gp);
		return;
	}

	if(gp != nil && (runtime·sigtab[sig].flags & SigPanic)) {
		// Make it look like a call to the signal func.
		// Have to pass arguments out of band since
		// augmenting the stack frame would break
		// the unwinding code.
		gp->sig = sig;
		gp->sigcode0 = info->si_code;
		gp->sigcode1 = ((uintptr*)info)[3];
		gp->sigpc = r->eip;

		// Only push runtime·sigpanic if r->eip != 0.
		// If r->eip == 0, probably panicked because of a
		// call to a nil func.  Not pushing that onto sp will
		// make the trace look like a call to runtime·sigpanic instead.
		// (Otherwise the trace will end at runtime·sigpanic and we
		// won't get to see who faulted.)
		if(r->eip != 0) {
			sp = (uintptr*)r->esp;
			*--sp = r->eip;
			r->esp = (uintptr)sp;
		}
		r->eip = (uintptr)runtime·sigpanic;
		return;
	}

	if(runtime·sigtab[sig].flags & SigQueue) {
		if(runtime·sigsend(sig) || (runtime·sigtab[sig].flags & SigIgnore))
			return;
		runtime·exit(2);	// SIGINT, SIGTERM, etc
	}

	if(runtime·panicking)	// traceback already printed
		runtime·exit(2);
	runtime·panicking = 1;

	if(sig < 0 || sig >= NSIG)
		runtime·printf("Signal %d\n", sig);
	else
		runtime·printf("%s\n", runtime·sigtab[sig].name);

	runtime·printf("PC=%X\n", r->eip);
	runtime·printf("\n");

	if(runtime·gotraceback()){
		runtime·traceback((void*)r->eip, (void*)r->esp, 0, gp);
		runtime·tracebackothers(gp);
		runtime·dumpregs(r);
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

static void
sigaction(int32 i, void (*fn)(int32, Siginfo*, void*, G*), bool restart)
{
	Sigaction sa;

	runtime·memclr((byte*)&sa, sizeof sa);
	sa.sa_flags = SA_ONSTACK | SA_SIGINFO | SA_RESTORER;
	if(restart)
		sa.sa_flags |= SA_RESTART;
	sa.sa_mask = ~0ULL;
	sa.sa_restorer = (void*)runtime·sigreturn;
	if(fn == runtime·sighandler)
		fn = (void*)runtime·sigtramp;
	sa.k_sa_handler = fn;
	runtime·rt_sigaction(i, &sa, nil, 8);
}

void
runtime·initsig(int32 queue)
{
	int32 i;
	void *fn;

	runtime·siginit();

	for(i = 0; i<NSIG; i++) {
		if(runtime·sigtab[i].flags) {
			if((runtime·sigtab[i].flags & SigQueue) != queue)
				continue;
			if(runtime·sigtab[i].flags & (SigCatch | SigQueue))
				fn = runtime·sighandler;
			else
				fn = runtime·sigignore;
			sigaction(i, fn, (runtime·sigtab[i].flags & SigRestart) != 0);
		}
	}
}

void
runtime·resetcpuprofiler(int32 hz)
{
	Itimerval it;
	
	runtime·memclr((byte*)&it, sizeof it);
	if(hz == 0) {
		runtime·setitimer(ITIMER_PROF, &it, nil);
		sigaction(SIGPROF, SIG_IGN, true);
	} else {
		sigaction(SIGPROF, runtime·sighandler, true);
		it.it_interval.tv_sec = 0;
		it.it_interval.tv_usec = 1000000 / hz;
		it.it_value = it.it_interval;
		runtime·setitimer(ITIMER_PROF, &it, nil);
	}
	m->profilehz = hz;
}

void
os·sigpipe(void)
{
	sigaction(SIGPIPE, SIG_DFL, false);
	runtime·raisesigpipe();
}

#define AT_NULL		0
#define AT_SYSINFO	32
extern uint32 runtime·_vdso;

#pragma textflag 7
void runtime·linux_setup_vdso(int32 argc, void *argv_list)
{
	byte **argv = &argv_list;
	byte **envp;
	uint32 *auxv;

	// skip envp to get to ELF auxiliary vector.
	for(envp = &argv[argc+1]; *envp != nil; envp++)
		;
	envp++;
	
	for(auxv=(uint32*)envp; auxv[0] != AT_NULL; auxv += 2) {
		if(auxv[0] == AT_SYSINFO) {
			runtime·_vdso = auxv[1];
			break;
		}
	}		
}
