#include "runtime.h"
#include "defs.h"
#include "signals.h"
#include "os.h"

extern void sigtramp(void);

typedef struct sigaction {
	union {
		void    (*__sa_handler)(int32);
		void    (*__sa_sigaction)(int32, Siginfo*, void *);
	} __sigaction_u;		/* signal handler */
	int32	sa_flags;		/* see signal options below */
	int64	sa_mask;		/* signal mask to apply */
} Sigaction;

void
dumpregs(Mcontext *r)
{
	printf("rax     %X\n", r->mc_rax);
	printf("rbx     %X\n", r->mc_rbx);
	printf("rcx     %X\n", r->mc_rcx);
	printf("rdx     %X\n", r->mc_rdx);
	printf("rdi     %X\n", r->mc_rdi);
	printf("rsi     %X\n", r->mc_rsi);
	printf("rbp     %X\n", r->mc_rbp);
	printf("rsp     %X\n", r->mc_rsp);
	printf("r8      %X\n", r->mc_r8 );
	printf("r9      %X\n", r->mc_r9 );
	printf("r10     %X\n", r->mc_r10);
	printf("r11     %X\n", r->mc_r11);
	printf("r12     %X\n", r->mc_r12);
	printf("r13     %X\n", r->mc_r13);
	printf("r14     %X\n", r->mc_r14);
	printf("r15     %X\n", r->mc_r15);
	printf("rip     %X\n", r->mc_rip);
	printf("rflags  %X\n", r->mc_flags);
	printf("cs      %X\n", r->mc_cs);
	printf("fs      %X\n", r->mc_fs);
	printf("gs      %X\n", r->mc_gs);
}

void
sighandler(int32 sig, Siginfo* info, void* context)
{
	Ucontext *uc;
	Mcontext *mc;

	if(panicking)	// traceback already printed
		exit(2);
	panicking = 1;

	uc = context;
	mc = &uc->uc_mcontext;

	if(sig < 0 || sig >= NSIG)
		printf("Signal %d\n", sig);
	else
		printf("%s\n", sigtab[sig].name);

	printf("Faulting address: %p\n", info->si_addr);
	printf("PC=%X\n", mc->mc_rip);
	printf("\n");

	if(gotraceback()){
		traceback((void*)mc->mc_rip, (void*)mc->mc_rsp, (void*)mc->mc_r15);
		tracebackothers((void*)mc->mc_r15);
		dumpregs(mc);
	}

	breakpoint();
	exit(2);
}

void
sigignore(void)
{
}

void
signalstack(byte *p, int32 n)
{
	Sigaltstack st;

	st.ss_sp = (int8*)p;
	st.ss_size = n;
	st.ss_flags = 0;
	sigaltstack(&st, nil);
}

void
initsig(void)
{
	static Sigaction sa;

	int32 i;
	sa.sa_flags |= SA_ONSTACK | SA_SIGINFO;
	sa.sa_mask = ~0x0ull;
	
	for(i = 0; i < NSIG; i++) {
		if(sigtab[i].flags) {
			if(sigtab[i].flags & SigCatch)
				sa.__sigaction_u.__sa_handler = (void*) sigtramp;
			else
				sa.__sigaction_u.__sa_handler = (void*) sigignore;

			if(sigtab[i].flags & SigRestart)
				sa.sa_flags |= SA_RESTART;
			else
				sa.sa_flags &= ~SA_RESTART;

			sigaction(i, &sa, nil);
		}
	}
}
