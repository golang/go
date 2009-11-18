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
dumpregs(Sigcontext *r)
{
	printf("eax     %x\n", r->sc_eax);
	printf("ebx     %x\n", r->sc_ebx);
	printf("ecx     %x\n", r->sc_ecx);
	printf("edx     %x\n", r->sc_edx);
	printf("edi     %x\n", r->sc_edi);
	printf("esi     %x\n", r->sc_esi);
	printf("ebp     %x\n", r->sc_ebp);
	printf("esp     %x\n", r->sc_esp);
	printf("eip     %x\n", r->sc_eip);
	printf("eflags  %x\n", r->sc_efl);
	printf("cs      %x\n", r->sc_cs);
	printf("fs      %x\n", r->sc_fsbase);
	printf("gs      %x\n", r->sc_gsbase);
}

void
sighandler(int32 sig, Siginfo* info, void* context)
{
	Ucontext *uc;
	Mcontext *mc;
	Sigcontext *sc;

	if(panicking)	// traceback already printed
		exit(2);
	panicking = 1;

	uc = context;
	mc = &uc->uc_mcontext;
	sc = (Sigcontext*)mc;	// same layout, more conveient names

	if(sig < 0 || sig >= NSIG)
		printf("Signal %d\n", sig);
	else
		printf("%s\n", sigtab[sig].name);

	printf("Faulting address: %p\n", info->si_addr);
	printf("PC=%X\n", sc->sc_eip);
	printf("\n");

	if(gotraceback()){
		traceback((void*)sc->sc_eip, (void*)sc->sc_esp, m->curg);
		tracebackothers(m->curg);
		dumpregs(sc);
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
