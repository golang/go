// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "defs.h"
#include "signals.h"

/* From /usr/include/asm-x86_64/sigcontext.h */
struct _fpstate {
  uint16   cwd;
  uint16   swd;
  uint16   twd;    /* Note this is not the same as the 32bit/x87/FSAVE twd */
  uint16   fop;
  uint64   rip;
  uint32   rdp;
  uint32   mxcsr;
  uint32   mxcsr_mask;
  uint32   st_space[32];   /* 8*16 bytes for each FP-reg */
  uint32   xmm_space[64];  /* 16*16 bytes for each XMM-reg  */
  uint32   reserved2[24];
};

struct sigcontext {
  uint64 r8;
  uint64 r9;
  uint64 r10;
  uint64 r11;
  uint64 r12;
  uint64 r13;
  uint64 r14;
  uint64 r15;
  uint64 rdi;
  uint64 rsi;
  uint64 rbp;
  uint64 rbx;
  uint64 rdx;
  uint64 rax;
  uint64 rcx;
  uint64 rsp;
  uint64 rip;
  uint64 eflags;           /* RFLAGS */
  uint16 cs;
  uint16 gs;
  uint16 fs;
  uint16 __pad0;
  uint64 err;
  uint64 trapno;
  uint64 oldmask;
  uint64 cr2;
  struct _fpstate *fpstate;       /* zero when no FPU context */
  uint64 reserved1[8];
};


/* From /usr/include/asm-x86_64/signal.h */
typedef struct sigaltstack {
	void /*__user*/ *ss_sp;
	int32 ss_flags;
	uint64 ss_size;
} stack_t;

typedef uint64 sigset_t;


/* From /usr/include/asm-x86_64/ucontext.h */
struct ucontext {
  uint64            uc_flags;
  struct ucontext  *uc_link;
  stack_t           uc_stack;
  struct sigcontext uc_mcontext;
  sigset_t          uc_sigmask;   /* mask last for extensibility */
};


void
print_sigcontext(struct sigcontext *sc)
{
	prints("\nrax     ");  sys·printhex(sc->rax);
	prints("\nrbx     ");  sys·printhex(sc->rbx);
	prints("\nrcx     ");  sys·printhex(sc->rcx);
	prints("\nrdx     ");  sys·printhex(sc->rdx);
	prints("\nrdi     ");  sys·printhex(sc->rdi);
	prints("\nrsi     ");  sys·printhex(sc->rsi);
	prints("\nrbp     ");  sys·printhex(sc->rbp);
	prints("\nrsp     ");  sys·printhex(sc->rsp);
	prints("\nr8      ");  sys·printhex(sc->r8 );
	prints("\nr9      ");  sys·printhex(sc->r9 );
	prints("\nr10     ");  sys·printhex(sc->r10);
	prints("\nr11     ");  sys·printhex(sc->r11);
	prints("\nr12     ");  sys·printhex(sc->r12);
	prints("\nr13     ");  sys·printhex(sc->r13);
	prints("\nr14     ");  sys·printhex(sc->r14);
	prints("\nr15     ");  sys·printhex(sc->r15);
	prints("\nrip     ");  sys·printhex(sc->rip);
	prints("\nrflags  ");  sys·printhex(sc->eflags);
	prints("\ncs      ");  sys·printhex(sc->cs);
	prints("\nfs      ");  sys·printhex(sc->fs);
	prints("\ngs      ");  sys·printhex(sc->gs);
	prints("\n");
}


/*
 * This assembler routine takes the args from registers, puts them on the stack,
 * and calls sighandler().
 */
extern void sigtramp(void);
extern void sigignore(void);	// just returns
extern void sigreturn(void);	// calls sigreturn

/*
 * Rudimentary reverse-engineered definition of signal interface.
 * You'd think it would be documented.
 */
/* From /usr/include/bits/siginfo.h */
struct siginfo {
	int32	si_signo;		/* signal number */
	int32	si_errno;		/* errno association */
	int32	si_code;		/* signal code */
	int32	si_status;		/* exit value */
	void	*si_addr;		/* faulting address */
	/* more stuff here */
};

// This is a struct sigaction from /usr/include/asm/signal.h
struct sigaction {
	void (*sa_handler)(int32, struct siginfo*, void*);
	uint64 sa_flags;
	void (*sa_restorer)(void);
	uint64 sa_mask;
};

void
sighandler(int32 sig, struct siginfo* info, void** context)
{
	if(panicking)	// traceback already printed
		sys_Exit(2);

	struct sigcontext *sc = &(((struct ucontext *)context)->uc_mcontext);

	if(sig < 0 || sig >= NSIG){
		prints("Signal ");
		sys·printint(sig);
	}else{
		prints(sigtab[sig].name);
	}

	prints("\nFaulting address: ");  sys·printpointer(info->si_addr);
	prints("\npc: ");  sys·printhex(sc->rip);
	prints("\n\n");

	if(gotraceback()){
		traceback((void *)sc->rip, (void *)sc->rsp, (void *)sc->r15);
		tracebackothers((void*)sc->r15);
		print_sigcontext(sc);
	}

	sys·Breakpoint();
	sys_Exit(2);
}

struct stack_t {
	void *sp;
	int32 flags;
	int32 pad;
	int64 size;
};

void
signalstack(byte *p, int32 n)
{
	struct stack_t st;

	st.sp = p;
	st.size = n;
	st.pad = 0;
	st.flags = 0;
	sigaltstack(&st, nil);
}

void	rt_sigaction(int64, void*, void*, uint64);

enum {
	SA_RESTART = 0x10000000,
	SA_ONSTACK = 0x08000000,
	SA_RESTORER = 0x04000000,
	SA_SIGINFO = 0x00000004,
};

void
initsig(void)
{
	static struct sigaction sa;

	int32 i;
	sa.sa_flags = SA_ONSTACK | SA_SIGINFO | SA_RESTORER;
	sa.sa_mask = 0xFFFFFFFFFFFFFFFFULL;
	sa.sa_restorer = (void*)sigreturn;
	for(i = 0; i<NSIG; i++) {
		if(sigtab[i].flags) {
			if(sigtab[i].flags & SigCatch)
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

