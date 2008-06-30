// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
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
	prints("\nrax     0x");  sys·printpointer((void*)sc->rax);
	prints("\nrbx     0x");  sys·printpointer((void*)sc->rbx);
	prints("\nrcx     0x");  sys·printpointer((void*)sc->rcx);
	prints("\nrdx     0x");  sys·printpointer((void*)sc->rdx);
	prints("\nrdi     0x");  sys·printpointer((void*)sc->rdi);
	prints("\nrsi     0x");  sys·printpointer((void*)sc->rsi);
	prints("\nrbp     0x");  sys·printpointer((void*)sc->rbp);
	prints("\nrsp     0x");  sys·printpointer((void*)sc->rsp);
	prints("\nr8      0x");  sys·printpointer((void*)sc->r8 );
	prints("\nr9      0x");  sys·printpointer((void*)sc->r9 );
	prints("\nr10     0x");  sys·printpointer((void*)sc->r10);
	prints("\nr11     0x");  sys·printpointer((void*)sc->r11);
	prints("\nr12     0x");  sys·printpointer((void*)sc->r12);
	prints("\nr13     0x");  sys·printpointer((void*)sc->r13);
	prints("\nr14     0x");  sys·printpointer((void*)sc->r14);
	prints("\nr15     0x");  sys·printpointer((void*)sc->r15);
	prints("\nrip     0x");  sys·printpointer((void*)sc->rip);
	prints("\nrflags  0x");  sys·printpointer((void*)sc->eflags);
	prints("\ncs      0x");  sys·printpointer((void*)sc->cs);
	prints("\nfs      0x");  sys·printpointer((void*)sc->fs);
	prints("\ngs      0x");  sys·printpointer((void*)sc->gs);
	prints("\n");
}


/*
 * This assembler routine takes the args from registers, puts them on the stack,
 * and calls sighandler().
 */
extern void sigtramp();

/*
 * Rudimentary reverse-engineered definition of signal interface.
 * You'd think it would be documented.
 */
/* From /usr/include/bits/siginfo.h */
typedef struct siginfo {
	int32	si_signo;		/* signal number */
	int32	si_errno;		/* errno association */
	int32	si_code;		/* signal code */
	int32	si_status;		/* exit value */
	void	*si_addr;		/* faulting address */
	/* more stuff here */
} siginfo;


/* From /usr/include/bits/sigaction.h */
/* (gri) Is this correct? See e.g. /usr/include/asm-x86_64/signal.h */
typedef struct sigaction {
 	union {
		void (*sa_handler)(int32);
		void (*sa_sigaction)(int32, siginfo *, void *);
	} u;				/* signal handler */
	uint8 sa_mask[128];		/* signal mask to apply. 128? are they MORONS? */
	int32 sa_flags;			/* see signal options below */
	void (*sa_restorer) (void);	/* unused here; needed to return from trap? */
} sigaction;


void
sighandler(int32 sig, siginfo* info, void** context)
{
	int32 i;

	if(sig < 0 || sig >= NSIG){
		prints("Signal ");
		sys·printint(sig);
	}else{
		prints(sigtab[sig].name);
	}
        
        struct sigcontext *sc = &(((struct ucontext *)context)->uc_mcontext);
        
        prints("\nFaulting address: 0x");  sys·printpointer(info->si_addr);
        prints("\npc: 0x");  sys·printpointer((void *)sc->rip);
        prints("\n\n");
        
	traceback((void *)sc->rip, (void *)sc->rsp, (void *)sc->r15);
        print_sigcontext(sc);

	sys·breakpoint();
	sys·exit(2);
}


sigaction a;

void
initsig(void)
{
	int32 i;
	a.u.sa_sigaction = (void*)sigtramp;
	a.sa_flags = 0x04;  /* SA_SIGINFO */
	for(i=0; i<sizeof(a.sa_mask); i++)
		a.sa_mask[i] = 0xFF;

	for(i = 0; i<NSIG; i++)
		if(sigtab[i].catch){
			sys·rt_sigaction(i, &a, (void*)0, 8);
		}
}
