// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "amd64_linux.h"
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
	tracebackothers((void*)sc->r15);
	print_sigcontext(sc);

	sys·breakpoint();
	sys·exit(2);
}


static sigaction a;

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

// Linux futex.  The simple cases really are simple:
//
//	futex(addr, FUTEX_WAIT, val, duration, _, _)
//		Inside the kernel, atomically check that *addr == val
//		and go to sleep for at most duration.
//
//	futex(addr, FUTEX_WAKE, val, _, _, _)
//		Wake up at least val procs sleeping on addr.
//
// (Of course, they have added more complicated things since then.)

enum
{	
	FUTEX_WAIT = 0,
	FUTEX_WAKE = 1,
	
	EINTR = 4,
	EAGAIN = 11,
};

// TODO(rsc) I tried using 1<<40 here but it woke up (-ETIMEDOUT).
// I wonder if the timespec that gets to the kernel
// actually has two 32-bit numbers in it, so that
// a 64-bit 1<<40 ends up being 0 seconds, 
// 1<<8 nanoseconds.
static struct timespec longtime =
{
	1<<30,	// 34 years
	0
};

static void
efutex(uint32 *addr, int32 op, int32 val, struct timespec *ts)
{
	int64 ret;

again:
	ret = futex(addr, op, val, ts, nil, 0);

	// These happen when you use a debugger, among other times.
	if(ret == -EAGAIN || ret == -EINTR){
		// If we were sleeping, it's okay to wake up early.
		if(op == FUTEX_WAIT)
			return;
		
		// If we were waking someone up, we don't know
		// whether that succeeded, so wake someone else up too.
		if(op == FUTEX_WAKE){
prints("futexwake ");
sys·printint(ret);
prints("\n");
			goto again;
		}
	}

	if(ret < 0){
		prints("futex error addr=");
		sys·printpointer(addr);
		prints(" op=");
		sys·printint(op);
		prints(" val=");
		sys·printint(val);
		prints(" ts=");
		sys·printpointer(ts);
		prints(" returned ");
		sys·printint(-ret);
		prints("\n");
		*(int32*)101 = 202;
	}
}

// Lock and unlock.  
// A zeroed Lock is unlocked (no need to initialize each lock).
// The l->key is either 0 (unlocked), 1 (locked), or >=2 (contended).

void
lock(Lock *l)
{
	uint32 v;
	
	if(l->key != 0) *(int32*)0x1001 = 0x1001;
	l->key = 1;
	return;

	for(;;){
		// Try for lock.  If we incremented it from 0 to 1, we win.
		if((v=xadd(&l->key, 1)) == 1)
			return;

		// We lose.  It was already >=1 and is now >=2.
		// Use futex to atomically check that the value is still
		// what we think it is and go to sleep.
		efutex(&l->key, FUTEX_WAIT, v, &longtime);
	}
}

void
unlock(Lock *l)
{
	uint32 v;

	if(l->key != 1) *(int32*)0x1002 = 0x1002;
	l->key = 0;
	return;

	// Unlock the lock.  If we decremented from 1 to 0, wasn't contended.
	if((v=xadd(&l->key, -1)) == 0)
		return;
	
	// The lock was contended.  Mark it as unlocked and wake a waiter.
	l->key = 0;
	efutex(&l->key, FUTEX_WAKE, 1, nil);
}

// Sleep and wakeup (see description in runtime.h)

void
rsleep(Rendez *r)
{
	// Record that we're about to go to sleep and drop the lock.
	r->sleeping = 1;
	unlock(r->l);
	
	// Go to sleep if r->sleeping is still 1.
	efutex(&r->sleeping, FUTEX_WAIT, 1, &longtime);

	// Reacquire the lock.
	lock(r->l);
}

void
rwakeup(Rendez *r)
{
	if(!r->sleeping)
		return;

	// Clear the sleeping flag in case sleeper
	// is between unlock and futex.
	r->sleeping = 0;
	
	// Wake up if actually made it to sleep.
	efutex(&r->sleeping, FUTEX_WAKE, 1, nil);
}

// Like rwakeup(r), unlock(r->l), but drops the lock before
// waking the other proc.  This reduces bouncing back and forth
// in the scheduler: the first thing the other proc wants to do
// is acquire r->l, so it helps to unlock it before we wake him.
void
rwakeupandunlock(Rendez *r)
{
	int32 wassleeping;
	
	if(!r->sleeping){
		unlock(r->l);
		return;
	}

	r->sleeping = 0;
	unlock(r->l);
	efutex(&r->sleeping, FUTEX_WAKE, 1, nil);
}

enum
{
	CLONE_VM = 0x100,
	CLONE_FS = 0x200,
	CLONE_FILES = 0x400,
	CLONE_SIGHAND = 0x800,
	CLONE_PTRACE = 0x2000,
	CLONE_VFORK = 0x4000,
	CLONE_PARENT = 0x8000,
	CLONE_THREAD = 0x10000,
	CLONE_NEWNS = 0x20000,
	CLONE_SYSVSEM = 0x40000,
	CLONE_SETTLS = 0x80000,
	CLONE_PARENT_SETTID = 0x100000,
	CLONE_CHILD_CLEARTID = 0x200000,
	CLONE_UNTRACED = 0x800000,
	CLONE_CHILD_SETTID = 0x1000000,
	CLONE_STOPPED = 0x2000000,
	CLONE_NEWUTS = 0x4000000,
	CLONE_NEWIPC = 0x8000000,
};

void
newosproc(M *mm, G *gg, void *stk, void (*fn)(void*), void *arg)
{
	int64 ret;
	int32 flags;
	
	flags = CLONE_PARENT	/* getppid doesn't change in child */
		| CLONE_VM	/* share memory */
		| CLONE_FS	/* share cwd, etc */
		| CLONE_FILES	/* share fd table */
		| CLONE_SIGHAND	/* share sig handler table */
		| CLONE_PTRACE	/* revisit - okay for now */
		| CLONE_THREAD	/* revisit - okay for now */
		;

	if(0){
		prints("newosproc stk=");
		sys·printpointer(stk);
		prints(" mm=");
		sys·printpointer(mm);
		prints(" gg=");
		sys·printpointer(gg);
		prints(" fn=");
		sys·printpointer(fn);
		prints(" arg=");
		sys·printpointer(arg);
		prints(" clone=");
		sys·printpointer(clone);
		prints("\n");
	}

	ret = clone(flags, stk, mm, gg, fn, arg);
	if(ret < 0)
		*(int32*)123 = 123;
}

void
sys·sleep(int64 ms)
{
	struct timeval tv;

	tv.tv_sec = ms/1000;
	tv.tv_usec = ms%1000 * 1000;
	select(0, nil, nil, nil, &tv);
}

