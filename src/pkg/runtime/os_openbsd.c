// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "defs_GOOS_GOARCH.h"
#include "os_GOOS.h"
#include "signal_unix.h"
#include "stack.h"

enum
{
	ESRCH = 3,
	ENOTSUP = 91,

	// From OpenBSD's sys/time.h
	CLOCK_REALTIME = 0,
	CLOCK_VIRTUAL = 1,
	CLOCK_PROF = 2,
	CLOCK_MONOTONIC = 3
};

extern SigTab runtime·sigtab[];

static Sigset sigset_none;
static Sigset sigset_all = ~(Sigset)0;

extern int64 runtime·tfork(void *param, uintptr psize, M *mp, G *gp, void (*fn)(void));
extern int32 runtime·thrsleep(void *ident, int32 clock_id, void *tsp, void *lock, const int32 *abort);
extern int32 runtime·thrwakeup(void *ident, int32 n);

// From OpenBSD's <sys/sysctl.h>
#define	CTL_HW	6
#define	HW_NCPU	3

static int32
getncpu(void)
{
	uint32 mib[2];
	uint32 out;
	int32 ret;
	uintptr nout;

	// Fetch hw.ncpu via sysctl.
	mib[0] = CTL_HW;
	mib[1] = HW_NCPU;
	nout = sizeof out;
	out = 0;
	ret = runtime·sysctl(mib, 2, (byte*)&out, &nout, nil, 0);
	if(ret >= 0)
		return out;
	else
		return 1;
}

uintptr
runtime·semacreate(void)
{
	return 1;
}

int32
runtime·semasleep(int64 ns)
{
	Timespec ts;
	int64 secs;

	// spin-mutex lock
	while(runtime·xchg(&m->waitsemalock, 1))
		runtime·osyield();

	for(;;) {
		// lock held
		if(m->waitsemacount == 0) {
			// sleep until semaphore != 0 or timeout.
			// thrsleep unlocks m->waitsemalock.
			if(ns < 0)
				runtime·thrsleep(&m->waitsemacount, 0, nil, &m->waitsemalock, nil);
			else {
				ns += runtime·nanotime();
				secs = ns/1000000000LL;
				// Avoid overflow
				if(secs >= 1LL<<31)
					secs = (1LL<<31) - 1;
				ts.tv_sec = secs;
				ts.tv_nsec = ns%1000000000LL;
				runtime·thrsleep(&m->waitsemacount, CLOCK_REALTIME, &ts, &m->waitsemalock, nil);
			}
			// reacquire lock
			while(runtime·xchg(&m->waitsemalock, 1))
				runtime·osyield();
		}

		// lock held (again)
		if(m->waitsemacount != 0) {
			// semaphore is available.
			m->waitsemacount--;
			// spin-mutex unlock
			runtime·atomicstore(&m->waitsemalock, 0);
			return 0;  // semaphore acquired
		}

		// semaphore not available.
		// if there is a timeout, stop now.
		// otherwise keep trying.
		if(ns >= 0)
			break;
	}

	// lock held but giving up
	// spin-mutex unlock
	runtime·atomicstore(&m->waitsemalock, 0);
	return -1;
}

void
runtime·semawakeup(M *mp)
{
	uint32 ret;

	// spin-mutex lock
	while(runtime·xchg(&mp->waitsemalock, 1))
		runtime·osyield();
	mp->waitsemacount++;
	ret = runtime·thrwakeup(&mp->waitsemacount, 1);
	if(ret != 0 && ret != ESRCH)
		runtime·printf("thrwakeup addr=%p sem=%d ret=%d\n", &mp->waitsemacount, mp->waitsemacount, ret);
	// spin-mutex unlock
	runtime·atomicstore(&mp->waitsemalock, 0);
}

void
runtime·newosproc(M *mp, void *stk)
{
	Tfork param;
	Sigset oset;
	int32 ret;

	if(0) {
		runtime·printf(
			"newosproc stk=%p m=%p g=%p id=%d/%d ostk=%p\n",
			stk, mp, mp->g0, mp->id, (int32)mp->tls[0], &mp);
	}

	mp->tls[0] = mp->id;	// so 386 asm can find it

	param.tf_tcb = (byte*)&mp->tls[0];
	param.tf_tid = (int32*)&mp->procid;
	param.tf_stack = stk;

	oset = runtime·sigprocmask(SIG_SETMASK, sigset_all);
	ret = runtime·tfork((byte*)&param, sizeof(param), mp, mp->g0, runtime·mstart);
	runtime·sigprocmask(SIG_SETMASK, oset);

	if(ret < 0) {
		runtime·printf("runtime: failed to create new OS thread (have %d already; errno=%d)\n", runtime·mcount() - 1, -ret);
		if (ret == -ENOTSUP)
			runtime·printf("runtime: is kern.rthreads disabled?\n");
		runtime·throw("runtime.newosproc");
	}
}

void
runtime·osinit(void)
{
	runtime·ncpu = getncpu();
}

void
runtime·get_random_data(byte **rnd, int32 *rnd_len)
{
	static byte urandom_data[HashRandomBytes];
	int32 fd;
	fd = runtime·open("/dev/urandom", 0 /* O_RDONLY */, 0);
	if(runtime·read(fd, urandom_data, HashRandomBytes) == HashRandomBytes) {
		*rnd = urandom_data;
		*rnd_len = HashRandomBytes;
	} else {
		*rnd = nil;
		*rnd_len = 0;
	}
	runtime·close(fd);
}

void
runtime·goenvs(void)
{
	runtime·goenvs_unix();
}

// Called to initialize a new m (including the bootstrap m).
// Called on the parent thread (main thread in case of bootstrap), can allocate memory.
void
runtime·mpreinit(M *mp)
{
	mp->gsignal = runtime·malg(32*1024);
}

// Called to initialize a new m (including the bootstrap m).
// Called on the new thread, can not allocate memory.
void
runtime·minit(void)
{
	// Initialize signal handling
	runtime·signalstack((byte*)m->gsignal->stackguard - StackGuard, 32*1024);
	runtime·sigprocmask(SIG_SETMASK, sigset_none);
}

// Called from dropm to undo the effect of an minit.
void
runtime·unminit(void)
{
	runtime·signalstack(nil, 0);
}

void
runtime·sigpanic(void)
{
	switch(g->sig) {
	case SIGBUS:
		if(g->sigcode0 == BUS_ADRERR && g->sigcode1 < 0x1000) {
			if(g->sigpc == 0)
				runtime·panicstring("call of nil func value");
			runtime·panicstring("invalid memory address or nil pointer dereference");
		}
		runtime·printf("unexpected fault address %p\n", g->sigcode1);
		runtime·throw("fault");
	case SIGSEGV:
		if((g->sigcode0 == 0 || g->sigcode0 == SEGV_MAPERR || g->sigcode0 == SEGV_ACCERR) && g->sigcode1 < 0x1000) {
			if(g->sigpc == 0)
				runtime·panicstring("call of nil func value");
			runtime·panicstring("invalid memory address or nil pointer dereference");
		}
		runtime·printf("unexpected fault address %p\n", g->sigcode1);
		runtime·throw("fault");
	case SIGFPE:
		switch(g->sigcode0) {
		case FPE_INTDIV:
			runtime·panicstring("integer divide by zero");
		case FPE_INTOVF:
			runtime·panicstring("integer overflow");
		}
		runtime·panicstring("floating point error");
	}
	runtime·panicstring(runtime·sigtab[g->sig].name);
}

uintptr
runtime·memlimit(void)
{
	return 0;
}

void
runtime·setprof(bool on)
{
	USED(on);
}

static int8 badcallback[] = "runtime: cgo callback on thread not created by Go.\n";

// This runs on a foreign stack, without an m or a g.  No stack split.
#pragma textflag 7
void
runtime·badcallback(void)
{
	runtime·write(2, badcallback, sizeof badcallback - 1);
}

static int8 badsignal[] = "runtime: signal received on thread not created by Go: ";

// This runs on a foreign stack, without an m or a g.  No stack split.
#pragma textflag 7
void
runtime·badsignal(int32 sig)
{
	int32 len;

	if (sig == SIGPROF) {
		return;  // Ignore SIGPROFs intended for a non-Go thread.
	}
	runtime·write(2, badsignal, sizeof badsignal - 1);
	if (0 <= sig && sig < NSIG) {
		// Can't call findnull() because it will split stack.
		for(len = 0; runtime·sigtab[sig].name[len]; len++)
			;
		runtime·write(2, runtime·sigtab[sig].name, len);
	}
	runtime·write(2, "\n", 1);
	runtime·exit(1);
}

extern void runtime·sigtramp(void);

typedef struct sigaction {
	union {
		void    (*__sa_handler)(int32);
		void    (*__sa_sigaction)(int32, Siginfo*, void *);
	} __sigaction_u;		/* signal handler */
	uint32	sa_mask;		/* signal mask to apply */
	int32	sa_flags;		/* see signal options below */
} Sigaction;

void
runtime·setsig(int32 i, GoSighandler *fn, bool restart)
{
	Sigaction sa;

	runtime·memclr((byte*)&sa, sizeof sa);
	sa.sa_flags = SA_SIGINFO|SA_ONSTACK;
	if(restart)
		sa.sa_flags |= SA_RESTART;
	sa.sa_mask = ~0U;
	if(fn == runtime·sighandler)
		fn = (void*)runtime·sigtramp;
	sa.__sigaction_u.__sa_sigaction = (void*)fn;
	runtime·sigaction(i, &sa, nil);
}

GoSighandler*
runtime·getsig(int32 i)
{
	Sigaction sa;

	runtime·memclr((byte*)&sa, sizeof sa);
	runtime·sigaction(i, nil, &sa);
	if((void*)sa.__sigaction_u.__sa_sigaction == runtime·sigtramp)
		return runtime·sighandler;
	return (void*)sa.__sigaction_u.__sa_sigaction;
}

void
runtime·signalstack(byte *p, int32 n)
{
	StackT st;

	st.ss_sp = (void*)p;
	st.ss_size = n;
	st.ss_flags = 0;
	if(p == nil)
		st.ss_flags = SS_DISABLE;
	runtime·sigaltstack(&st, nil);
}
