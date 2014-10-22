// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "defs_GOOS_GOARCH.h"
#include "os_GOOS.h"
#include "signal_unix.h"
#include "stack.h"
#include "textflag.h"

enum
{
	ESRCH = 3,
	EAGAIN = 35,
	EWOULDBLOCK = EAGAIN,
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

extern int32 runtime·tfork(TforkT *param, uintptr psize, M *mp, G *gp, void (*fn)(void));
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

#pragma textflag NOSPLIT
uintptr
runtime·semacreate(void)
{
	return 1;
}

#pragma textflag NOSPLIT
int32
runtime·semasleep(int64 ns)
{
	Timespec ts, *tsp = nil;

	// Compute sleep deadline.
	if(ns >= 0) {
		int32 nsec;
		ns += runtime·nanotime();
		ts.tv_sec = runtime·timediv(ns, 1000000000, &nsec);
		ts.tv_nsec = nsec; // tv_nsec is int64 on amd64
		tsp = &ts;
	}

	for(;;) {
		int32 ret;

		// spin-mutex lock
		while(runtime·xchg(&g->m->waitsemalock, 1))
			runtime·osyield();

		if(g->m->waitsemacount != 0) {
			// semaphore is available.
			g->m->waitsemacount--;
			// spin-mutex unlock
			runtime·atomicstore(&g->m->waitsemalock, 0);
			return 0;  // semaphore acquired
		}

		// sleep until semaphore != 0 or timeout.
		// thrsleep unlocks m->waitsemalock.
		ret = runtime·thrsleep(&g->m->waitsemacount, CLOCK_MONOTONIC, tsp, &g->m->waitsemalock, (int32 *)&g->m->waitsemacount);
		if(ret == EWOULDBLOCK)
			return -1;
	}
}

static void badsemawakeup(void);

#pragma textflag NOSPLIT
void
runtime·semawakeup(M *mp)
{
	uint32 ret;
	void *oldptr;
	uint32 oldscalar;
	void (*fn)(void);

	// spin-mutex lock
	while(runtime·xchg(&mp->waitsemalock, 1))
		runtime·osyield();
	mp->waitsemacount++;
	ret = runtime·thrwakeup(&mp->waitsemacount, 1);
	if(ret != 0 && ret != ESRCH) {
		// semawakeup can be called on signal stack.
		// Save old ptrarg/scalararg so we can restore them.
		oldptr = g->m->ptrarg[0];
		oldscalar = g->m->scalararg[0];
		g->m->ptrarg[0] = mp;
		g->m->scalararg[0] = ret;
		fn = badsemawakeup;
		if(g == g->m->gsignal)
			fn();
		else
			runtime·onM(&fn);
		g->m->ptrarg[0] = oldptr;
		g->m->scalararg[0] = oldscalar;
	}
	// spin-mutex unlock
	runtime·atomicstore(&mp->waitsemalock, 0);
}

static void
badsemawakeup(void)
{
	M *mp;
	int32 ret;

	mp = g->m->ptrarg[0];
	g->m->ptrarg[0] = nil;
	ret = g->m->scalararg[0];
	g->m->scalararg[0] = 0;

	runtime·printf("thrwakeup addr=%p sem=%d ret=%d\n", &mp->waitsemacount, mp->waitsemacount, ret);
}

void
runtime·newosproc(M *mp, void *stk)
{
	TforkT param;
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
	ret = runtime·tfork(&param, sizeof(param), mp, mp->g0, runtime·mstart);
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

#pragma textflag NOSPLIT
void
runtime·get_random_data(byte **rnd, int32 *rnd_len)
{
	#pragma dataflag NOPTR
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
	mp->gsignal->m = mp;
}

// Called to initialize a new m (including the bootstrap m).
// Called on the new thread, can not allocate memory.
void
runtime·minit(void)
{
	// Initialize signal handling
	runtime·signalstack((byte*)g->m->gsignal->stack.lo, 32*1024);
	runtime·sigprocmask(SIG_SETMASK, sigset_none);
}

// Called from dropm to undo the effect of an minit.
void
runtime·unminit(void)
{
	runtime·signalstack(nil, 0);
}

uintptr
runtime·memlimit(void)
{
	return 0;
}

extern void runtime·sigtramp(void);

typedef struct sigaction {
	union {
		void    (*__sa_handler)(int32);
		void    (*__sa_sigaction)(int32, Siginfo*, void *);
	} __sigaction_u;		/* signal handler */
	uint32	sa_mask;		/* signal mask to apply */
	int32	sa_flags;		/* see signal options below */
} SigactionT;

void
runtime·setsig(int32 i, GoSighandler *fn, bool restart)
{
	SigactionT sa;

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
	SigactionT sa;

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

void
runtime·unblocksignals(void)
{
	runtime·sigprocmask(SIG_SETMASK, sigset_none);
}

#pragma textflag NOSPLIT
int8*
runtime·signame(int32 sig)
{
	return runtime·sigtab[sig].name;
}
