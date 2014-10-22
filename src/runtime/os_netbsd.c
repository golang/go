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
	ENOTSUP = 91,

	// From NetBSD's <sys/time.h>
	CLOCK_REALTIME = 0,
	CLOCK_VIRTUAL = 1,
	CLOCK_PROF = 2,
	CLOCK_MONOTONIC = 3
};

extern SigTab runtime·sigtab[];

static Sigset sigset_none;
static Sigset sigset_all = { ~(uint32)0, ~(uint32)0, ~(uint32)0, ~(uint32)0, };

extern void runtime·getcontext(UcontextT *context);
extern int32 runtime·lwp_create(UcontextT *context, uintptr flags, void *lwpid);
extern void runtime·lwp_mcontext_init(void *mc, void *stack, M *mp, G *gp, void (*fn)(void));
extern int32 runtime·lwp_park(Timespec *abstime, int32 unpark, void *hint, void *unparkhint);
extern int32 runtime·lwp_unpark(int32 lwp, void *hint);
extern int32 runtime·lwp_self(void);

// From NetBSD's <sys/sysctl.h>
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

static void
semasleep(void)
{
	int64 ns;
	Timespec ts;

	ns = (int64)(uint32)g->m->scalararg[0] | (int64)(uint32)g->m->scalararg[1]<<32;
	g->m->scalararg[0] = 0;
	g->m->scalararg[1] = 0;

	// spin-mutex lock
	while(runtime·xchg(&g->m->waitsemalock, 1))
		runtime·osyield();

	for(;;) {
		// lock held
		if(g->m->waitsemacount == 0) {
			// sleep until semaphore != 0 or timeout.
			// thrsleep unlocks m->waitsemalock.
			if(ns < 0) {
				// TODO(jsing) - potential deadlock!
				//
				// There is a potential deadlock here since we
				// have to release the waitsemalock mutex
				// before we call lwp_park() to suspend the
				// thread. This allows another thread to
				// release the lock and call lwp_unpark()
				// before the thread is actually suspended.
				// If this occurs the current thread will end
				// up sleeping indefinitely. Unfortunately
				// the NetBSD kernel does not appear to provide
				// a mechanism for unlocking the userspace
				// mutex once the thread is actually parked.
				runtime·atomicstore(&g->m->waitsemalock, 0);
				runtime·lwp_park(nil, 0, &g->m->waitsemacount, nil);
			} else {
				ns = ns + runtime·nanotime();
				// NOTE: tv_nsec is int64 on amd64, so this assumes a little-endian system.
				ts.tv_nsec = 0;
				ts.tv_sec = runtime·timediv(ns, 1000000000, (int32*)&ts.tv_nsec);
				// TODO(jsing) - potential deadlock!
				// See above for details.
				runtime·atomicstore(&g->m->waitsemalock, 0);
				runtime·lwp_park(&ts, 0, &g->m->waitsemacount, nil);
			}
			// reacquire lock
			while(runtime·xchg(&g->m->waitsemalock, 1))
				runtime·osyield();
		}

		// lock held (again)
		if(g->m->waitsemacount != 0) {
			// semaphore is available.
			g->m->waitsemacount--;
			// spin-mutex unlock
			runtime·atomicstore(&g->m->waitsemalock, 0);
			g->m->scalararg[0] = 0; // semaphore acquired
			return;
		}

		// semaphore not available.
		// if there is a timeout, stop now.
		// otherwise keep trying.
		if(ns >= 0)
			break;
	}

	// lock held but giving up
	// spin-mutex unlock
	runtime·atomicstore(&g->m->waitsemalock, 0);
	g->m->scalararg[0] = -1;
	return;
}

#pragma textflag NOSPLIT
int32
runtime·semasleep(int64 ns)
{
	int32 r;
	void (*fn)(void);

	g->m->scalararg[0] = (uint32)ns;
	g->m->scalararg[1] = (uint32)(ns>>32);
	fn = semasleep;
	runtime·onM(&fn);
	r = g->m->scalararg[0];
	g->m->scalararg[0] = 0;
	return r;
}

static void badsemawakeup(void);

#pragma textflag NOSPLIT
void
runtime·semawakeup(M *mp)
{
	uint32 ret;
	void (*fn)(void);
	void *oldptr;
	uintptr oldscalar;

	// spin-mutex lock
	while(runtime·xchg(&mp->waitsemalock, 1))
		runtime·osyield();
	mp->waitsemacount++;
	// TODO(jsing) - potential deadlock, see semasleep() for details.
	// Confirm that LWP is parked before unparking...
	ret = runtime·lwp_unpark(mp->procid, &mp->waitsemacount);
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
	UcontextT uc;
	int32 ret;

	if(0) {
		runtime·printf(
			"newosproc stk=%p m=%p g=%p id=%d/%d ostk=%p\n",
			stk, mp, mp->g0, mp->id, (int32)mp->tls[0], &mp);
	}

	mp->tls[0] = mp->id;	// so 386 asm can find it

	runtime·getcontext(&uc);
	
	uc.uc_flags = _UC_SIGMASK | _UC_CPU;
	uc.uc_link = nil;
	uc.uc_sigmask = sigset_all;

	runtime·lwp_mcontext_init(&uc.uc_mcontext, stk, mp, mp->g0, runtime·mstart);

	ret = runtime·lwp_create(&uc, 0, &mp->procid);

	if(ret < 0) {
		runtime·printf("runtime: failed to create new OS thread (have %d already; errno=%d)\n", runtime·mcount() - 1, -ret);
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
	g->m->procid = runtime·lwp_self();

	// Initialize signal handling
	runtime·signalstack((byte*)g->m->gsignal->stack.lo, 32*1024);
	runtime·sigprocmask(SIG_SETMASK, &sigset_none, nil);
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
		void    (*_sa_handler)(int32);
		void    (*_sa_sigaction)(int32, Siginfo*, void *);
	} _sa_u;			/* signal handler */
	uint32	sa_mask[4];		/* signal mask to apply */
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
	sa.sa_mask[0] = ~0U;
	sa.sa_mask[1] = ~0U;
	sa.sa_mask[2] = ~0U;
	sa.sa_mask[3] = ~0U;
	if (fn == runtime·sighandler)
		fn = (void*)runtime·sigtramp;
	sa._sa_u._sa_sigaction = (void*)fn;
	runtime·sigaction(i, &sa, nil);
}

GoSighandler*
runtime·getsig(int32 i)
{
	SigactionT sa;

	runtime·memclr((byte*)&sa, sizeof sa);
	runtime·sigaction(i, nil, &sa);
	if((void*)sa._sa_u._sa_sigaction == runtime·sigtramp)
		return runtime·sighandler;
	return (void*)sa._sa_u._sa_sigaction;
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
	runtime·sigprocmask(SIG_SETMASK, &sigset_none, nil);
}

#pragma textflag NOSPLIT
int8*
runtime·signame(int32 sig)
{
	return runtime·sigtab[sig].name;
}
