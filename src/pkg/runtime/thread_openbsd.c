// Use of this source file is governed by a BSD-style
// license that can be found in the LICENSE file.`

#include "runtime.h"
#include "defs_GOOS_GOARCH.h"
#include "os_GOOS.h"
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

static Sigset sigset_all = ~(Sigset)0;
static Sigset sigset_none;

extern int64 runtime·tfork_thread(void *param, void *stack, M *m, G *g, void (*fn)(void));
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
				ts.tv_sec = ns/1000000000LL;
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
runtime·newosproc(M *m, G *g, void *stk, void (*fn)(void))
{
	Tfork param;
	Sigset oset;
	int32 ret;

	if(0) {
		runtime·printf(
			"newosproc stk=%p m=%p g=%p fn=%p id=%d/%d ostk=%p\n",
			stk, m, g, fn, m->id, m->tls[0], &m);
	}

	m->tls[0] = m->id;	// so 386 asm can find it

	param.tf_tcb = (byte*)&m->tls[0];
	param.tf_tid = (int32*)&m->procid;
	param.tf_flags = (int32)0;

	oset = runtime·sigprocmask(SIG_SETMASK, sigset_all);
	ret = runtime·tfork_thread((byte*)&param, stk, m, g, fn);
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
runtime·goenvs(void)
{
	runtime·goenvs_unix();
}

// Called to initialize a new m (including the bootstrap m).
void
runtime·minit(void)
{
	// Initialize signal handling
	m->gsignal = runtime·malg(32*1024);
	runtime·signalstack((byte*)m->gsignal->stackguard - StackGuard, 32*1024);
	runtime·sigprocmask(SIG_SETMASK, sigset_none);
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

static int8 badsignal[] = "runtime: signal received on thread not created by Go.\n";

// This runs on a foreign stack, without an m or a g.  No stack split.
#pragma textflag 7
void
runtime·badsignal(int32 sig)
{
	if (sig == SIGPROF) {
		return;  // Ignore SIGPROFs intended for a non-Go thread.
	}
	runtime·write(2, badsignal, sizeof badsignal - 1);
	runtime·exit(1);
}
