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

	// From NetBSD's <sys/time.h>
	CLOCK_REALTIME = 0,
	CLOCK_VIRTUAL = 1,
	CLOCK_PROF = 2,
	CLOCK_MONOTONIC = 3
};

extern SigTab runtime·sigtab[];

static Sigset sigset_all = { ~(uint32)0, ~(uint32)0, ~(uint32)0, ~(uint32)0, };
static Sigset sigset_none;

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
				runtime·atomicstore(&m->waitsemalock, 0);
				runtime·lwp_park(nil, 0, &m->waitsemacount, nil);
			} else {
				ns += runtime·nanotime();
				ts.tv_sec = ns/1000000000LL;
				ts.tv_nsec = ns%1000000000LL;
				// TODO(jsing) - potential deadlock!
				// See above for details.
				runtime·atomicstore(&m->waitsemalock, 0);
				runtime·lwp_park(&ts, 0, &m->waitsemacount, nil);
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
	// TODO(jsing) - potential deadlock, see semasleep() for details.
	// Confirm that LWP is parked before unparking...
	ret = runtime·lwp_unpark(mp->procid, &mp->waitsemacount);
	if(ret != 0 && ret != ESRCH)
		runtime·printf("thrwakeup addr=%p sem=%d ret=%d\n", &mp->waitsemacount, mp->waitsemacount, ret);
	// spin-mutex unlock
	runtime·atomicstore(&mp->waitsemalock, 0);
}

void
runtime·newosproc(M *mp, G *gp, void *stk, void (*fn)(void))
{
	UcontextT uc;
	int32 ret;

	if(0) {
		runtime·printf(
			"newosproc stk=%p m=%p g=%p fn=%p id=%d/%d ostk=%p\n",
			stk, mp, gp, fn, mp->id, mp->tls[0], &mp);
	}

	mp->tls[0] = mp->id;	// so 386 asm can find it

	runtime·getcontext(&uc);
	
	uc.uc_flags = _UC_SIGMASK | _UC_CPU;
	uc.uc_link = nil;
	uc.uc_sigmask = sigset_all;

	runtime·lwp_mcontext_init(&uc.uc_mcontext, stk, mp, gp, fn);

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

void
runtime·goenvs(void)
{
	runtime·goenvs_unix();
}

// Called to initialize a new m (including the bootstrap m).
void
runtime·minit(void)
{
	m->procid = runtime·lwp_self();

	// Initialize signal handling
	m->gsignal = runtime·malg(32*1024);
	runtime·signalstack((byte*)m->gsignal->stackguard - StackGuard, 32*1024);
	runtime·sigprocmask(SIG_SETMASK, &sigset_none, nil);
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
	if (sig == SIGPROF) {
		return;  // Ignore SIGPROFs intended for a non-Go thread.
	}
	runtime·write(2, badsignal, sizeof badsignal - 1);
	if (0 <= sig && sig < NSIG) {
		// Call runtime·findnull dynamically to circumvent static stack size check.
		static int32 (*findnull)(byte*) = runtime·findnull;
		runtime·write(2, runtime·sigtab[sig].name, findnull((byte*)runtime·sigtab[sig].name));
	}
	runtime·write(2, "\n", 1);
	runtime·exit(1);
}
