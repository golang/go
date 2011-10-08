// Use of this source file is governed by a BSD-style
// license that can be found in the LICENSE file.`

#include "runtime.h"
#include "defs.h"
#include "os.h"
#include "stack.h"

enum
{
	MUTEX_UNLOCKED = 0,
	MUTEX_LOCKED = 1,
	MUTEX_SLEEPING = 2,

	ACTIVE_SPIN = 4,
	ACTIVE_SPIN_CNT = 30,
	PASSIVE_SPIN = 1,

	ESRCH = 3,
	ENOTSUP = 91,
};

extern SigTab runtime·sigtab[];

extern int64 runtime·rfork_thread(int32 flags, void *stack, M *m, G *g, void (*fn)(void));
extern int32 runtime·thrsleep(void *, void *, void*, void *);
extern int32 runtime·thrwakeup(void *, int32);

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

// Possible lock states are MUTEX_UNLOCKED, MUTEX_LOCKED and MUTEX_SLEEPING.
// MUTEX_SLEEPING means that there is potentially at least one sleeping thread.
// Note that there can be spinning threads during all states - they do not
// affect the mutex's state.
static void
lock(Lock *l)
{
	uint32 i, v, wait, spin;
	int32 ret;

	// Speculative grab for lock.
	v = runtime·xchg(&l->key, MUTEX_LOCKED);
	if(v == MUTEX_UNLOCKED)
		return;

	// If we ever change the lock from MUTEX_SLEEPING to some other value,
	// we must be careful to change it back to MUTEX_SLEEPING before
	// returning, to ensure that the sleeping thread gets its wakeup call.
	wait = v;

	// No point spinning unless there are multiple processors.
	spin = 0;
	if(runtime·ncpu > 1)
		spin = ACTIVE_SPIN;

	for(;;) {
		// Try for lock, spinning.
		for(i = 0; i < spin; i++) {
			while(l->key == MUTEX_UNLOCKED)
				if(runtime·cas(&l->key, MUTEX_UNLOCKED, wait))
					return;
			runtime·procyield(ACTIVE_SPIN_CNT);
		}

		// Try for lock, rescheduling.
		for(i = 0; i < PASSIVE_SPIN; i++) {
			while(l->key == MUTEX_UNLOCKED)
				if(runtime·cas(&l->key, MUTEX_UNLOCKED, wait))
					return;
			runtime·osyield();
		}

		// Grab a lock on sema and sleep - sema will be unlocked by
		// thrsleep() and we'll get woken by another thread.
		// Note that thrsleep unlocks on a _spinlock_lock_t which is
		// an int on amd64, so we need to be careful here.
		while (!runtime·cas(&l->sema, MUTEX_UNLOCKED, MUTEX_LOCKED))
			runtime·osyield();
		v = runtime·xchg(&l->key, MUTEX_SLEEPING);
		if(v == MUTEX_UNLOCKED) {
			l->sema = MUTEX_UNLOCKED;
			return;
		}
		wait = v;
		ret = runtime·thrsleep(&l->key, 0, 0, &l->sema);
		if (ret != 0) {
			runtime·printf("thrsleep addr=%p sema=%d ret=%d\n",
				&l->key, l->sema, ret);
			l->sema = MUTEX_UNLOCKED;
		}
	}
}

static void
unlock(Lock *l)
{
	uint32 v, ret;

	while (!runtime·cas(&l->sema, MUTEX_UNLOCKED, MUTEX_LOCKED))
		runtime·osyield();
	v = runtime·xchg(&l->key, MUTEX_UNLOCKED);
	l->sema = MUTEX_UNLOCKED;
	if(v == MUTEX_UNLOCKED)
		runtime·throw("unlock of unlocked lock");
	if(v == MUTEX_SLEEPING) {
		ret = runtime·thrwakeup(&l->key, 0);
		if (ret != 0 && ret != ESRCH) {
			runtime·printf("thrwakeup addr=%p sem=%d ret=%d\n",
				&l->key, l->sema, ret);
		}
	}
}

void
runtime·lock(Lock *l)
{
	if(m->locks < 0)
		runtime·throw("lock count");
	m->locks++;
	lock(l);
}

void 
runtime·unlock(Lock *l)
{
	m->locks--;
	if(m->locks < 0)
		runtime·throw("lock count");
	unlock(l);
}

// Event notifications.
void
runtime·noteclear(Note *n)
{
	n->lock.key = 0;
	lock(&n->lock);
}

void
runtime·notesleep(Note *n)
{
	lock(&n->lock);
	unlock(&n->lock);
}

void
runtime·notewakeup(Note *n)
{
	unlock(&n->lock);
}

// From OpenBSD's sys/param.h
#define	RFPROC		(1<<4)	/* change child (else changes curproc) */
#define	RFMEM		(1<<5)	/* share `address space' */
#define	RFNOWAIT	(1<<6)	/* parent need not wait() on child */
#define	RFTHREAD	(1<<13)	/* create a thread, not a process */

void
runtime·newosproc(M *m, G *g, void *stk, void (*fn)(void))
{
	int32 flags;
	int32 ret;

	flags = RFPROC | RFTHREAD | RFMEM | RFNOWAIT;

	if (0) {
		runtime·printf(
			"newosproc stk=%p m=%p g=%p fn=%p id=%d/%d ostk=%p\n",
			stk, m, g, fn, m->id, m->tls[0], &m);
	}

	m->tls[0] = m->id;	// so 386 asm can find it

	if((ret = runtime·rfork_thread(flags, stk, m, g, fn)) < 0) {
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
	runtime·signalstack(m->gsignal->stackguard - StackGuard, 32*1024);
}

void
runtime·sigpanic(void)
{
	switch(g->sig) {
	case SIGBUS:
		if(g->sigcode0 == BUS_ADRERR && g->sigcode1 < 0x1000)
			runtime·panicstring("invalid memory address or nil pointer dereference");
		runtime·printf("unexpected fault address %p\n", g->sigcode1);
		runtime·throw("fault");
	case SIGSEGV:
		if((g->sigcode0 == 0 || g->sigcode0 == SEGV_MAPERR || g->sigcode0 == SEGV_ACCERR) && g->sigcode1 < 0x1000)
			runtime·panicstring("invalid memory address or nil pointer dereference");
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
