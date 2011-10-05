// Use of this source file is governed by a BSD-style
// license that can be found in the LICENSE file.`

#include "runtime.h"
#include "defs.h"
#include "os.h"
#include "stack.h"

extern SigTab runtime·sigtab[];

extern int64 runtime·rfork_thread(int32 flags, void *stack, M *m, G *g, void (*fn)(void));

enum
{
	ENOTSUP = 91,
};

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

// Basic spinlocks using CAS. We can improve on these later.
static void
lock(Lock *l)
{
	for(;;) {
		if(runtime·cas(&l->key, 0, 1))
			return;
		runtime·osyield();
	}
}

static void
unlock(Lock *l)
{
	uint32 v;

	for (;;) {
		v = l->key;
		if((v&1) == 0)
			runtime·throw("unlock of unlocked lock");
		if(runtime·cas(&l->key, v, 0))
			break;
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
