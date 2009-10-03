// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "defs.h"
#include "signals.h"
#include "os.h"

// Linux futex.
//
//	futexsleep(uint32 *addr, uint32 val)
//	futexwakeup(uint32 *addr)
//
// Futexsleep atomically checks if *addr == val and if so, sleeps on addr.
// Futexwakeup wakes up one thread sleeping on addr.
// Futexsleep is allowed to wake up spuriously.

enum
{
	FUTEX_WAIT = 0,
	FUTEX_WAKE = 1,

	EINTR = 4,
	EAGAIN = 11,
};

// TODO(rsc): I tried using 1<<40 here but futex woke up (-ETIMEDOUT).
// I wonder if the timespec that gets to the kernel
// actually has two 32-bit numbers in it, so that
// a 64-bit 1<<40 ends up being 0 seconds,
// 1<<8 nanoseconds.
static Timespec longtime =
{
	1<<30,	// 34 years
	0
};

// Atomically,
//	if(*addr == val) sleep
// Might be woken up spuriously; that's allowed.
static void
futexsleep(uint32 *addr, uint32 val)
{
	int32 ret;

	ret = futex(addr, FUTEX_WAIT, val, &longtime, nil, 0);
	if(ret >= 0 || ret == -EAGAIN || ret == -EINTR)
		return;

	prints("futexsleep addr=");
	sys·printpointer(addr);
	prints(" val=");
	sys·printint(val);
	prints(" returned ");
	sys·printint(ret);
	prints("\n");
	*(int32*)0x1005 = 0x1005;
}

// If any procs are sleeping on addr, wake up at least one.
static void
futexwakeup(uint32 *addr)
{
	int64 ret;

	ret = futex(addr, FUTEX_WAKE, 1, nil, nil, 0);

	if(ret >= 0)
		return;

	// I don't know that futex wakeup can return
	// EAGAIN or EINTR, but if it does, it would be
	// safe to loop and call futex again.

	prints("futexwakeup addr=");
	sys·printpointer(addr);
	prints(" returned ");
	sys·printint(ret);
	prints("\n");
	*(int32*)0x1006 = 0x1006;
}


// Lock and unlock.
//
// The lock state is a single 32-bit word that holds
// a 31-bit count of threads waiting for the lock
// and a single bit (the low bit) saying whether the lock is held.
// The uncontended case runs entirely in user space.
// When contention is detected, we defer to the kernel (futex).
//
// A reminder: compare-and-swap cas(addr, old, new) does
//	if(*addr == old) { *addr = new; return 1; }
//	else return 0;
// but atomically.

static void
futexlock(Lock *l)
{
	uint32 v;

again:
	v = l->key;
	if((v&1) == 0){
		if(cas(&l->key, v, v|1)){
			// Lock wasn't held; we grabbed it.
			return;
		}
		goto again;
	}

	// Lock was held; try to add ourselves to the waiter count.
	if(!cas(&l->key, v, v+2))
		goto again;

	// We're accounted for, now sleep in the kernel.
	//
	// We avoid the obvious lock/unlock race because
	// the kernel won't put us to sleep if l->key has
	// changed underfoot and is no longer v+2.
	//
	// We only really care that (v&1) == 1 (the lock is held),
	// and in fact there is a futex variant that could
	// accomodate that check, but let's not get carried away.)
	futexsleep(&l->key, v+2);

	// We're awake: remove ourselves from the count.
	for(;;){
		v = l->key;
		if(v < 2)
			throw("bad lock key");
		if(cas(&l->key, v, v-2))
			break;
	}

	// Try for the lock again.
	goto again;
}

static void
futexunlock(Lock *l)
{
	uint32 v;

	// Atomically get value and clear lock bit.
again:
	v = l->key;
	if((v&1) == 0)
		throw("unlock of unlocked lock");
	if(!cas(&l->key, v, v&~1))
		goto again;

	// If there were waiters, wake one.
	if(v & ~1)
		futexwakeup(&l->key);
}

void
lock(Lock *l)
{
	if(m->locks < 0)
		throw("lock count");
	m->locks++;
	futexlock(l);
}

void
unlock(Lock *l)
{
	m->locks--;
	if(m->locks < 0)
		throw("lock count");
	futexunlock(l);
}


// One-time notifications.
//
// Since the lock/unlock implementation already
// takes care of sleeping in the kernel, we just reuse it.
// (But it's a weird use, so it gets its own interface.)
//
// We use a lock to represent the event:
// unlocked == event has happened.
// Thus the lock starts out locked, and to wait for the
// event you try to lock the lock.  To signal the event,
// you unlock the lock.

void
noteclear(Note *n)
{
	n->lock.key = 0;	// memset(n, 0, sizeof *n)
	futexlock(&n->lock);
}

void
notewakeup(Note *n)
{
	futexunlock(&n->lock);
}

void
notesleep(Note *n)
{
	futexlock(&n->lock);
	futexunlock(&n->lock);	// Let other sleepers find out too.
}


// Clone, the Linux rfork.
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
newosproc(M *m, G *g, void *stk, void (*fn)(void))
{
	int32 ret;
	int32 flags;

	/*
	 * note: strace gets confused if we use CLONE_PTRACE here.
	 */
	flags = CLONE_PARENT	/* getppid doesn't change in child */
		| CLONE_VM	/* share memory */
		| CLONE_FS	/* share cwd, etc */
		| CLONE_FILES	/* share fd table */
		| CLONE_SIGHAND	/* share sig handler table */
		| CLONE_THREAD	/* revisit - okay for now */
		;

	m->tls[0] = m->id;	// so 386 asm can find it
	if(0){
		printf("newosproc stk=%p m=%p g=%p fn=%p clone=%p id=%d/%d ostk=%p\n",
			stk, m, g, fn, clone, m->id, m->tls[0], &m);
	}

	ret = clone(flags, stk, m, g, fn);

	if(ret < 0)
		*(int32*)123 = 123;
}

void
osinit(void)
{
}

// Called to initialize a new m (including the bootstrap m).
void
minit(void)
{
	// Initialize signal handling.
	m->gsignal = malg(32*1024);	// OS X wants >=8K, Linux >=2K
	signalstack(m->gsignal->stackguard, 32*1024);
}
