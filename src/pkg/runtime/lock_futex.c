// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build dragonfly freebsd linux

#include "runtime.h"
#include "stack.h"
#include "../../cmd/ld/textflag.h"

// This implementation depends on OS-specific implementations of
//
//	runtime·futexsleep(uint32 *addr, uint32 val, int64 ns)
//		Atomically,
//			if(*addr == val) sleep
//		Might be woken up spuriously; that's allowed.
//		Don't sleep longer than ns; ns < 0 means forever.
//
//	runtime·futexwakeup(uint32 *addr, uint32 cnt)
//		If any procs are sleeping on addr, wake up at most cnt.

enum
{
	MUTEX_UNLOCKED = 0,
	MUTEX_LOCKED = 1,
	MUTEX_SLEEPING = 2,

	ACTIVE_SPIN = 4,
	ACTIVE_SPIN_CNT = 30,
	PASSIVE_SPIN = 1,
};

// Possible lock states are MUTEX_UNLOCKED, MUTEX_LOCKED and MUTEX_SLEEPING.
// MUTEX_SLEEPING means that there is presumably at least one sleeping thread.
// Note that there can be spinning threads during all states - they do not
// affect mutex's state.
void
runtime·lock(Lock *l)
{
	uint32 i, v, wait, spin;

	if(m->locks++ < 0)
		runtime·throw("runtime·lock: lock count");

	// Speculative grab for lock.
	v = runtime·xchg((uint32*)&l->key, MUTEX_LOCKED);
	if(v == MUTEX_UNLOCKED)
		return;

	// wait is either MUTEX_LOCKED or MUTEX_SLEEPING
	// depending on whether there is a thread sleeping
	// on this mutex.  If we ever change l->key from
	// MUTEX_SLEEPING to some other value, we must be
	// careful to change it back to MUTEX_SLEEPING before
	// returning, to ensure that the sleeping thread gets
	// its wakeup call.
	wait = v;

	// On uniprocessor's, no point spinning.
	// On multiprocessors, spin for ACTIVE_SPIN attempts.
	spin = 0;
	if(runtime·ncpu > 1)
		spin = ACTIVE_SPIN;

	for(;;) {
		// Try for lock, spinning.
		for(i = 0; i < spin; i++) {
			while(l->key == MUTEX_UNLOCKED)
				if(runtime·cas((uint32*)&l->key, MUTEX_UNLOCKED, wait))
					return;
			runtime·procyield(ACTIVE_SPIN_CNT);
		}

		// Try for lock, rescheduling.
		for(i=0; i < PASSIVE_SPIN; i++) {
			while(l->key == MUTEX_UNLOCKED)
				if(runtime·cas((uint32*)&l->key, MUTEX_UNLOCKED, wait))
					return;
			runtime·osyield();
		}

		// Sleep.
		v = runtime·xchg((uint32*)&l->key, MUTEX_SLEEPING);
		if(v == MUTEX_UNLOCKED)
			return;
		wait = MUTEX_SLEEPING;
		runtime·futexsleep((uint32*)&l->key, MUTEX_SLEEPING, -1);
	}
}

void
runtime·unlock(Lock *l)
{
	uint32 v;

	v = runtime·xchg((uint32*)&l->key, MUTEX_UNLOCKED);
	if(v == MUTEX_UNLOCKED)
		runtime·throw("unlock of unlocked lock");
	if(v == MUTEX_SLEEPING)
		runtime·futexwakeup((uint32*)&l->key, 1);

	if(--m->locks < 0)
		runtime·throw("runtime·unlock: lock count");
	if(m->locks == 0 && g->preempt)  // restore the preemption request in case we've cleared it in newstack
		g->stackguard0 = StackPreempt;
}

// One-time notifications.
void
runtime·noteclear(Note *n)
{
	n->key = 0;
}

void
runtime·notewakeup(Note *n)
{
	uint32 old;

	old = runtime·xchg((uint32*)&n->key, 1);
	if(old != 0) {
		runtime·printf("notewakeup - double wakeup (%d)\n", old);
		runtime·throw("notewakeup - double wakeup");
	}
	runtime·futexwakeup((uint32*)&n->key, 1);
}

void
runtime·notesleep(Note *n)
{
	if(g != m->g0)
		runtime·throw("notesleep not on g0");
	while(runtime·atomicload((uint32*)&n->key) == 0) {
		m->blocked = true;
		runtime·futexsleep((uint32*)&n->key, 0, -1);
		m->blocked = false;
	}
}

#pragma textflag NOSPLIT
static bool
notetsleep(Note *n, int64 ns, int64 deadline, int64 now)
{
	// Conceptually, deadline and now are local variables.
	// They are passed as arguments so that the space for them
	// does not count against our nosplit stack sequence.

	if(ns < 0) {
		while(runtime·atomicload((uint32*)&n->key) == 0) {
			m->blocked = true;
			runtime·futexsleep((uint32*)&n->key, 0, -1);
			m->blocked = false;
		}
		return true;
	}

	if(runtime·atomicload((uint32*)&n->key) != 0)
		return true;

	deadline = runtime·nanotime() + ns;
	for(;;) {
		m->blocked = true;
		runtime·futexsleep((uint32*)&n->key, 0, ns);
		m->blocked = false;
		if(runtime·atomicload((uint32*)&n->key) != 0)
			break;
		now = runtime·nanotime();
		if(now >= deadline)
			break;
		ns = deadline - now;
	}
	return runtime·atomicload((uint32*)&n->key) != 0;
}

bool
runtime·notetsleep(Note *n, int64 ns)
{
	bool res;

	if(g != m->g0 && !m->gcing)
		runtime·throw("notetsleep not on g0");

	res = notetsleep(n, ns, 0, 0);
	return res;
}

// same as runtime·notetsleep, but called on user g (not g0)
// calls only nosplit functions between entersyscallblock/exitsyscall
bool
runtime·notetsleepg(Note *n, int64 ns)
{
	bool res;

	if(g == m->g0)
		runtime·throw("notetsleepg on g0");

	runtime·entersyscallblock();
	res = notetsleep(n, ns, 0, 0);
	runtime·exitsyscall();
	return res;
}
