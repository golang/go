// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin netbsd openbsd plan9 windows

#include "runtime.h"

// This implementation depends on OS-specific implementations of
//
//	uintptr runtime·semacreate(void)
//		Create a semaphore, which will be assigned to m->waitsema.
//		The zero value is treated as absence of any semaphore,
//		so be sure to return a non-zero value.
//
//	int32 runtime·semasleep(int64 ns)
//		If ns < 0, acquire m->waitsema and return 0.
//		If ns >= 0, try to acquire m->waitsema for at most ns nanoseconds.
//		Return 0 if the semaphore was acquired, -1 if interrupted or timed out.
//
//	int32 runtime·semawakeup(M *mp)
//		Wake up mp, which is or will soon be sleeping on mp->waitsema.
//

enum
{
	LOCKED = 1,

	ACTIVE_SPIN = 4,
	ACTIVE_SPIN_CNT = 30,
	PASSIVE_SPIN = 1,
};

void
runtime·lock(Lock *l)
{
	uintptr v;
	uint32 i, spin;

	if(m->locks++ < 0)
		runtime·throw("runtime·lock: lock count");

	// Speculative grab for lock.
	if(runtime·casp((void**)&l->key, nil, (void*)LOCKED))
		return;

	if(m->waitsema == 0)
		m->waitsema = runtime·semacreate();

	// On uniprocessor's, no point spinning.
	// On multiprocessors, spin for ACTIVE_SPIN attempts.
	spin = 0;
	if(runtime·ncpu > 1)
		spin = ACTIVE_SPIN;

	for(i=0;; i++) {
		v = (uintptr)runtime·atomicloadp((void**)&l->key);
		if((v&LOCKED) == 0) {
unlocked:
			if(runtime·casp((void**)&l->key, (void*)v, (void*)(v|LOCKED)))
				return;
			i = 0;
		}
		if(i<spin)
			runtime·procyield(ACTIVE_SPIN_CNT);
		else if(i<spin+PASSIVE_SPIN)
			runtime·osyield();
		else {
			// Someone else has it.
			// l->waitm points to a linked list of M's waiting
			// for this lock, chained through m->nextwaitm.
			// Queue this M.
			for(;;) {
				m->nextwaitm = (void*)(v&~LOCKED);
				if(runtime·casp((void**)&l->key, (void*)v, (void*)((uintptr)m|LOCKED)))
					break;
				v = (uintptr)runtime·atomicloadp((void**)&l->key);
				if((v&LOCKED) == 0)
					goto unlocked;
			}
			if(v&LOCKED) {
				// Queued.  Wait.
				runtime·semasleep(-1);
				i = 0;
			}
		}
	}
}

void
runtime·unlock(Lock *l)
{
	uintptr v;
	M *mp;

	if(--m->locks < 0)
		runtime·throw("runtime·unlock: lock count");

	for(;;) {
		v = (uintptr)runtime·atomicloadp((void**)&l->key);
		if(v == LOCKED) {
			if(runtime·casp((void**)&l->key, (void*)LOCKED, nil))
				break;
		} else {
			// Other M's are waiting for the lock.
			// Dequeue an M.
			mp = (void*)(v&~LOCKED);
			if(runtime·casp((void**)&l->key, (void*)v, mp->nextwaitm)) {
				// Dequeued an M.  Wake it.
				runtime·semawakeup(mp);
				break;
			}
		}
	}
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
	M *mp;

	do
		mp = runtime·atomicloadp((void**)&n->key);
	while(!runtime·casp((void**)&n->key, mp, (void*)LOCKED));

	// Successfully set waitm to LOCKED.
	// What was it before?
	if(mp == nil) {
		// Nothing was waiting.  Done.
	} else if(mp == (M*)LOCKED) {
		// Two notewakeups!  Not allowed.
		runtime·throw("notewakeup - double wakeup");
	} else {
		// Must be the waiting m.  Wake it up.
		runtime·semawakeup(mp);
	}
}

void
runtime·notesleep(Note *n)
{
	if(m->waitsema == 0)
		m->waitsema = runtime·semacreate();
	if(!runtime·casp((void**)&n->key, nil, m)) {  // must be LOCKED (got wakeup)
		if(n->key != LOCKED)
			runtime·throw("notesleep - waitm out of sync");
		return;
	}
	// Queued.  Sleep.
	if(m->profilehz > 0)
		runtime·setprof(false);
	runtime·semasleep(-1);
	if(m->profilehz > 0)
		runtime·setprof(true);
}

void
runtime·notetsleep(Note *n, int64 ns)
{
	M *mp;
	int64 deadline, now;

	if(ns < 0) {
		runtime·notesleep(n);
		return;
	}

	if(m->waitsema == 0)
		m->waitsema = runtime·semacreate();

	// Register for wakeup on n->waitm.
	if(!runtime·casp((void**)&n->key, nil, m)) {  // must be LOCKED (got wakeup already)
		if(n->key != LOCKED)
			runtime·throw("notetsleep - waitm out of sync");
		return;
	}

	if(m->profilehz > 0)
		runtime·setprof(false);
	deadline = runtime·nanotime() + ns;
	for(;;) {
		// Registered.  Sleep.
		if(runtime·semasleep(ns) >= 0) {
			// Acquired semaphore, semawakeup unregistered us.
			// Done.
			if(m->profilehz > 0)
				runtime·setprof(true);
			return;
		}

		// Interrupted or timed out.  Still registered.  Semaphore not acquired.
		now = runtime·nanotime();
		if(now >= deadline)
			break;

		// Deadline hasn't arrived.  Keep sleeping.
		ns = deadline - now;
	}

	if(m->profilehz > 0)
		runtime·setprof(true);

	// Deadline arrived.  Still registered.  Semaphore not acquired.
	// Want to give up and return, but have to unregister first,
	// so that any notewakeup racing with the return does not
	// try to grant us the semaphore when we don't expect it.
	for(;;) {
		mp = runtime·atomicloadp((void**)&n->key);
		if(mp == m) {
			// No wakeup yet; unregister if possible.
			if(runtime·casp((void**)&n->key, mp, nil))
				return;
		} else if(mp == (M*)LOCKED) {
			// Wakeup happened so semaphore is available.
			// Grab it to avoid getting out of sync.
			if(runtime·semasleep(-1) < 0)
				runtime·throw("runtime: unable to acquire - semaphore out of sync");
			return;
		} else {
			runtime·throw("runtime: unexpected waitm - semaphore out of sync");
		}
	}
}
