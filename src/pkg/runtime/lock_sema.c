// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

enum
{
	LOCKED = 1,

	ACTIVE_SPIN = 4,
	ACTIVE_SPIN_CNT = 30,
	PASSIVE_SPIN = 1,
};

// creates per-M semaphore (must not return 0)
uintptr	runtime·semacreate(void);
// acquires per-M semaphore 
void	runtime·semasleep(void);
// releases mp's per-M semaphore
void	runtime·semawakeup(M *mp);

void
runtime·lock(Lock *l)
{
	uintptr v;
	uint32 i, spin;

	if(m->locks++ < 0)
		runtime·throw("runtime·lock: lock count");

	// Speculative grab for lock.
	if(runtime·casp(&l->waitm, nil, (void*)LOCKED))
		return;

	if(m->waitsema == 0)
		m->waitsema = runtime·semacreate();
	
	// On uniprocessor's, no point spinning.
	// On multiprocessors, spin for ACTIVE_SPIN attempts.
	spin = 0;
	if(runtime·ncpu > 1)
		spin = ACTIVE_SPIN;
	
	for(i=0;; i++) {
		v = (uintptr)runtime·atomicloadp(&l->waitm);
		if((v&LOCKED) == 0) {
unlocked:
			if(runtime·casp(&l->waitm, (void*)v, (void*)(v|LOCKED)))
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
				if(runtime·casp(&l->waitm, (void*)v, (void*)((uintptr)m|LOCKED)))
					break;
				v = (uintptr)runtime·atomicloadp(&l->waitm);
				if((v&LOCKED) == 0)
					goto unlocked;
			}
			if(v&LOCKED) {
				// Wait.
				runtime·semasleep();
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
		v = (uintptr)runtime·atomicloadp(&l->waitm);
		if(v == LOCKED) {
			if(runtime·casp(&l->waitm, (void*)LOCKED, nil))
				break;
		} else {
			// Other M's are waiting for the lock.
			// Dequeue an M.
			mp = (void*)(v&~LOCKED);
			if(runtime·casp(&l->waitm, (void*)v, mp->nextwaitm)) {
				// Wake that M.
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
	n->waitm = nil;
}

void
runtime·notewakeup(Note *n)
{
	if(runtime·casp(&n->waitm, nil, (void*)LOCKED))
		return;
	runtime·semawakeup(n->waitm);
}

void
runtime·notesleep(Note *n)
{
	if(m->waitsema == 0)
		m->waitsema = runtime·semacreate();
	if(runtime·casp(&n->waitm, nil, m))
		runtime·semasleep();
}
