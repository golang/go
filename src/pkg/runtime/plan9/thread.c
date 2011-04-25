// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "os.h"

int8 *goos = "plan9";

void
runtime·minit(void)
{
}

void
runtime·osinit(void)
{
}

void
runtime·goenvs(void)
{
}

void
runtime·initsig(int32 queue)
{
}

void
runtime·exit(int32)
{
	runtime·exits(nil);
}

void
runtime·newosproc(M *m, G *g, void *stk, void (*fn)(void))
{
	USED(m, g, stk, fn);
	
	m->tls[0] = m->id;	// so 386 asm can find it
	if(0){
		runtime·printf("newosproc stk=%p m=%p g=%p fn=%p rfork=%p id=%d/%d ostk=%p\n",
			stk, m, g, fn, runtime·rfork, m->id, m->tls[0], &m);
	}        
	
	if (runtime·rfork(RFPROC | RFMEM, stk, m, g, fn) < 0 )
		runtime·throw("newosproc: rfork failed");
}

// Blocking locks.

// Implement Locks, using semaphores.
// l->key is the number of threads who want the lock.
// In a race, one thread increments l->key from 0 to 1
// and the others increment it from >0 to >1.  The thread
// who does the 0->1 increment gets the lock, and the
// others wait on the semaphore.  When the 0->1 thread
// releases the lock by decrementing l->key, l->key will
// be >0, so it will increment the semaphore to wake up
// one of the others.  This is the same algorithm used
// in Plan 9's user-level locks.

void
runtime·lock(Lock *l)
{
	if(m->locks < 0)
		runtime·throw("lock count");
	m->locks++;
	
	if(runtime·xadd(&l->key, 1) == 1)
		return; // changed from 0 -> 1; we hold lock
	// otherwise wait in kernel
	while(runtime·plan9_semacquire(&l->sema, 1) < 0) {
		/* interrupted; try again */
	}
}

void
runtime·unlock(Lock *l)
{
	m->locks--;
	if(m->locks < 0)
		runtime·throw("lock count");

	if(runtime·xadd(&l->key, -1) == 0)
		return; // changed from 1 -> 0: no contention
	
	runtime·plan9_semrelease(&l->sema, 1);
}


void 
runtime·destroylock(Lock *l)
{
	// nothing
}

// User-level semaphore implementation:
// try to do the operations in user space on u,
// but when it's time to block, fall back on the kernel semaphore k.
// This is the same algorithm used in Plan 9.
void
runtime·usemacquire(Usema *s)
{
	if((int32)runtime·xadd(&s->u, -1) < 0)
		while(runtime·plan9_semacquire(&s->k, 1) < 0) {
			/* interrupted; try again */
		}
}

void
runtime·usemrelease(Usema *s)
{
	if((int32)runtime·xadd(&s->u, 1) <= 0)
		runtime·plan9_semrelease(&s->k, 1);
}


// Event notifications.
void
runtime·noteclear(Note *n)
{
	n->wakeup = 0;
}

void
runtime·notesleep(Note *n)
{
	while(!n->wakeup)
		runtime·usemacquire(&n->sema);
}

void
runtime·notewakeup(Note *n)
{
	n->wakeup = 1;
	runtime·usemrelease(&n->sema);
}

void
os·sigpipe(void)
{
	runtime·throw("too many writes on closed pipe");
}
