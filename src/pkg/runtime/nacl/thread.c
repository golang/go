// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "defs.h"
#include "os.h"

int8 *goos = "nacl";

// Thread-safe allocation of a mutex.
// (The name sema is left over from the Darwin implementation.
// Native Client implements semaphores too, but it is just a shim
// over the host implementation, which on some hosts imposes a very
// low limit on how many semaphores can be created.)
//
// Psema points at a mutex descriptor.
// It starts out zero, meaning no mutex.
// Fill it in, being careful of others calling initsema
// simultaneously.
static void
initsema(uint32 *psema)
{
	uint32 sema;

	if(*psema != 0)	// already have one
		return;

	sema = mutex_create();
	if((int32)sema < 0) {
		printf("mutex_create failed\n");
		breakpoint();
	}
	// mutex_create returns a file descriptor;
	// shift it up and add the 1 bit so that can
	// distinguish unintialized from fd 0.
	sema = (sema<<1) | 1;
	if(!cas(psema, 0, sema)){
		// Someone else filled it in.  Use theirs.
		close(sema);
		return;
	}
}

// Lock and unlock.
// Defer entirely to Native Client.
// The expense of a call into Native Client is more like
// a function call than a system call, so as long as the
// Native Client lock implementation is good, we can't
// do better ourselves.

static void
xlock(int32 fd)
{
	if(mutex_lock(fd) < 0) {
		printf("mutex_lock failed\n");
		breakpoint();
	}
}

static void
xunlock(int32 fd)
{
	if(mutex_unlock(fd) < 0) {
		printf("mutex_lock failed\n");
		breakpoint();
	}
}

void
lock(Lock *l)
{
	if(m->locks < 0)
		throw("lock count");
	m->locks++;
	if(l->sema == 0)
		initsema(&l->sema);
	xlock(l->sema>>1);
}

void
unlock(Lock *l)
{
	m->locks--;
	if(m->locks < 0)
		throw("lock count");
	xunlock(l->sema>>1);
}

void
destroylock(Lock *l)
{
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
//
// Native Client does not require that the thread acquiring
// a lock be the thread that releases the lock, so this is safe.

void
noteclear(Note *n)
{
	if(n->lock.sema == 0)
		initsema(&n->lock.sema);
	xlock(n->lock.sema>>1);
}

void
notewakeup(Note *n)
{
	if(n->lock.sema == 0) {
		printf("notewakeup without noteclear");
		breakpoint();
	}
	xunlock(n->lock.sema>>1);
}

void
notesleep(Note *n)
{
	if(n->lock.sema == 0) {
		printf("notesleep without noteclear");
		breakpoint();
	}
	xlock(n->lock.sema>>1);
	xunlock(n->lock.sema>>1);	// Let other sleepers find out too.
}

void
newosproc(M *m, G *g, void *stk, void (*fn)(void))
{
	void **vstk;

	// I wish every OS made thread creation this easy.
	m->tls[0] = (uint32)g;
	m->tls[1] = (uint32)m;
	vstk = stk;
	*--vstk = nil;
	if(thread_create(fn, vstk, m->tls, sizeof m->tls) < 0) {
		printf("thread_create failed\n");
		breakpoint();
	}
}

void
osinit(void)
{
}

// Called to initialize a new m (including the bootstrap m).
void
minit(void)
{
}
