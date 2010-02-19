// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

int8 *goos = "tiny";

void
minit(void)
{
}

void
osinit(void)
{
}

void
initsig(void)
{
}

void
exit(int32)
{
	for(;;);
}

// single processor, no interrupts,
// so no need for real concurrency or atomicity

void
newosproc(M *m, G *g, void *stk, void (*fn)(void))
{
	USED(m, g, stk, fn);
	throw("newosproc");
}

void
lock(Lock *l)
{
	if(m->locks < 0)
		throw("lock count");
	m->locks++;
	if(l->key != 0)
		throw("deadlock");
	l->key = 1;
}

void
unlock(Lock *l)
{
	m->locks--;
	if(m->locks < 0)
		throw("lock count");
	if(l->key != 1)
		throw("unlock of unlocked lock");
	l->key = 0;
}

void
noteclear(Note *n)
{
	n->lock.key = 0;
}

void
notewakeup(Note *n)
{
	n->lock.key = 1;
}

void
notesleep(Note *n)
{
	if(n->lock.key != 1)
		throw("notesleep");
}

