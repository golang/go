// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Semaphore implementation exposed to Go.
// Intended use is provide a sleep and wakeup
// primitive that can be used in the contended case
// of other synchronization primitives.
// Thus it targets the same goal as Linux's futex,
// but it has much simpler semantics.
//
// That is, don't think of these as semaphores.
// Think of them as a way to implement sleep and wakeup
// such that every sleep is paired with a single wakeup,
// even if, due to races, the wakeup happens before the sleep.
//
// See Mullender and Cox, ``Semaphores in Plan 9,''
// http://swtch.com/semaphore.pdf

#include "runtime.h"

typedef struct Sema Sema;
struct Sema
{
	uint32 *addr;
	G *g;
	Sema *prev;
	Sema *next;
};

// TODO: For now, a linked list; maybe a hash table of linked lists later.
static Sema *semfirst, *semlast;
static Lock semlock;

static void
semqueue(uint32 *addr, Sema *s)
{
	s->addr = addr;
	s->g = nil;

	lock(&semlock);
	s->prev = semlast;
	s->next = nil;
	if(semlast)
		semlast->next = s;
	else
		semfirst = s;
	semlast = s;
	unlock(&semlock);
}

static void
semdequeue(Sema *s)
{
	lock(&semlock);
	if(s->next)
		s->next->prev = s->prev;
	else
		semlast = s->prev;
	if(s->prev)
		s->prev->next = s->next;
	else
		semfirst = s->next;
	s->prev = nil;
	s->next = nil;
	unlock(&semlock);
}

static void
semwakeup(uint32 *addr)
{
	Sema *s;

	lock(&semlock);
	for(s=semfirst; s; s=s->next) {
		if(s->addr == addr && s->g) {
			ready(s->g);
			s->g = nil;
			break;
		}
	}
	unlock(&semlock);
}

// Step 1 of sleep: make ourselves available for wakeup.
// TODO(rsc): Maybe we can write a version without
// locks by using cas on s->g.  Maybe not: I need to
// think more about whether it would be correct.
static void
semsleep1(Sema *s)
{
	lock(&semlock);
	s->g = g;
	unlock(&semlock);
}

// Decided not to go through with it: undo step 1.
static void
semsleepundo1(Sema *s)
{
	lock(&semlock);
	if(s->g != nil) {
		s->g = nil;	// back ourselves out
	} else {
		// If s->g == nil already, semwakeup
		// already readied us.  Since we never stopped
		// running, readying us just set g->readyonstop.
		// Clear it.
		if(g->readyonstop == 0)
			*(int32*)0x555 = 555;
		g->readyonstop = 0;
	}
	unlock(&semlock);
}

// Step 2: wait for the wakeup.
static void
semsleep2(Sema *s)
{
	USED(s);
	g->status = Gwaiting;
	sys·Gosched();
}

static int32
cansemacquire(uint32 *addr)
{
	uint32 v;

	while((v = *addr) > 0)
		if(cas(addr, v, v-1))
			return 1;
	return 0;
}

// func sync.semacquire(addr *uint32)
// For now has no return value.
// Might return an ok (not interrupted) bool in the future?
void
sync·semacquire(uint32 *addr)
{
	Sema s;

	// Easy case.
	if(cansemacquire(addr))
		return;

	// Harder case:
	//	queue
	//	try semacquire one more time, sleep if failed
	//	dequeue
	//	wake up one more guy to avoid races (TODO(rsc): maybe unnecessary?)
	semqueue(addr, &s);
	for(;;) {
		semsleep1(&s);
		if(cansemacquire(addr)) {
			semsleepundo1(&s);
			break;
		}
		semsleep2(&s);
	}
	semdequeue(&s);
	semwakeup(addr);
}

// func sync.semrelease(addr *uint32)
void
sync·semrelease(uint32 *addr)
{
	uint32 v;

	for(;;) {
		v = *addr;
		if(cas(addr, v, v+1))
			break;
	}
	semwakeup(addr);
}
