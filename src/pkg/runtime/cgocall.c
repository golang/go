// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "cgocall.h"

void *initcgo;	/* filled in by dynamic linker when Cgo is available */
int64 ncgocall;
void ·entersyscall(void);
void ·exitsyscall(void);

void
cgocall(void (*fn)(void*), void *arg)
{
	G *oldlock;

	if(initcgo == nil)
		throw("cgocall unavailable");

	ncgocall++;

	/*
	 * Lock g to m to ensure we stay on the same stack if we do a
	 * cgo callback.
	 */
	oldlock = m->lockedg;
	m->lockedg = g;
	g->lockedm = m;

	/*
	 * Announce we are entering a system call
	 * so that the scheduler knows to create another
	 * M to run goroutines while we are in the
	 * foreign code.
	 */
	·entersyscall();
	runcgo(fn, arg);
	·exitsyscall();

	m->lockedg = oldlock;
	if(oldlock == nil)
		g->lockedm = nil;

	return;
}

// When a C function calls back into Go, the wrapper function will
// call this.  This switches to a Go stack, copies the arguments
// (arg/argsize) on to the stack, calls the function, copies the
// arguments back where they came from, and finally returns to the old
// stack.
void
cgocallback(void (*fn)(void), void *arg, int32 argsize)
{
	Gobuf oldsched;
	G *g1;
	void *sp;

	if(g != m->g0)
		throw("bad g in cgocallback");

	oldsched = m->sched;

	g1 = m->curg;

	startcgocallback(g1);

	sp = g1->sched.sp - argsize;
	if(sp < g1->stackguard)
		throw("g stack overflow in cgocallback");
	mcpy(sp, arg, argsize);

	runcgocallback(g1, sp, fn);

	mcpy(arg, sp, argsize);

	endcgocallback(g1);

	m->sched = oldsched;
}

void
·Cgocalls(int64 ret)
{
	ret = ncgocall;
	FLUSH(&ret);
}

void (*_cgo_malloc)(void*);
void (*_cgo_free)(void*);

void*
cmalloc(uintptr n)
{
	struct a {
		uint64 n;
		void *ret;
	} a;

	a.n = n;
	a.ret = nil;
	cgocall(_cgo_malloc, &a);
	return a.ret;
}

void
cfree(void *p)
{
	cgocall(_cgo_free, p);
}

