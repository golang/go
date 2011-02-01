// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "cgocall.h"

void *initcgo;	/* filled in by dynamic linker when Cgo is available */
int64 ncgocall;
void runtime·entersyscall(void);
void runtime·exitsyscall(void);

void
runtime·cgocall(void (*fn)(void*), void *arg)
{
	G *oldlock;

	if(!runtime·iscgo)
		runtime·throw("cgocall unavailable");

	if(fn == 0)
		runtime·throw("cgocall nil");

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
	runtime·entersyscall();
	runtime·runcgo(fn, arg);
	runtime·exitsyscall();

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
uintptr
runtime·cgocallback(void (*fn)(void), void *arg, int32 argsize)
{
	Gobuf oldsched, oldg1sched;
	G *g1;
	void *sp;
	uintptr ret;

	if(g != m->g0)
		runtime·throw("bad g in cgocallback");

	g1 = m->curg;
	oldsched = m->sched;
	oldg1sched = g1->sched;

	runtime·startcgocallback(g1);

	sp = g1->sched.sp - argsize;
	if(sp < g1->stackguard - StackGuard + 4) // +4 for return address
		runtime·throw("g stack overflow in cgocallback");
	runtime·mcpy(sp, arg, argsize);

	ret = runtime·runcgocallback(g1, sp, fn);

	runtime·mcpy(arg, sp, argsize);

	runtime·endcgocallback(g1);

	m->sched = oldsched;
	g1->sched = oldg1sched;

	return ret;
}

void
runtime·Cgocalls(int64 ret)
{
	ret = ncgocall;
	FLUSH(&ret);
}

void (*_cgo_malloc)(void*);
void (*_cgo_free)(void*);

void*
runtime·cmalloc(uintptr n)
{
	struct {
		uint64 n;
		void *ret;
	} a;

	a.n = n;
	a.ret = nil;
	runtime·cgocall(_cgo_malloc, &a);
	return a.ret;
}

void
runtime·cfree(void *p)
{
	runtime·cgocall(_cgo_free, p);
}

