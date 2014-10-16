// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "arch_GOARCH.h"
#include "stack.h"
#include "malloc.h"
#include "textflag.h"

// Code related to defer, panic and recover.

// TODO: remove once code is moved to Go
extern Defer* runtime·newdefer(int32 siz);
extern runtime·freedefer(Defer *d);

uint32 runtime·panicking;
static Mutex paniclk;

void
runtime·deferproc_m(void)
{
	int32 siz;
	FuncVal *fn;
	uintptr argp;
	uintptr callerpc;
	Defer *d;

	siz = g->m->scalararg[0];
	fn = g->m->ptrarg[0];
	argp = g->m->scalararg[1];
	callerpc = g->m->scalararg[2];
	g->m->ptrarg[0] = nil;
	g->m->scalararg[1] = 0;

	d = runtime·newdefer(siz);
	if(d->panic != nil)
		runtime·throw("deferproc: d->panic != nil after newdefer");
	d->fn = fn;
	d->pc = callerpc;
	d->argp = argp;
	runtime·memmove(d+1, (void*)argp, siz);
}

// Unwind the stack after a deferred function calls recover
// after a panic.  Then arrange to continue running as though
// the caller of the deferred function returned normally.
void
runtime·recovery_m(G *gp)
{
	void *argp;
	uintptr pc;
	
	// Info about defer passed in G struct.
	argp = (void*)gp->sigcode0;
	pc = (uintptr)gp->sigcode1;

	// d's arguments need to be in the stack.
	if(argp != nil && ((uintptr)argp < gp->stack.lo || gp->stack.hi < (uintptr)argp)) {
		runtime·printf("recover: %p not in [%p, %p]\n", argp, gp->stack.lo, gp->stack.hi);
		runtime·throw("bad recovery");
	}

	// Make the deferproc for this d return again,
	// this time returning 1.  The calling function will
	// jump to the standard return epilogue.
	// The -2*sizeof(uintptr) makes up for the
	// two extra words that are on the stack at
	// each call to deferproc.
	// (The pc we're returning to does pop pop
	// before it tests the return value.)
	// On the arm there are 2 saved LRs mixed in too.
	if(thechar == '5')
		gp->sched.sp = (uintptr)argp - 4*sizeof(uintptr);
	else
		gp->sched.sp = (uintptr)argp - 2*sizeof(uintptr);
	gp->sched.pc = pc;
	gp->sched.lr = 0;
	gp->sched.ret = 1;
	runtime·gogo(&gp->sched);
}

void
runtime·startpanic_m(void)
{
	if(runtime·mheap.cachealloc.size == 0) { // very early
		runtime·printf("runtime: panic before malloc heap initialized\n");
		g->m->mallocing = 1; // tell rest of panic not to try to malloc
	} else if(g->m->mcache == nil) // can happen if called from signal handler or throw
		g->m->mcache = runtime·allocmcache();
	switch(g->m->dying) {
	case 0:
		g->m->dying = 1;
		if(g != nil) {
			g->writebuf.array = nil;
			g->writebuf.len = 0;
			g->writebuf.cap = 0;
		}
		runtime·xadd(&runtime·panicking, 1);
		runtime·lock(&paniclk);
		if(runtime·debug.schedtrace > 0 || runtime·debug.scheddetail > 0)
			runtime·schedtrace(true);
		runtime·freezetheworld();
		return;
	case 1:
		// Something failed while panicing, probably the print of the
		// argument to panic().  Just print a stack trace and exit.
		g->m->dying = 2;
		runtime·printf("panic during panic\n");
		runtime·dopanic(0);
		runtime·exit(3);
	case 2:
		// This is a genuine bug in the runtime, we couldn't even
		// print the stack trace successfully.
		g->m->dying = 3;
		runtime·printf("stack trace unavailable\n");
		runtime·exit(4);
	default:
		// Can't even print!  Just exit.
		runtime·exit(5);
	}
}

void
runtime·dopanic_m(void)
{
	G *gp;
	uintptr sp, pc;
	static bool didothers;
	bool crash;
	int32 t;

	gp = g->m->ptrarg[0];
	g->m->ptrarg[0] = nil;
	pc = g->m->scalararg[0];
	sp = g->m->scalararg[1];
	g->m->scalararg[1] = 0;
	if(gp->sig != 0)
		runtime·printf("[signal %x code=%p addr=%p pc=%p]\n",
			gp->sig, gp->sigcode0, gp->sigcode1, gp->sigpc);

	if((t = runtime·gotraceback(&crash)) > 0){
		if(gp != gp->m->g0) {
			runtime·printf("\n");
			runtime·goroutineheader(gp);
			runtime·traceback(pc, sp, 0, gp);
		} else if(t >= 2 || g->m->throwing > 0) {
			runtime·printf("\nruntime stack:\n");
			runtime·traceback(pc, sp, 0, gp);
		}
		if(!didothers) {
			didothers = true;
			runtime·tracebackothers(gp);
		}
	}
	runtime·unlock(&paniclk);
	if(runtime·xadd(&runtime·panicking, -1) != 0) {
		// Some other m is panicking too.
		// Let it print what it needs to print.
		// Wait forever without chewing up cpu.
		// It will exit when it's done.
		static Mutex deadlock;
		runtime·lock(&deadlock);
		runtime·lock(&deadlock);
	}
	
	if(crash)
		runtime·crash();

	runtime·exit(2);
}

#pragma textflag NOSPLIT
bool
runtime·canpanic(G *gp)
{
	M *m;
	uint32 status;

	// Note that g is m->gsignal, different from gp.
	// Note also that g->m can change at preemption, so m can go stale
	// if this function ever makes a function call.
	m = g->m;

	// Is it okay for gp to panic instead of crashing the program?
	// Yes, as long as it is running Go code, not runtime code,
	// and not stuck in a system call.
	if(gp == nil || gp != m->curg)
		return false;
	if(m->locks-m->softfloat != 0 || m->mallocing != 0 || m->throwing != 0 || m->gcing != 0 || m->dying != 0)
		return false;
	status = runtime·readgstatus(gp);
	if((status&~Gscan) != Grunning || gp->syscallsp != 0)
		return false;
#ifdef GOOS_windows
	if(m->libcallsp != 0)
		return false;
#endif
	return true;
}
