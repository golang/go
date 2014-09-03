// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "arch_GOARCH.h"
#include "stack.h"
#include "malloc.h"
#include "../../cmd/ld/textflag.h"

// Code related to defer, panic and recover.

// TODO: remove once code is moved to Go
extern Defer* runtime·newdefer(int32 siz);
extern runtime·freedefer(Defer *d);

uint32 runtime·panicking;
static Mutex paniclk;

void
runtime·deferproc_m(void) {
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

	d = runtime·newdefer(siz);
	d->fn = fn;
	d->pc = callerpc;
	d->argp = argp;
	runtime·memmove(d->args, (void*)argp, siz);
}

// Print all currently active panics.  Used when crashing.
static void
printpanics(Panic *p)
{
	if(p->link) {
		printpanics(p->link);
		runtime·printf("\t");
	}
	runtime·printf("panic: ");
	runtime·printany(p->arg);
	if(p->recovered)
		runtime·printf(" [recovered]");
	runtime·printf("\n");
}

static void recovery(G*);
static void abortpanic(Panic*);
static FuncVal abortpanicV = { (void(*)(void))abortpanic };

// The implementation of the predeclared function panic.
void
runtime·panic(Eface e)
{
	Defer *d, dabort;
	Panic p;
	uintptr pc, argp;
	void (*fn)(G*);

	runtime·memclr((byte*)&p, sizeof p);
	p.arg = e;
	p.link = g->panic;
	p.stackbase = g->stackbase;
	g->panic = &p;

	dabort.fn = &abortpanicV;
	dabort.siz = sizeof(&p);
	dabort.args[0] = &p;
	dabort.argp = NoArgs;
	dabort.special = true;

	for(;;) {
		d = g->defer;
		if(d == nil)
			break;
		// take defer off list in case of recursive panic
		g->defer = d->link;
		g->ispanic = true;	// rock for runtime·newstack, where runtime·newstackcall ends up
		argp = d->argp;
		pc = d->pc;

		// The deferred function may cause another panic,
		// so newstackcall may not return. Set up a defer
		// to mark this panic aborted if that happens.
		dabort.link = g->defer;
		g->defer = &dabort;
		p.defer = d;

		runtime·newstackcall(d->fn, (byte*)d->args, d->siz);

		// Newstackcall did not panic. Remove dabort.
		if(g->defer != &dabort)
			runtime·throw("bad defer entry in panic");
		g->defer = dabort.link;

		runtime·freedefer(d);
		if(p.recovered) {
			g->panic = p.link;
			// Aborted panics are marked but remain on the g->panic list.
			// Recovery will unwind the stack frames containing their Panic structs.
			// Remove them from the list and free the associated defers.
			while(g->panic && g->panic->aborted) {
				runtime·freedefer(g->panic->defer);
				g->panic = g->panic->link;
			}
			if(g->panic == nil)	// must be done with signal
				g->sig = 0;
			// Pass information about recovering frame to recovery.
			g->sigcode0 = (uintptr)argp;
			g->sigcode1 = (uintptr)pc;
			fn = recovery;
			runtime·mcall(&fn);
			runtime·throw("recovery failed"); // mcall should not return
		}
	}

	// ran out of deferred calls - old-school panic now
	runtime·startpanic();
	printpanics(g->panic);
	runtime·dopanic(0);	// should not return
	runtime·exit(1);	// not reached
}

static void
abortpanic(Panic *p)
{
	p->aborted = true;
}

// Unwind the stack after a deferred function calls recover
// after a panic.  Then arrange to continue running as though
// the caller of the deferred function returned normally.
static void
recovery(G *gp)
{
	void *argp;
	uintptr pc;
	
	// Info about defer passed in G struct.
	argp = (void*)gp->sigcode0;
	pc = (uintptr)gp->sigcode1;

	// Unwind to the stack frame with d's arguments in it.
	runtime·unwindstack(gp, argp);

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

// Free stack frames until we hit the last one
// or until we find the one that contains the sp.
void
runtime·unwindstack(G *gp, byte *sp)
{
	Stktop *top;
	byte *stk;

	// Must be called from a different goroutine, usually m->g0.
	if(g == gp)
		runtime·throw("unwindstack on self");

	while((top = (Stktop*)gp->stackbase) != 0 && top->stackbase != 0) {
		stk = (byte*)gp->stackguard - StackGuard;
		if(stk <= sp && sp < (byte*)gp->stackbase)
			break;
		gp->stackbase = top->stackbase;
		gp->stackguard = top->stackguard;
		gp->stackguard0 = gp->stackguard;
		runtime·stackfree(gp, stk, top);
	}

	if(sp != nil && (sp < (byte*)gp->stackguard - StackGuard || (byte*)gp->stackbase < sp)) {
		runtime·printf("recover: %p not in [%p, %p]\n", sp, gp->stackguard - StackGuard, gp->stackbase);
		runtime·throw("bad unwindstack");
	}
}

// The implementation of the predeclared function recover.
// Cannot split the stack because it needs to reliably
// find the stack segment of its caller.
#pragma textflag NOSPLIT
void
runtime·recover(byte *argp, GoOutput retbase, ...)
{
	Panic *p;
	Stktop *top;
	Eface *ret;

	// Must be an unrecovered panic in progress.
	// Must be on a stack segment created for a deferred call during a panic.
	// Must be at the top of that segment, meaning the deferred call itself
	// and not something it called. The top frame in the segment will have
	// argument pointer argp == top - top->argsize.
	// The subtraction of g->panicwrap allows wrapper functions that
	// do not count as official calls to adjust what we consider the top frame
	// while they are active on the stack. The linker emits adjustments of
	// g->panicwrap in the prologue and epilogue of functions marked as wrappers.
	ret = (Eface*)&retbase;
	top = (Stktop*)g->stackbase;
	p = g->panic;
	if(p != nil && !p->recovered && top->panic && argp == (byte*)top - top->argsize - g->panicwrap) {
		p->recovered = 1;
		*ret = p->arg;
	} else {
		ret->type = nil;
		ret->data = nil;
	}
}

void
runtime·startpanic(void)
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
runtime·dopanic(int32 unused)
{
	static bool didothers;
	bool crash;
	int32 t;

	if(g->sig != 0)
		runtime·printf("[signal %x code=%p addr=%p pc=%p]\n",
			g->sig, g->sigcode0, g->sigcode1, g->sigpc);

	if((t = runtime·gotraceback(&crash)) > 0){
		if(g != g->m->g0) {
			runtime·printf("\n");
			runtime·goroutineheader(g);
			runtime·traceback((uintptr)runtime·getcallerpc(&unused), (uintptr)runtime·getcallersp(&unused), 0, g);
		} else if(t >= 2 || g->m->throwing > 0) {
			runtime·printf("\nruntime stack:\n");
			runtime·traceback((uintptr)runtime·getcallerpc(&unused), (uintptr)runtime·getcallersp(&unused), 0, g);
		}
		if(!didothers) {
			didothers = true;
			runtime·tracebackothers(g);
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

void
runtime·throw(int8 *s)
{
	if(g->m->throwing == 0)
		g->m->throwing = 1;
	runtime·startpanic();
	runtime·printf("fatal error: %s\n", s);
	runtime·dopanic(0);
	*(int32*)0 = 0;	// not reached
	runtime·exit(1);	// even more not reached
}

void
runtime·gothrow(String s)
{
	if(g->m->throwing == 0)
		g->m->throwing = 1;
	runtime·startpanic();
	runtime·printf("fatal error: %S\n", s);
	runtime·dopanic(0);
	*(int32*)0 = 0;	// not reached
	runtime·exit(1);	// even more not reached
}

void
runtime·panicstring(int8 *s)
{
	Eface err;

	// m->softfloat is set during software floating point,
	// which might cause a fault during a memory load.
	// It increments m->locks to avoid preemption.
	// If we're panicking, the software floating point frames
	// will be unwound, so decrement m->locks as they would.
	if(g->m->softfloat) {
		g->m->locks--;
		g->m->softfloat = 0;
	}

	if(g->m->mallocing) {
		runtime·printf("panic: %s\n", s);
		runtime·throw("panic during malloc");
	}
	if(g->m->gcing) {
		runtime·printf("panic: %s\n", s);
		runtime·throw("panic during gc");
	}
	if(g->m->locks) {
		runtime·printf("panic: %s\n", s);
		runtime·throw("panic holding locks");
	}
	runtime·newErrorCString(s, &err);
	runtime·panic(err);
}
