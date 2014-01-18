// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "arch_GOARCH.h"
#include "stack.h"
#include "malloc.h"
#include "../../cmd/ld/textflag.h"

// Code related to defer, panic and recover.

uint32 runtime·panicking;
static Lock paniclk;

enum
{
	DeferChunkSize = 2048
};

// Allocate a Defer, usually as part of the larger frame of deferred functions.
// Each defer must be released with both popdefer and freedefer.
static Defer*
newdefer(int32 siz)
{
	int32 total;
	DeferChunk *c;
	Defer *d;
	
	c = g->dchunk;
	total = sizeof(*d) + ROUND(siz, sizeof(uintptr)) - sizeof(d->args);
	if(c == nil || total > DeferChunkSize - c->off) {
		if(total > DeferChunkSize / 2) {
			// Not worth putting in any chunk.
			// Allocate a separate block.
			d = runtime·malloc(total);
			d->siz = siz;
			d->special = 1;
			d->free = 1;
			d->link = g->defer;
			g->defer = d;
			return d;
		}

		// Cannot fit in current chunk.
		// Switch to next chunk, allocating if necessary.
		c = g->dchunknext;
		if(c == nil)
			c = runtime·malloc(DeferChunkSize);
		c->prev = g->dchunk;
		c->off = sizeof(*c);
		g->dchunk = c;
		g->dchunknext = nil;
	}

	d = (Defer*)((byte*)c + c->off);
	c->off += total;
	d->siz = siz;
	d->special = 0;
	d->free = 0;
	d->link = g->defer;
	g->defer = d;
	return d;	
}

// Pop the current defer from the defer stack.
// Its contents are still valid until the goroutine begins executing again.
// In particular it is safe to call reflect.call(d->fn, d->argp, d->siz) after
// popdefer returns.
static void
popdefer(void)
{
	Defer *d;
	DeferChunk *c;
	int32 total;
	
	d = g->defer;
	if(d == nil)
		runtime·throw("runtime: popdefer nil");
	g->defer = d->link;
	if(d->special) {
		// Nothing else to do.
		return;
	}
	total = sizeof(*d) + ROUND(d->siz, sizeof(uintptr)) - sizeof(d->args);
	c = g->dchunk;
	if(c == nil || (byte*)d+total != (byte*)c+c->off)
		runtime·throw("runtime: popdefer phase error");
	c->off -= total;
	if(c->off == sizeof(*c)) {
		// Chunk now empty, so pop from stack.
		// Save in dchunknext both to help with pingponging between frames
		// and to make sure d is still valid on return.
		if(g->dchunknext != nil)
			runtime·free(g->dchunknext);
		g->dchunknext = c;
		g->dchunk = c->prev;
	}
}

// Free the given defer.
// For defers in the per-goroutine chunk this just clears the saved arguments.
// For large defers allocated on the heap, this frees them.
// The defer cannot be used after this call.
static void
freedefer(Defer *d)
{
	int32 total;

	if(d->special) {
		if(d->free)
			runtime·free(d);
	} else {
		// Wipe out any possible pointers in argp/pc/fn/args.
		total = sizeof(*d) + ROUND(d->siz, sizeof(uintptr)) - sizeof(d->args);
		runtime·memclr((byte*)d, total);
	}
}

// Create a new deferred function fn with siz bytes of arguments.
// The compiler turns a defer statement into a call to this.
// Cannot split the stack because it assumes that the arguments
// are available sequentially after &fn; they would not be
// copied if a stack split occurred.  It's OK for this to call
// functions that split the stack.
#pragma textflag NOSPLIT
uintptr
runtime·deferproc(int32 siz, FuncVal *fn, ...)
{
	Defer *d;

	d = newdefer(siz);
	d->fn = fn;
	d->pc = runtime·getcallerpc(&siz);
	if(thechar == '5')
		d->argp = (byte*)(&fn+2);  // skip caller's saved link register
	else
		d->argp = (byte*)(&fn+1);
	runtime·memmove(d->args, d->argp, d->siz);

	// deferproc returns 0 normally.
	// a deferred func that stops a panic
	// makes the deferproc return 1.
	// the code the compiler generates always
	// checks the return value and jumps to the
	// end of the function if deferproc returns != 0.
	return 0;
}

// Run a deferred function if there is one.
// The compiler inserts a call to this at the end of any
// function which calls defer.
// If there is a deferred function, this will call runtime·jmpdefer,
// which will jump to the deferred function such that it appears
// to have been called by the caller of deferreturn at the point
// just before deferreturn was called.  The effect is that deferreturn
// is called again and again until there are no more deferred functions.
// Cannot split the stack because we reuse the caller's frame to
// call the deferred function.

// The single argument isn't actually used - it just has its address
// taken so it can be matched against pending defers.
#pragma textflag NOSPLIT
void
runtime·deferreturn(uintptr arg0)
{
	Defer *d;
	byte *argp;
	FuncVal *fn;

	d = g->defer;
	if(d == nil)
		return;
	argp = (byte*)&arg0;
	if(d->argp != argp)
		return;

	// Moving arguments around.
	// Do not allow preemption here, because the garbage collector
	// won't know the form of the arguments until the jmpdefer can
	// flip the PC over to fn.
	m->locks++;
	runtime·memmove(argp, d->args, d->siz);
	fn = d->fn;
	popdefer();
	freedefer(d);
	m->locks--;
	if(m->locks == 0 && g->preempt)
		g->stackguard0 = StackPreempt;
	runtime·jmpdefer(fn, argp);
}

// Run all deferred functions for the current goroutine.
static void
rundefer(void)
{
	Defer *d;

	while((d = g->defer) != nil) {
		popdefer();
		reflect·call(d->fn, (byte*)d->args, d->siz);
		freedefer(d);
	}
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

// The implementation of the predeclared function panic.
void
runtime·panic(Eface e)
{
	Defer *d;
	Panic *p;
	void *pc, *argp;
	
	p = runtime·mal(sizeof *p);
	p->arg = e;
	p->link = g->panic;
	p->stackbase = g->stackbase;
	g->panic = p;

	for(;;) {
		d = g->defer;
		if(d == nil)
			break;
		// take defer off list in case of recursive panic
		popdefer();
		g->ispanic = true;	// rock for newstack, where reflect.newstackcall ends up
		argp = d->argp;
		pc = d->pc;
		runtime·newstackcall(d->fn, (byte*)d->args, d->siz);
		freedefer(d);
		if(p->recovered) {
			g->panic = p->link;
			if(g->panic == nil)	// must be done with signal
				g->sig = 0;
			runtime·free(p);
			// Pass information about recovering frame to recovery.
			g->sigcode0 = (uintptr)argp;
			g->sigcode1 = (uintptr)pc;
			runtime·mcall(recovery);
			runtime·throw("recovery failed"); // mcall should not return
		}
	}

	// ran out of deferred calls - old-school panic now
	runtime·startpanic();
	printpanics(g->panic);
	runtime·dopanic(0);
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
		if(top->free != 0) {
			gp->stacksize -= top->free;
			runtime·stackfree(stk, top->free);
		}
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
runtime·recover(byte *argp, Eface ret)
{
	Panic *p;
	Stktop *top;

	// Must be an unrecovered panic in progress.
	// Must be on a stack segment created for a deferred call during a panic.
	// Must be at the top of that segment, meaning the deferred call itself
	// and not something it called. The top frame in the segment will have
	// argument pointer argp == top - top->argsize.
	// The subtraction of g->panicwrap allows wrapper functions that
	// do not count as official calls to adjust what we consider the top frame
	// while they are active on the stack. The linker emits adjustments of
	// g->panicwrap in the prologue and epilogue of functions marked as wrappers.
	top = (Stktop*)g->stackbase;
	p = g->panic;
	if(p != nil && !p->recovered && top->panic && argp == (byte*)top - top->argsize - g->panicwrap) {
		p->recovered = 1;
		ret = p->arg;
	} else {
		ret.type = nil;
		ret.data = nil;
	}
	FLUSH(&ret);
}

void
runtime·startpanic(void)
{
	if(runtime·mheap.cachealloc.size == 0) { // very early
		runtime·printf("runtime: panic before malloc heap initialized\n");
		m->mallocing = 1; // tell rest of panic not to try to malloc
	} else if(m->mcache == nil) // can happen if called from signal handler or throw
		m->mcache = runtime·allocmcache();
	if(m->dying) {
		runtime·printf("panic during panic\n");
		runtime·dopanic(0);
		runtime·exit(3); // not reached
	}
	m->dying = 1;
	if(g != nil)
		g->writebuf = nil;
	runtime·xadd(&runtime·panicking, 1);
	runtime·lock(&paniclk);
	if(runtime·debug.schedtrace > 0 || runtime·debug.scheddetail > 0)
		runtime·schedtrace(true);
	runtime·freezetheworld();
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
		if(g != m->g0) {
			runtime·printf("\n");
			runtime·goroutineheader(g);
			runtime·traceback((uintptr)runtime·getcallerpc(&unused), (uintptr)runtime·getcallersp(&unused), 0, g);
		} else if(t >= 2 || m->throwing > 0) {
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
		static Lock deadlock;
		runtime·lock(&deadlock);
		runtime·lock(&deadlock);
	}
	
	if(crash)
		runtime·crash();

	runtime·exit(2);
}

void
runtime·panicindex(void)
{
	runtime·panicstring("index out of range");
}

void
runtime·panicslice(void)
{
	runtime·panicstring("slice bounds out of range");
}

void
runtime·throwreturn(void)
{
	// can only happen if compiler is broken
	runtime·throw("no return at end of a typed function - compiler is broken");
}

void
runtime·throwinit(void)
{
	// can only happen with linker skew
	runtime·throw("recursive call during initialization - linker skew");
}

void
runtime·throw(int8 *s)
{
	if(m->throwing == 0)
		m->throwing = 1;
	runtime·startpanic();
	runtime·printf("fatal error: %s\n", s);
	runtime·dopanic(0);
	*(int32*)0 = 0;	// not reached
	runtime·exit(1);	// even more not reached
}

void
runtime·panicstring(int8 *s)
{
	Eface err;

	if(m->mallocing) {
		runtime·printf("panic: %s\n", s);
		runtime·throw("panic during malloc");
	}
	if(m->gcing) {
		runtime·printf("panic: %s\n", s);
		runtime·throw("panic during gc");
	}
	runtime·newErrorCString(s, &err);
	runtime·panic(err);
}

void
runtime·Goexit(void)
{
	rundefer();
	runtime·goexit();
}
