// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "arch_GOARCH.h"
#include "stack.h"
#include "malloc.h"

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
	if(d->special) {
		if(d->free)
			runtime·free(d);
	} else {
		runtime·memclr((byte*)d->args, d->siz);
	}
}

// Create a new deferred function fn with siz bytes of arguments.
// The compiler turns a defer statement into a call to this.
// Cannot split the stack because it assumes that the arguments
// are available sequentially after &fn; they would not be
// copied if a stack split occurred.  It's OK for this to call
// functions that split the stack.
#pragma textflag 7
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
#pragma textflag 7
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
	runtime·memmove(argp, d->args, d->siz);
	fn = d->fn;
	popdefer();
	freedefer(d);
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
	p->stackbase = (byte*)g->stackbase;
	g->panic = p;

	for(;;) {
		d = g->defer;
		if(d == nil)
			break;
		// take defer off list in case of recursive panic
		popdefer();
		g->ispanic = true;	// rock for newstack, where reflect.call ends up
		argp = d->argp;
		pc = d->pc;
		reflect·call(d->fn, (byte*)d->args, d->siz);
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
	void *pc;
	
	// Info about defer passed in G struct.
	argp = (void*)gp->sigcode0;
	pc = (void*)gp->sigcode1;

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
	runtime·gogo(&gp->sched, 1);
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

	while((top = (Stktop*)gp->stackbase) != nil && top->stackbase != nil) {
		stk = (byte*)gp->stackguard - StackGuard;
		if(stk <= sp && sp < (byte*)gp->stackbase)
			break;
		gp->stackbase = (uintptr)top->stackbase;
		gp->stackguard = (uintptr)top->stackguard;
		if(top->free != 0)
			runtime·stackfree(stk, top->free);
	}

	if(sp != nil && (sp < (byte*)gp->stackguard - StackGuard || (byte*)gp->stackbase < sp)) {
		runtime·printf("recover: %p not in [%p, %p]\n", sp, gp->stackguard - StackGuard, gp->stackbase);
		runtime·throw("bad unwindstack");
	}
}

// The implementation of the predeclared function recover.
// Cannot split the stack because it needs to reliably
// find the stack segment of its caller.
#pragma textflag 7
void
runtime·recover(byte *argp, Eface ret)
{
	Stktop *top, *oldtop;
	Panic *p;

	// Must be a panic going on.
	if((p = g->panic) == nil || p->recovered)
		goto nomatch;

	// Frame must be at the top of the stack segment,
	// because each deferred call starts a new stack
	// segment as a side effect of using reflect.call.
	// (There has to be some way to remember the
	// variable argument frame size, and the segment
	// code already takes care of that for us, so we
	// reuse it.)
	//
	// As usual closures complicate things: the fp that
	// the closure implementation function claims to have
	// is where the explicit arguments start, after the
	// implicit pointer arguments and PC slot.
	// If we're on the first new segment for a closure,
	// then fp == top - top->args is correct, but if
	// the closure has its own big argument frame and
	// allocated a second segment (see below),
	// the fp is slightly above top - top->args.
	// That condition can't happen normally though
	// (stack pointers go down, not up), so we can accept
	// any fp between top and top - top->args as
	// indicating the top of the segment.
	top = (Stktop*)g->stackbase;
	if(argp < (byte*)top - top->argsize || (byte*)top < argp)
		goto nomatch;

	// The deferred call makes a new segment big enough
	// for the argument frame but not necessarily big
	// enough for the function's local frame (size unknown
	// at the time of the call), so the function might have
	// made its own segment immediately.  If that's the
	// case, back top up to the older one, the one that
	// reflect.call would have made for the panic.
	//
	// The fp comparison here checks that the argument
	// frame that was copied during the split (the top->args
	// bytes above top->fp) abuts the old top of stack.
	// This is a correct test for both closure and non-closure code.
	oldtop = (Stktop*)top->stackbase;
	if(oldtop != nil && top->argp == (byte*)oldtop - top->argsize)
		top = oldtop;

	// Now we have the segment that was created to
	// run this call.  It must have been marked as a panic segment.
	if(!top->panic)
		goto nomatch;

	// Okay, this is the top frame of a deferred call
	// in response to a panic.  It can see the panic argument.
	p->recovered = 1;
	ret = p->arg;
	FLUSH(&ret);
	return;

nomatch:
	ret.type = nil;
	ret.data = nil;
	FLUSH(&ret);
}

void
runtime·startpanic(void)
{
	if(runtime·mheap == 0 || runtime·mheap->cachealloc.size == 0) { // very early
		runtime·printf("runtime: panic before malloc heap initialized\n");
		m->mallocing = 1; // tell rest of panic not to try to malloc
	} else if(m->mcache == nil) // can happen if called from signal handler or throw
		m->mcache = runtime·allocmcache();
	if(m->dying) {
		runtime·printf("panic during panic\n");
		runtime·exit(3);
	}
	m->dying = 1;
	runtime·xadd(&runtime·panicking, 1);
	runtime·lock(&paniclk);
}

void
runtime·dopanic(int32 unused)
{
	static bool didothers;
	bool crash;

	if(g->sig != 0)
		runtime·printf("[signal %x code=%p addr=%p pc=%p]\n",
			g->sig, g->sigcode0, g->sigcode1, g->sigpc);

	if(runtime·gotraceback(&crash)){
		if(g != m->g0) {
			runtime·printf("\n");
			runtime·goroutineheader(g);
			runtime·traceback(runtime·getcallerpc(&unused), runtime·getcallersp(&unused), 0, g);
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

	if(m->gcing) {
		runtime·printf("panic: %s\n", s);
		runtime·throw("panic during gc");
	}
	runtime·newErrorString(runtime·gostringnocopy((byte*)s), &err);
	runtime·panic(err);
}

void
runtime·Goexit(void)
{
	rundefer();
	runtime·goexit();
}
