// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "arch_GOARCH.h"
#include "malloc.h"
#include "stack.h"

typedef struct StackCacheNode StackCacheNode;
struct StackCacheNode
{
	StackCacheNode *next;
	void*	batch[StackCacheBatch-1];
};

static StackCacheNode *stackcache;
static Lock stackcachemu;

// stackcacherefill/stackcacherelease implement a global cache of stack segments.
// The cache is required to prevent unlimited growth of per-thread caches.
static void
stackcacherefill(void)
{
	StackCacheNode *n;
	int32 i, pos;

	runtime·lock(&stackcachemu);
	n = stackcache;
	if(n)
		stackcache = n->next;
	runtime·unlock(&stackcachemu);
	if(n == nil) {
		n = (StackCacheNode*)runtime·SysAlloc(FixedStack*StackCacheBatch);
		if(n == nil)
			runtime·throw("out of memory (stackcacherefill)");
		runtime·xadd64(&mstats.stacks_sys, FixedStack*StackCacheBatch);
		for(i = 0; i < StackCacheBatch-1; i++)
			n->batch[i] = (byte*)n + (i+1)*FixedStack;
	}
	pos = m->stackcachepos;
	for(i = 0; i < StackCacheBatch-1; i++) {
		m->stackcache[pos] = n->batch[i];
		pos = (pos + 1) % StackCacheSize;
	}
	m->stackcache[pos] = n;
	pos = (pos + 1) % StackCacheSize;
	m->stackcachepos = pos;
	m->stackcachecnt += StackCacheBatch;
}

static void
stackcacherelease(void)
{
	StackCacheNode *n;
	uint32 i, pos;

	pos = (m->stackcachepos - m->stackcachecnt) % StackCacheSize;
	n = (StackCacheNode*)m->stackcache[pos];
	pos = (pos + 1) % StackCacheSize;
	for(i = 0; i < StackCacheBatch-1; i++) {
		n->batch[i] = m->stackcache[pos];
		pos = (pos + 1) % StackCacheSize;
	}
	m->stackcachecnt -= StackCacheBatch;
	runtime·lock(&stackcachemu);
	n->next = stackcache;
	stackcache = n;
	runtime·unlock(&stackcachemu);
}

void*
runtime·stackalloc(uint32 n)
{
	uint32 pos;
	void *v;

	// Stackalloc must be called on scheduler stack, so that we
	// never try to grow the stack during the code that stackalloc runs.
	// Doing so would cause a deadlock (issue 1547).
	if(g != m->g0)
		runtime·throw("stackalloc not on scheduler stack");

	// Stack allocator uses malloc/free most of the time,
	// but if we're in the middle of malloc and need stack,
	// we have to do something else to avoid deadlock.
	// In that case, we fall back on a fixed-size free-list
	// allocator, assuming that inside malloc all the stack
	// frames are small, so that all the stack allocations
	// will be a single size, the minimum (right now, 5k).
	if(n == FixedStack || m->mallocing || m->gcing) {
		if(n != FixedStack) {
			runtime·printf("stackalloc: in malloc, size=%d want %d\n", FixedStack, n);
			runtime·throw("stackalloc");
		}
		if(m->stackcachecnt == 0)
			stackcacherefill();
		pos = m->stackcachepos;
		pos = (pos - 1) % StackCacheSize;
		v = m->stackcache[pos];
		m->stackcachepos = pos;
		m->stackcachecnt--;
		m->stackinuse++;
		return v;
	}
	return runtime·mallocgc(n, FlagNoProfiling|FlagNoGC, 0, 0);
}

void
runtime·stackfree(void *v, uintptr n)
{
	uint32 pos;

	if(n == FixedStack || m->mallocing || m->gcing) {
		if(m->stackcachecnt == StackCacheSize)
			stackcacherelease();
		pos = m->stackcachepos;
		m->stackcache[pos] = v;
		m->stackcachepos = (pos + 1) % StackCacheSize;
		m->stackcachecnt++;
		m->stackinuse--;
		return;
	}
	runtime·free(v);
}

// Called from runtime·lessstack when returning from a function which
// allocated a new stack segment.  The function's return value is in
// m->cret.
void
runtime·oldstack(void)
{
	Stktop *top;
	Gobuf label;
	uint32 argsize;
	uintptr cret;
	byte *sp, *old;
	uintptr *src, *dst, *dstend;
	G *gp;
	int64 goid;

//printf("oldstack m->cret=%p\n", m->cret);

	gp = m->curg;
	top = (Stktop*)gp->stackbase;
	old = (byte*)gp->stackguard - StackGuard;
	sp = (byte*)top;
	argsize = top->argsize;
	if(argsize > 0) {
		sp -= argsize;
		dst = (uintptr*)top->argp;
		dstend = dst + argsize/sizeof(*dst);
		src = (uintptr*)sp;
		while(dst < dstend)
			*dst++ = *src++;
	}
	goid = top->gobuf.g->goid;	// fault if g is bad, before gogo
	USED(goid);

	label = top->gobuf;
	gp->stackbase = (uintptr)top->stackbase;
	gp->stackguard = (uintptr)top->stackguard;
	if(top->free != 0)
		runtime·stackfree(old, top->free);

	cret = m->cret;
	m->cret = 0;  // drop reference
	runtime·gogo(&label, cret);
}

// Called from reflect·call or from runtime·morestack when a new
// stack segment is needed.  Allocate a new stack big enough for
// m->moreframesize bytes, copy m->moreargsize bytes to the new frame,
// and then act as though runtime·lessstack called the function at
// m->morepc.
void
runtime·newstack(void)
{
	int32 framesize, minalloc, argsize;
	Stktop *top;
	byte *stk, *sp;
	uintptr *src, *dst, *dstend;
	G *gp;
	Gobuf label;
	bool reflectcall;
	uintptr free;

	framesize = m->moreframesize;
	argsize = m->moreargsize;
	gp = m->curg;

	if(m->morebuf.sp < gp->stackguard - StackGuard) {
		runtime·printf("runtime: split stack overflow: %p < %p\n", m->morebuf.sp, gp->stackguard - StackGuard);
		runtime·throw("runtime: split stack overflow");
	}
	if(argsize % sizeof(uintptr) != 0) {
		runtime·printf("runtime: stack split with misaligned argsize %d\n", argsize);
		runtime·throw("runtime: stack split argsize");
	}

	minalloc = 0;
	reflectcall = framesize==1;
	if(reflectcall) {
		framesize = 0;
		// moreframesize_minalloc is only set in runtime·gc(),
		// that calls newstack via reflect·call().
		minalloc = m->moreframesize_minalloc;
		m->moreframesize_minalloc = 0;
		if(framesize < minalloc)
			framesize = minalloc;
	}

	if(reflectcall && minalloc == 0 && m->morebuf.sp - sizeof(Stktop) - argsize - 32 > gp->stackguard) {
		// special case: called from reflect.call (framesize==1)
		// to call code with an arbitrary argument size,
		// and we have enough space on the current stack.
		// the new Stktop* is necessary to unwind, but
		// we don't need to create a new segment.
		top = (Stktop*)(m->morebuf.sp - sizeof(*top));
		stk = (byte*)gp->stackguard - StackGuard;
		free = 0;
	} else {
		// allocate new segment.
		framesize += argsize;
		framesize += StackExtra;	// room for more functions, Stktop.
		if(framesize < StackMin)
			framesize = StackMin;
		framesize += StackSystem;
		stk = runtime·stackalloc(framesize);
		top = (Stktop*)(stk+framesize-sizeof(*top));
		free = framesize;
	}

	if(0) {
		runtime·printf("newstack framesize=%d argsize=%d morepc=%p moreargp=%p gobuf=%p, %p top=%p old=%p\n",
			framesize, argsize, m->morepc, m->moreargp, m->morebuf.pc, m->morebuf.sp, top, gp->stackbase);
	}

	top->stackbase = (byte*)gp->stackbase;
	top->stackguard = (byte*)gp->stackguard;
	top->gobuf = m->morebuf;
	top->argp = m->moreargp;
	top->argsize = argsize;
	top->free = free;
	m->moreargp = nil;
	m->morebuf.pc = nil;
	m->morebuf.sp = (uintptr)nil;

	// copy flag from panic
	top->panic = gp->ispanic;
	gp->ispanic = false;

	gp->stackbase = (uintptr)top;
	gp->stackguard = (uintptr)stk + StackGuard;

	sp = (byte*)top;
	if(argsize > 0) {
		sp -= argsize;
		dst = (uintptr*)sp;
		dstend = dst + argsize/sizeof(*dst);
		src = (uintptr*)top->argp;
		while(dst < dstend)
			*dst++ = *src++;
	}
	if(thechar == '5') {
		// caller would have saved its LR below args.
		sp -= sizeof(void*);
		*(void**)sp = nil;
	}

	// Continue as if lessstack had just called m->morepc
	// (the PC that decided to grow the stack).
	label.sp = (uintptr)sp;
	label.pc = (byte*)runtime·lessstack;
	label.g = m->curg;
	if(reflectcall)
		runtime·gogocallfn(&label, (FuncVal*)m->morepc);
	else
		runtime·gogocall(&label, m->morepc, m->cret);

	*(int32*)345 = 123;	// never return
}
