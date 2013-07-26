// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "arch_GOARCH.h"
#include "malloc.h"
#include "stack.h"

enum
{
	StackDebug = 0,
};

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

	// Stacks are usually allocated with a fixed-size free-list allocator,
	// but if we need a stack of non-standard size, we fall back on malloc
	// (assuming that inside malloc and GC all the stack frames are small,
	// so that we do not deadlock).
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
	return runtime·mallocgc(n, 0, FlagNoProfiling|FlagNoGC|FlagNoZero|FlagNoInvokeGC);
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
	uint32 argsize;
	byte *sp, *old;
	uintptr *src, *dst, *dstend;
	G *gp;
	int64 goid;
	int32 oldstatus;

	gp = m->curg;
	top = (Stktop*)gp->stackbase;
	old = (byte*)gp->stackguard - StackGuard;
	sp = (byte*)top;
	argsize = top->argsize;

	if(StackDebug) {
		runtime·printf("runtime: oldstack gobuf={pc:%p sp:%p lr:%p} cret=%p argsize=%p\n",
			top->gobuf.pc, top->gobuf.sp, top->gobuf.lr, m->cret, (uintptr)argsize);
	}

	// gp->status is usually Grunning, but it could be Gsyscall if a stack split
	// happens during a function call inside entersyscall.
	oldstatus = gp->status;
	
	gp->sched = top->gobuf;
	gp->sched.ret = m->cret;
	m->cret = 0; // drop reference
	gp->status = Gwaiting;
	gp->waitreason = "stack unsplit";

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

	gp->stackbase = top->stackbase;
	gp->stackguard = top->stackguard;
	gp->stackguard0 = gp->stackguard;

	if(top->free != 0)
		runtime·stackfree(old, top->free);

	gp->status = oldstatus;
	runtime·gogo(&gp->sched);
}

// Called from reflect·call or from runtime·morestack when a new
// stack segment is needed.  Allocate a new stack big enough for
// m->moreframesize bytes, copy m->moreargsize bytes to the new frame,
// and then act as though runtime·lessstack called the function at
// m->morepc.
void
runtime·newstack(void)
{
	int32 framesize, argsize, oldstatus;
	Stktop *top;
	byte *stk;
	uintptr sp;
	uintptr *src, *dst, *dstend;
	G *gp;
	Gobuf label;
	bool reflectcall;
	uintptr free;

	// gp->status is usually Grunning, but it could be Gsyscall if a stack split
	// happens during a function call inside entersyscall.
	gp = m->curg;
	oldstatus = gp->status;

	framesize = m->moreframesize;
	argsize = m->moreargsize;
	gp->status = Gwaiting;
	gp->waitreason = "stack split";
	reflectcall = framesize==1;
	if(reflectcall)
		framesize = 0;

	// For reflectcall the context already points to beginning of reflect·call.
	if(!reflectcall)
		runtime·rewindmorestack(&gp->sched);

	sp = gp->sched.sp;
	if(thechar == '6' || thechar == '8') {
		// The call to morestack cost a word.
		sp -= sizeof(uintptr);
	}
	if(StackDebug || sp < gp->stackguard - StackGuard) {
		runtime·printf("runtime: newstack framesize=%p argsize=%p sp=%p stack=[%p, %p]\n"
			"\tmorebuf={pc:%p sp:%p lr:%p}\n"
			"\tsched={pc:%p sp:%p lr:%p ctxt:%p}\n",
			(uintptr)framesize, (uintptr)argsize, sp, gp->stackguard - StackGuard, gp->stackbase,
			m->morebuf.pc, m->morebuf.sp, m->morebuf.lr,
			gp->sched.pc, gp->sched.sp, gp->sched.lr, gp->sched.ctxt);
	}
	if(sp < gp->stackguard - StackGuard) {
		runtime·printf("runtime: split stack overflow: %p < %p\n", sp, gp->stackguard - StackGuard);
		runtime·throw("runtime: split stack overflow");
	}

	if(argsize % sizeof(uintptr) != 0) {
		runtime·printf("runtime: stack split with misaligned argsize %d\n", argsize);
		runtime·throw("runtime: stack split argsize");
	}

	if(gp->stackguard0 == (uintptr)StackPreempt) {
		if(gp == m->g0)
			runtime·throw("runtime: preempt g0");
		if(oldstatus == Grunning && m->p == nil)
			runtime·throw("runtime: g is running but p is not");
		// Be conservative about where we preempt.
		// We are interested in preempting user Go code, not runtime code.
		if(oldstatus != Grunning || m->locks || m->mallocing || m->gcing || m->p->status != Prunning) {
			// Let the goroutine keep running for now.
			// gp->preempt is set, so it will be preempted next time.
			gp->stackguard0 = gp->stackguard;
			gp->status = oldstatus;
			runtime·gogo(&gp->sched);	// never return
		}
		// Act like goroutine called runtime.Gosched.
		gp->status = oldstatus;
		runtime·gosched0(gp);	// never return
	}

	if(reflectcall && m->morebuf.sp - sizeof(Stktop) - argsize - 32 > gp->stackguard) {
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

	if(StackDebug) {
		runtime·printf("\t-> new stack [%p, %p]\n", stk, top);
	}

	top->stackbase = gp->stackbase;
	top->stackguard = gp->stackguard;
	top->gobuf = m->morebuf;
	top->argp = m->moreargp;
	top->argsize = argsize;
	top->free = free;
	m->moreargp = nil;
	m->morebuf.pc = (uintptr)nil;
	m->morebuf.lr = (uintptr)nil;
	m->morebuf.sp = (uintptr)nil;

	// copy flag from panic
	top->panic = gp->ispanic;
	gp->ispanic = false;

	gp->stackbase = (uintptr)top;
	gp->stackguard = (uintptr)stk + StackGuard;
	gp->stackguard0 = gp->stackguard;

	sp = (uintptr)top;
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
	runtime·memclr((byte*)&label, sizeof label);
	label.sp = sp;
	label.pc = (uintptr)runtime·lessstack;
	label.g = m->curg;
	if(reflectcall)
		runtime·gostartcallfn(&label, (FuncVal*)m->cret);
	else {
		runtime·gostartcall(&label, (void(*)(void))gp->sched.pc, gp->sched.ctxt);
		gp->sched.ctxt = nil;
	}
	gp->status = oldstatus;
	runtime·gogo(&label);

	*(int32*)345 = 123;	// never return
}

// adjust Gobuf as if it executed a call to fn
// and then did an immediate gosave.
void
runtime·gostartcallfn(Gobuf *gobuf, FuncVal *fv)
{
	runtime·gostartcall(gobuf, fv->fn, fv);
}
