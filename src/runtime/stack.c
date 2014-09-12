// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "arch_GOARCH.h"
#include "malloc.h"
#include "stack.h"
#include "funcdata.h"
#include "typekind.h"
#include "type.h"
#include "race.h"
#include "mgc0.h"
#include "textflag.h"

enum
{
	// StackDebug == 0: no logging
	//            == 1: logging of per-stack operations
	//            == 2: logging of per-frame operations
	//            == 3: logging of per-word updates
	//            == 4: logging of per-word reads
	StackDebug = 0,
	StackFromSystem = 0,	// allocate stacks from system memory instead of the heap
	StackFaultOnFree = 0,	// old stacks are mapped noaccess to detect use after free

	StackCache = 1,
};

// Global pool of spans that have free stacks.
// Stacks are assigned an order according to size.
//     order = log_2(size/FixedStack)
// There is a free list for each order.
static MSpan stackpool[NumStackOrders];
static Mutex stackpoolmu;
// TODO: one lock per order?

void
runtime·stackinit(void)
{
	int32 i;

	if((StackCacheSize & PageMask) != 0)
		runtime·throw("cache size must be a multiple of page size");

	for(i = 0; i < NumStackOrders; i++)
		runtime·MSpanList_Init(&stackpool[i]);
}

// Allocates a stack from the free pool.  Must be called with
// stackpoolmu held.
static MLink*
poolalloc(uint8 order)
{
	MSpan *list;
	MSpan *s;
	MLink *x;
	uintptr i;

	list = &stackpool[order];
	s = list->next;
	if(s == list) {
		// no free stacks.  Allocate another span worth.
		s = runtime·MHeap_AllocStack(&runtime·mheap, StackCacheSize >> PageShift);
		if(s == nil)
			runtime·throw("out of memory");
		if(s->ref != 0)
			runtime·throw("bad ref");
		if(s->freelist != nil)
			runtime·throw("bad freelist");
		for(i = 0; i < StackCacheSize; i += FixedStack << order) {
			x = (MLink*)((s->start << PageShift) + i);
			x->next = s->freelist;
			s->freelist = x;
		}
		runtime·MSpanList_Insert(list, s);
	}
	x = s->freelist;
	if(x == nil)
		runtime·throw("span has no free stacks");
	s->freelist = x->next;
	s->ref++;
	if(s->freelist == nil) {
		// all stacks in s are allocated.
		runtime·MSpanList_Remove(s);
	}
	return x;
}

// Adds stack x to the free pool.  Must be called with stackpoolmu held.
static void
poolfree(MLink *x, uint8 order)
{
	MSpan *s;

	s = runtime·MHeap_Lookup(&runtime·mheap, x);
	if(s->state != MSpanStack)
		runtime·throw("freeing stack not in a stack span");
	if(s->freelist == nil) {
		// s will now have a free stack
		runtime·MSpanList_Insert(&stackpool[order], s);
	}
	x->next = s->freelist;
	s->freelist = x;
	s->ref--;
	if(s->ref == 0) {
		// span is completely free - return to heap
		runtime·MSpanList_Remove(s);
		s->freelist = nil;
		runtime·MHeap_FreeStack(&runtime·mheap, s);
	}
}

// stackcacherefill/stackcacherelease implement a global pool of stack segments.
// The pool is required to prevent unlimited growth of per-thread caches.
static void
stackcacherefill(MCache *c, uint8 order)
{
	MLink *x, *list;
	uintptr size;

	if(StackDebug >= 1)
		runtime·printf("stackcacherefill order=%d\n", order);

	// Grab some stacks from the global cache.
	// Grab half of the allowed capacity (to prevent thrashing).
	list = nil;
	size = 0;
	runtime·lock(&stackpoolmu);
	while(size < StackCacheSize/2) {
		x = poolalloc(order);
		x->next = list;
		list = x;
		size += FixedStack << order;
	}
	runtime·unlock(&stackpoolmu);

	c->stackcache[order].list = list;
	c->stackcache[order].size = size;
}

static void
stackcacherelease(MCache *c, uint8 order)
{
	MLink *x, *y;
	uintptr size;

	if(StackDebug >= 1)
		runtime·printf("stackcacherelease order=%d\n", order);
	x = c->stackcache[order].list;
	size = c->stackcache[order].size;
	runtime·lock(&stackpoolmu);
	while(size > StackCacheSize/2) {
		y = x->next;
		poolfree(x, order);
		x = y;
		size -= FixedStack << order;
	}
	runtime·unlock(&stackpoolmu);
	c->stackcache[order].list = x;
	c->stackcache[order].size = size;
}

void
runtime·stackcache_clear(MCache *c)
{
	uint8 order;
	MLink *x, *y;

	if(StackDebug >= 1)
		runtime·printf("stackcache clear\n");
	runtime·lock(&stackpoolmu);
	for(order = 0; order < NumStackOrders; order++) {
		x = c->stackcache[order].list;
		while(x != nil) {
			y = x->next;
			poolfree(x, order);
			x = y;
		}
		c->stackcache[order].list = nil;
		c->stackcache[order].size = 0;
	}
	runtime·unlock(&stackpoolmu);
}

Stack
runtime·stackalloc(uint32 n)
{
	uint8 order;
	uint32 n2;
	void *v;
	MLink *x;
	MSpan *s;
	MCache *c;

	// Stackalloc must be called on scheduler stack, so that we
	// never try to grow the stack during the code that stackalloc runs.
	// Doing so would cause a deadlock (issue 1547).
	if(g != g->m->g0)
		runtime·throw("stackalloc not on scheduler stack");
	if((n & (n-1)) != 0)
		runtime·throw("stack size not a power of 2");
	if(StackDebug >= 1)
		runtime·printf("stackalloc %d\n", n);

	if(runtime·debug.efence || StackFromSystem) {
		v = runtime·sysAlloc(ROUND(n, PageSize), &mstats.stacks_sys);
		if(v == nil)
			runtime·throw("out of memory (stackalloc)");
		return (Stack){(uintptr)v, (uintptr)v+n};
	}

	// Small stacks are allocated with a fixed-size free-list allocator.
	// If we need a stack of a bigger size, we fall back on allocating
	// a dedicated span.
	if(StackCache && n < FixedStack << NumStackOrders && n < StackCacheSize) {
		order = 0;
		n2 = n;
		while(n2 > FixedStack) {
			order++;
			n2 >>= 1;
		}
		c = g->m->mcache;
		if(c == nil || g->m->gcing || g->m->helpgc) {
			// c == nil can happen in the guts of exitsyscall or
			// procresize. Just get a stack from the global pool.
			// Also don't touch stackcache during gc
			// as it's flushed concurrently.
			runtime·lock(&stackpoolmu);
			x = poolalloc(order);
			runtime·unlock(&stackpoolmu);
		} else {
			x = c->stackcache[order].list;
			if(x == nil) {
				stackcacherefill(c, order);
				x = c->stackcache[order].list;
			}
			c->stackcache[order].list = x->next;
			c->stackcache[order].size -= n;
		}
		v = (byte*)x;
	} else {
		s = runtime·MHeap_AllocStack(&runtime·mheap, ROUND(n, PageSize) >> PageShift);
		if(s == nil)
			runtime·throw("out of memory");
		v = (byte*)(s->start<<PageShift);
	}
	
	if(raceenabled)
		runtime·racemalloc(v, n);
	if(StackDebug >= 1)
		runtime·printf("  allocated %p\n", v);
	return (Stack){(uintptr)v, (uintptr)v+n};
}

void
runtime·stackfree(Stack stk)
{
	uint8 order;
	uintptr n, n2;
	MSpan *s;
	MLink *x;
	MCache *c;
	void *v;
	
	n = stk.hi - stk.lo;
	v = (void*)stk.lo;
	if(n & (n-1))
		runtime·throw("stack not a power of 2");
	if(StackDebug >= 1) {
		runtime·printf("stackfree %p %d\n", v, (int32)n);
		runtime·memclr(v, n); // for testing, clobber stack data
	}
	if(runtime·debug.efence || StackFromSystem) {
		if(runtime·debug.efence || StackFaultOnFree)
			runtime·SysFault(v, n);
		else
			runtime·SysFree(v, n, &mstats.stacks_sys);
		return;
	}
	if(StackCache && n < FixedStack << NumStackOrders && n < StackCacheSize) {
		order = 0;
		n2 = n;
		while(n2 > FixedStack) {
			order++;
			n2 >>= 1;
		}
		x = (MLink*)v;
		c = g->m->mcache;
		if(c == nil || g->m->gcing || g->m->helpgc) {
			runtime·lock(&stackpoolmu);
			poolfree(x, order);
			runtime·unlock(&stackpoolmu);
		} else {
			if(c->stackcache[order].size >= StackCacheSize)
				stackcacherelease(c, order);
			x->next = c->stackcache[order].list;
			c->stackcache[order].list = x;
			c->stackcache[order].size += n;
		}
	} else {
		s = runtime·MHeap_Lookup(&runtime·mheap, v);
		if(s->state != MSpanStack) {
			runtime·printf("%p %p\n", s->start<<PageShift, v);
			runtime·throw("bad span state");
		}
		runtime·MHeap_FreeStack(&runtime·mheap, s);
	}
}

uintptr runtime·maxstacksize = 1<<20; // enough until runtime.main sets it for real

static uint8*
mapnames[] = {
	(uint8*)"---",
	(uint8*)"scalar",
	(uint8*)"ptr",
	(uint8*)"multi",
};

// Stack frame layout
//
// (x86)
// +------------------+
// | args from caller |
// +------------------+ <- frame->argp
// |  return address  |
// +------------------+ <- frame->varp
// |     locals       |
// +------------------+
// |  args to callee  |
// +------------------+ <- frame->sp
//
// (arm)
// +------------------+
// | args from caller |
// +------------------+ <- frame->argp
// | caller's retaddr |
// +------------------+ <- frame->varp
// |     locals       |
// +------------------+
// |  args to callee  |
// +------------------+
// |  return address  |
// +------------------+ <- frame->sp

void runtime·main(void);
void runtime·switchtoM(void(*)(void));

typedef struct AdjustInfo AdjustInfo;
struct AdjustInfo {
	Stack old;
	uintptr delta;  // ptr distance from old to new stack (newbase - oldbase)
};

// bv describes the memory starting at address scanp.
// Adjust any pointers contained therein.
static void
adjustpointers(byte **scanp, BitVector *bv, AdjustInfo *adjinfo, Func *f)
{
	uintptr delta;
	int32 num, i;
	byte *p, *minp, *maxp;
	Type *t;
	Itab *tab;
	
	minp = (byte*)adjinfo->old.lo;
	maxp = (byte*)adjinfo->old.hi;
	delta = adjinfo->delta;
	num = bv->n / BitsPerPointer;
	for(i = 0; i < num; i++) {
		if(StackDebug >= 4)
			runtime·printf("        %p:%s:%p\n", &scanp[i], mapnames[bv->data[i / (32 / BitsPerPointer)] >> (i * BitsPerPointer & 31) & 3], scanp[i]);
		switch(bv->data[i / (32 / BitsPerPointer)] >> (i * BitsPerPointer & 31) & 3) {
		case BitsDead:
			if(runtime·debug.gcdead)
				scanp[i] = (byte*)PoisonStack;
			break;
		case BitsScalar:
			break;
		case BitsPointer:
			p = scanp[i];
			if(f != nil && (byte*)0 < p && (p < (byte*)PageSize || (uintptr)p == PoisonGC || (uintptr)p == PoisonStack)) {
				// Looks like a junk value in a pointer slot.
				// Live analysis wrong?
				g->m->traceback = 2;
				runtime·printf("runtime: bad pointer in frame %s at %p: %p\n", runtime·funcname(f), &scanp[i], p);
				runtime·throw("bad pointer!");
			}
			if(minp <= p && p < maxp) {
				if(StackDebug >= 3)
					runtime·printf("adjust ptr %p %s\n", p, runtime·funcname(f));
				scanp[i] = p + delta;
			}
			break;
		case BitsMultiWord:
			switch(bv->data[(i+1) / (32 / BitsPerPointer)] >> ((i+1) * BitsPerPointer & 31) & 3) {
			default:
				runtime·throw("unexpected garbage collection bits");
			case BitsEface:
				t = (Type*)scanp[i];
				if(t != nil && ((t->kind & KindDirectIface) == 0 || (t->kind & KindNoPointers) == 0)) {
					p = scanp[i+1];
					if(minp <= p && p < maxp) {
						if(StackDebug >= 3)
							runtime·printf("adjust eface %p\n", p);
						if(t->size > PtrSize) // currently we always allocate such objects on the heap
							runtime·throw("large interface value found on stack");
						scanp[i+1] = p + delta;
					}
				}
				i++;
				break;
			case BitsIface:
				tab = (Itab*)scanp[i];
				if(tab != nil) {
					t = tab->type;
					//runtime·printf("          type=%p\n", t);
					if((t->kind & KindDirectIface) == 0 || (t->kind & KindNoPointers) == 0) {
						p = scanp[i+1];
						if(minp <= p && p < maxp) {
							if(StackDebug >= 3)
								runtime·printf("adjust iface %p\n", p);
							if(t->size > PtrSize) // currently we always allocate such objects on the heap
								runtime·throw("large interface value found on stack");
							scanp[i+1] = p + delta;
						}
					}
				}
				i++;
				break;
			}
			break;
		}
	}
}

// Note: the argument/return area is adjusted by the callee.
static bool
adjustframe(Stkframe *frame, void *arg)
{
	AdjustInfo *adjinfo;
	Func *f;
	StackMap *stackmap;
	int32 pcdata;
	BitVector bv;
	uintptr targetpc;

	adjinfo = arg;
	f = frame->fn;
	if(StackDebug >= 2)
		runtime·printf("    adjusting %s frame=[%p,%p] pc=%p continpc=%p\n", runtime·funcname(f), frame->sp, frame->fp, frame->pc, frame->continpc);
	if(f->entry == (uintptr)runtime·switchtoM) {
		// A special routine at the bottom of stack of a goroutine that does an onM call.
		// We will allow it to be copied even though we don't
		// have full GC info for it (because it is written in asm).
		return true;
	}
	targetpc = frame->continpc;
	if(targetpc == 0) {
		// Frame is dead.
		return true;
	}
	if(targetpc != f->entry)
		targetpc--;
	pcdata = runtime·pcdatavalue(f, PCDATA_StackMapIndex, targetpc);
	if(pcdata == -1)
		pcdata = 0; // in prologue

	// adjust local pointers
	if((byte*)frame->varp != (byte*)frame->sp) {
		stackmap = runtime·funcdata(f, FUNCDATA_LocalsPointerMaps);
		if(stackmap == nil)
			runtime·throw("no locals info");
		if(stackmap->n <= 0)
			runtime·throw("locals size info only");
		bv = runtime·stackmapdata(stackmap, pcdata);
		if(StackDebug >= 3)
			runtime·printf("      locals\n");
		adjustpointers((byte**)frame->varp - bv.n / BitsPerPointer, &bv, adjinfo, f);
	}
	// adjust inargs and outargs
	if(frame->arglen != 0) {
		if(frame->argmap != nil) {
			bv = *frame->argmap;
		} else {
			stackmap = runtime·funcdata(f, FUNCDATA_ArgsPointerMaps);
			if(stackmap == nil) {
				runtime·printf("size %d\n", (int32)frame->arglen);
				runtime·throw("no arg info");
			}
			bv = runtime·stackmapdata(stackmap, pcdata);
		}
		if(StackDebug >= 3)
			runtime·printf("      args\n");
		adjustpointers((byte**)frame->argp, &bv, adjinfo, nil);
	}
	return true;
}

static void
adjustctxt(G *gp, AdjustInfo *adjinfo)
{
	if(adjinfo->old.lo <= (uintptr)gp->sched.ctxt && (uintptr)gp->sched.ctxt < adjinfo->old.hi)
		gp->sched.ctxt = (byte*)gp->sched.ctxt + adjinfo->delta;
}

static void
adjustdefers(G *gp, AdjustInfo *adjinfo)
{
	Defer *d, **dp;
	Func *f;
	FuncVal *fn;
	StackMap *stackmap;
	BitVector bv;

	for(dp = &gp->defer, d = *dp; d != nil; dp = &d->link, d = *dp) {
		if(adjinfo->old.lo <= (uintptr)d && (uintptr)d < adjinfo->old.hi) {
			// The Defer record is on the stack.  Its fields will
			// get adjusted appropriately.
			// This only happens for runtime.main and runtime.gopanic now,
			// but a compiler optimization could do more of this.
			// If such an optimization were introduced, Defer.argp should
			// change to have pointer type so that it will be updated by
			// the stack copying. Today both of those on-stack defers
			// set argp = NoArgs, so no adjustment is necessary.
			*dp = (Defer*)((byte*)d + adjinfo->delta);
			continue;
		}
		if(d->argp == NoArgs)
			continue;
		if(d->argp < adjinfo->old.lo || adjinfo->old.hi <= d->argp) {
			runtime·printf("runtime: adjustdefers argp=%p stk=%p %p\n", d->argp, adjinfo->old.lo, adjinfo->old.hi);
			runtime·throw("adjustdefers: unexpected argp");
		}
		d->argp += adjinfo->delta;
		fn = d->fn;
		if(fn == nil) {
			// Defer of nil function.  It will panic when run.  See issue 8047.
			continue;
		}
		f = runtime·findfunc((uintptr)fn->fn);
		if(f == nil)
			runtime·throw("can't adjust unknown defer");
		if(StackDebug >= 4)
			runtime·printf("  checking defer %s\n", runtime·funcname(f));
		// Defer's FuncVal might be on the stack
		if(adjinfo->old.lo <= (uintptr)fn && (uintptr)fn < adjinfo->old.hi) {
			if(StackDebug >= 3)
				runtime·printf("    adjust defer fn %s\n", runtime·funcname(f));
			d->fn = (FuncVal*)((byte*)fn + adjinfo->delta);
		} else {
			// deferred function's args might point into the stack.
			if(StackDebug >= 3)
				runtime·printf("    adjust deferred args for %s\n", runtime·funcname(f));
			stackmap = runtime·funcdata(f, FUNCDATA_ArgsPointerMaps);
			if(stackmap == nil)
				runtime·throw("runtime: deferred function has no arg ptr map");
			bv = runtime·stackmapdata(stackmap, 0);
			adjustpointers(d->args, &bv, adjinfo, f);
		}
		// The FuncVal may have pointers in it, but fortunately for us
		// the compiler won't put pointers into the stack in a
		// heap-allocated FuncVal.
		// One day if we do need to check this, we can use the gc bits in the
		// heap to do the right thing (although getting the size will be expensive).
	}
}

static void
adjustpanics(G *gp, AdjustInfo *adjinfo)
{
	// Panic structs are all on the stack
	// and are adjusted by stack copying.
	// The only pointer we need to update is gp->panic, the head of the list.
	if(adjinfo->old.lo <= (uintptr)gp->panic && (uintptr)gp->panic < adjinfo->old.hi)
		gp->panic = (Panic*)((byte*)gp->panic + adjinfo->delta);
}

static void
adjustsudogs(G *gp, AdjustInfo *adjinfo)
{
	SudoG *s;
	byte *e;

	// the data elements pointed to by a SudoG structure
	// might be in the stack.
	for(s = gp->waiting; s != nil; s = s->waitlink) {
		e = s->elem;
		if(adjinfo->old.lo <= (uintptr)e && (uintptr)e < adjinfo->old.hi)
			s->elem = e + adjinfo->delta;
		e = (byte*)s->selectdone;
		if(adjinfo->old.lo <= (uintptr)e && (uintptr)e < adjinfo->old.hi)
			s->selectdone = (uint32*)(e + adjinfo->delta);
	}
}

// Copies gp's stack to a new stack of a different size.
static void
copystack(G *gp, uintptr newsize)
{
	Stack old, new;
	uintptr used;
	AdjustInfo adjinfo;
	uint32 oldstatus;
	bool (*cb)(Stkframe*, void*);

	if(gp->syscallsp != 0)
		runtime·throw("stack growth not allowed in system call");
	old = gp->stack;
	if(old.lo == 0)
		runtime·throw("nil stackbase");
	used = old.hi - gp->sched.sp;

	// allocate new stack
	new = runtime·stackalloc(newsize);

	if(StackDebug >= 1)
		runtime·printf("copystack gp=%p [%p %p %p]/%d -> [%p %p %p]/%d\n", gp, old.lo, old.hi-used, old.hi, (int32)(old.hi-old.lo), new.lo, new.hi-used, new.hi, (int32)newsize);
	
	// adjust pointers in the to-be-copied frames
	adjinfo.old = old;
	adjinfo.delta = new.hi - old.hi;
	cb = adjustframe;
	runtime·gentraceback(~(uintptr)0, ~(uintptr)0, 0, gp, 0, nil, 0x7fffffff, &cb, &adjinfo, false);
	
	// adjust other miscellaneous things that have pointers into stacks.
	adjustctxt(gp, &adjinfo);
	adjustdefers(gp, &adjinfo);
	adjustpanics(gp, &adjinfo);
	adjustsudogs(gp, &adjinfo);
	
	// copy the stack to the new location
	runtime·memmove((byte*)new.hi - used, (byte*)old.hi - used, used);

	oldstatus = runtime·readgstatus(gp);
	oldstatus &= ~Gscan;
	if(oldstatus == Gwaiting || oldstatus == Grunnable)
		runtime·casgstatus(gp, oldstatus, Gcopystack); // oldstatus is Gwaiting or Grunnable
	else
		runtime·throw("copystack: bad status, not Gwaiting or Grunnable");

	// Swap out old stack for new one
	gp->stack = new;
	gp->stackguard0 = new.lo + StackGuard; // NOTE: might clobber a preempt request
	gp->sched.sp = new.hi - used;

	runtime·casgstatus(gp, Gcopystack, oldstatus); // oldstatus is Gwaiting or Grunnable

	// free old stack
	runtime·stackfree(old);
}

// round x up to a power of 2.
int32
runtime·round2(int32 x)
{
	int32 s;

	s = 0;
	while((1 << s) < x)
		s++;
	return 1 << s;
}

// Called from runtime·morestack when more stack is needed.
// Allocate larger stack and relocate to new stack.
// Stack growth is multiplicative, for constant amortized cost.
//
// g->atomicstatus will be Grunning or Gscanrunning upon entry. 
// If the GC is trying to stop this g then it will set preemptscan to true.
void
runtime·newstack(void)
{
	int32 oldsize, newsize;
	uintptr sp;
	G *gp;
	Gobuf morebuf;

	if(g->m->morebuf.g->stackguard0 == (uintptr)StackFork)
		runtime·throw("stack growth after fork");
	if(g->m->morebuf.g != g->m->curg) {
		runtime·printf("runtime: newstack called from g=%p\n"
			"\tm=%p m->curg=%p m->g0=%p m->gsignal=%p\n",
			g->m->morebuf.g, g->m, g->m->curg, g->m->g0, g->m->gsignal);
		morebuf = g->m->morebuf;
		runtime·traceback(morebuf.pc, morebuf.sp, morebuf.lr, morebuf.g);
		runtime·throw("runtime: wrong goroutine in newstack");
	}
	if(g->throwsplit)
		runtime·throw("runtime: stack split at bad time");

	// The goroutine must be executing in order to call newstack,
	// so it must be Grunning or Gscanrunning.

	gp = g->m->curg;
	morebuf = g->m->morebuf;
	g->m->morebuf.pc = (uintptr)nil;
	g->m->morebuf.lr = (uintptr)nil;
	g->m->morebuf.sp = (uintptr)nil;

	runtime·casgstatus(gp, Grunning, Gwaiting);
	gp->waitreason = runtime·gostringnocopy((byte*)"stack growth");

	runtime·rewindmorestack(&gp->sched);

	if(gp->stack.lo == 0)
		runtime·throw("missing stack in newstack");
	sp = gp->sched.sp;
	if(thechar == '6' || thechar == '8') {
		// The call to morestack cost a word.
		sp -= sizeof(uintreg);
	}
	if(StackDebug >= 1 || sp < gp->stack.lo) {
		runtime·printf("runtime: newstack sp=%p stack=[%p, %p]\n"
			"\tmorebuf={pc:%p sp:%p lr:%p}\n"
			"\tsched={pc:%p sp:%p lr:%p ctxt:%p}\n",
			sp, gp->stack.lo, gp->stack.hi,
			g->m->morebuf.pc, g->m->morebuf.sp, g->m->morebuf.lr,
			gp->sched.pc, gp->sched.sp, gp->sched.lr, gp->sched.ctxt);
	}
	if(sp < gp->stack.lo) {
		runtime·printf("runtime: gp=%p, gp->status=%d\n ", (void*)gp, runtime·readgstatus(gp));
		runtime·printf("runtime: split stack overflow: %p < %p\n", sp, gp->stack.lo);
		runtime·throw("runtime: split stack overflow");
	}

	if(gp->stackguard0 == (uintptr)StackPreempt) {
		if(gp == g->m->g0)
			runtime·throw("runtime: preempt g0");
		if(g->m->p == nil && g->m->locks == 0)
			runtime·throw("runtime: g is running but p is not");
		if(gp->preemptscan) {
			runtime·gcphasework(gp);
			runtime·casgstatus(gp, Gwaiting, Grunning);
			gp->stackguard0 = gp->stack.lo + StackGuard;
			gp->preempt = false; 
			gp->preemptscan = false;        // Tells the GC premption was successful.
			runtime·gogo(&gp->sched);	// never return 
		}

		// Be conservative about where we preempt.
		// We are interested in preempting user Go code, not runtime code.
		if(g->m->locks || g->m->mallocing || g->m->gcing || g->m->p->status != Prunning) {
			// Let the goroutine keep running for now.
			// gp->preempt is set, so it will be preempted next time.
			gp->stackguard0 = gp->stack.lo + StackGuard;
			runtime·casgstatus(gp, Gwaiting, Grunning);
			runtime·gogo(&gp->sched);	// never return
		}
		// Act like goroutine called runtime.Gosched.
		runtime·casgstatus(gp, Gwaiting, Grunning);
		runtime·gosched_m(gp);	// never return
	}

	// Allocate a bigger segment and move the stack.
	oldsize = gp->stack.hi - gp->stack.lo;
	newsize = oldsize * 2;
	if(newsize > runtime·maxstacksize) {
		runtime·printf("runtime: goroutine stack exceeds %D-byte limit\n", (uint64)runtime·maxstacksize);
		runtime·throw("stack overflow");
	}

	// Note that the concurrent GC might be scanning the stack as we try to replace it.
	// copystack takes care of the appropriate coordination with the stack scanner.
	copystack(gp, newsize);
	if(StackDebug >= 1)
		runtime·printf("stack grow done\n");
	runtime·casgstatus(gp, Gwaiting, Grunning);
	runtime·gogo(&gp->sched);
}

#pragma textflag NOSPLIT
void
runtime·nilfunc(void)
{
	*(byte*)0 = 0;
}

// adjust Gobuf as if it executed a call to fn
// and then did an immediate gosave.
void
runtime·gostartcallfn(Gobuf *gobuf, FuncVal *fv)
{
	void *fn;

	if(fv != nil)
		fn = fv->fn;
	else
		fn = runtime·nilfunc;
	runtime·gostartcall(gobuf, fn, fv);
}

// Maybe shrink the stack being used by gp.
// Called at garbage collection time.
void
runtime·shrinkstack(G *gp)
{
	uintptr used, oldsize, newsize;

	if(runtime·readgstatus(gp) == Gdead)
		return;
	if(gp->stack.lo == 0)
		runtime·throw("missing stack in shrinkstack");

	oldsize = gp->stack.hi - gp->stack.lo;
	newsize = oldsize / 2;
	if(newsize < FixedStack)
		return; // don't shrink below the minimum-sized stack
	used = gp->stack.hi - gp->sched.sp;
	if(used >= oldsize / 4)
		return; // still using at least 1/4 of the segment.

	if(gp->syscallsp != 0) // TODO: can we handle this case?
		return;

#ifdef GOOS_windows
	if(gp->m != nil && gp->m->libcallsp != 0)
		return;
#endif
	if(StackDebug > 0)
		runtime·printf("shrinking stack %D->%D\n", (uint64)oldsize, (uint64)newsize);
	copystack(gp, newsize);
}

static void badc(void);

#pragma textflag NOSPLIT
void
runtime·morestackc(void)
{
	void (*fn)(void);
	
	fn = badc;
	runtime·onM(&fn);
}

static void
badc(void)
{
	runtime·throw("attempt to execute C code on Go stack");
}
