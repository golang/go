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
#include "../../cmd/ld/textflag.h"

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

void*
runtime·stackalloc(G *gp, uint32 n)
{
	uint8 order;
	uint32 n2;
	void *v;
	Stktop *top;
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

	gp->stacksize += n;
	if(runtime·debug.efence || StackFromSystem) {
		v = runtime·sysAlloc(ROUND(n, PageSize), &mstats.stacks_sys);
		if(v == nil)
			runtime·throw("out of memory (stackalloc)");
		return v;
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
	top = (Stktop*)((byte*)v+n-sizeof(Stktop));
	runtime·memclr((byte*)top, sizeof(*top));
	if(raceenabled)
		runtime·racemalloc(v, n);
	if(StackDebug >= 1)
		runtime·printf("  allocated %p\n", v);
	return v;
}

void
runtime·stackfree(G *gp, void *v, Stktop *top)
{
	uint8 order;
	uintptr n, n2;
	MSpan *s;
	MLink *x;
	MCache *c;
	
	n = (uintptr)(top+1) - (uintptr)v;
	if(n & (n-1))
		runtime·throw("stack not a power of 2");
	if(StackDebug >= 1) {
		runtime·printf("stackfree %p %d\n", v, (int32)n);
		runtime·memclr(v, n); // for testing, clobber stack data
	}
	gp->stacksize -= n;
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

	gp = g->m->curg;
	top = (Stktop*)gp->stackbase;
	if(top == nil)
		runtime·throw("nil stackbase");
	old = (byte*)gp->stackguard - StackGuard;
	sp = (byte*)top;
	argsize = top->argsize;

	if(StackDebug >= 1) {
		runtime·printf("runtime: oldstack gobuf={pc:%p sp:%p lr:%p} cret=%p argsize=%p\n",
			top->gobuf.pc, top->gobuf.sp, top->gobuf.lr, (uintptr)g->m->cret, (uintptr)argsize);
	}

	gp->sched = top->gobuf;
	gp->sched.ret = g->m->cret;
	g->m->cret = 0; // drop reference
	// gp->status is usually Grunning, but it could be Gsyscall if a stack overflow
	// happens during a function call inside entersyscall.

	oldstatus = runtime·readgstatus(gp);
	oldstatus &= ~Gscan;
	if(oldstatus != Grunning && oldstatus != Gsyscall) {
		runtime·printf("runtime: oldstack status=%d\n", oldstatus);
		runtime·throw("oldstack");
	}
	runtime·casgstatus(gp, oldstatus, Gcopystack);
	gp->waitreason = runtime·gostringnocopy((byte*)"stack unsplit");	

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
	gp->panicwrap = top->panicwrap;
	runtime·stackfree(gp, old, top);
	runtime·casgstatus(gp, Gcopystack, oldstatus); // oldstatus is Grunning or Gsyscall
	runtime·gogo(&gp->sched);
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
// (arm: TODO)

typedef struct CopyableInfo CopyableInfo;
struct CopyableInfo {
	byte *stk;	// bottom address of segment
	byte *base;	// top address of segment (including Stktop)
	int32 frames;	// count of copyable frames (-1 = not copyable)
};

void runtime·main(void);
void runtime·switchtoM(void(*)(void));

static bool
checkframecopy(Stkframe *frame, void *arg)
{
	CopyableInfo *cinfo;
	Func *f;
	StackMap *stackmap;

	cinfo = arg;
	f = frame->fn;
	if(StackDebug >= 2)
		runtime·printf("    checking %s frame=[%p,%p] stk=[%p,%p]\n", runtime·funcname(f), frame->sp, frame->fp, cinfo->stk, cinfo->base);
	// if we're not in the segment any more, return immediately.
	if((byte*)frame->varp < cinfo->stk || (byte*)frame->varp >= cinfo->base) {
		if(StackDebug >= 2)
			runtime·printf("    <next segment>\n");
		return false; // stop traceback
	}
	if(f->entry == (uintptr)runtime·main) {
		// A special routine at the TOS of the main routine.
		// We will allow it to be copied even though we don't
		// have full GC info for it (because it is written in C).
		cinfo->frames++;
		return false; // stop traceback
	}
	if(f->entry == (uintptr)runtime·switchtoM) {
		// A special routine at the bottom of stack of a goroutine that does onM call.
		// We will allow it to be copied even though we don't
		// have full GC info for it (because it is written in asm).
		cinfo->frames++;
		return true;
	}
	if((byte*)frame->varp != (byte*)frame->sp) { // not in prologue (and has at least one local or outarg)
		stackmap = runtime·funcdata(f, FUNCDATA_LocalsPointerMaps);
		if(stackmap == nil) {
			cinfo->frames = -1;
			if(StackDebug >= 1)
				runtime·printf("copystack: no locals info for %s\n", runtime·funcname(f));
			return false;
		}
		if(stackmap->n <= 0) {
			cinfo->frames = -1;
			if(StackDebug >= 1)
				runtime·printf("copystack: locals size info only for %s\n", runtime·funcname(f));
			return false;
		}
	}
	if(frame->arglen != 0) {
		stackmap = runtime·funcdata(f, FUNCDATA_ArgsPointerMaps);
		if(stackmap == nil) {
			cinfo->frames = -1;
			if(StackDebug >= 1)
				runtime·printf("copystack: no arg info for %s\n", runtime·funcname(f));
			return false;
		}
	}
	cinfo->frames++;
	return true; // this frame is ok; keep going
}

// If the top segment of the stack contains an uncopyable
// frame, return -1.  Otherwise return the number of frames
// in the top segment, all of which are copyable.
static int32
copyabletopsegment(G *gp)
{
	CopyableInfo cinfo;
	Defer *d;
	Func *f;
	FuncVal *fn;
	StackMap *stackmap;
	bool (*cb)(Stkframe*, void*);

	if(gp->stackbase == 0)
		runtime·throw("stackbase == 0");
	cinfo.stk = (byte*)gp->stackguard - StackGuard;
	cinfo.base = (byte*)gp->stackbase + sizeof(Stktop);
	cinfo.frames = 0;

	// Check that each frame is copyable.  As a side effect,
	// count the frames.
	cb = checkframecopy;
	runtime·gentraceback(~(uintptr)0, ~(uintptr)0, 0, gp, 0, nil, 0x7fffffff, &cb, &cinfo, false);
	if(StackDebug >= 1 && cinfo.frames != -1)
		runtime·printf("copystack: %d copyable frames\n", cinfo.frames);

	if(cinfo.frames == -1)
		return -1;

	// Check to make sure all Defers are copyable
	for(d = gp->defer; d != nil; d = d->link) {
		if(cinfo.stk <= (byte*)d && (byte*)d < cinfo.base) {
			// Defer is on the stack.  Its copyableness has
			// been established during stack walking.
			// For now, this only happens with the Defer in runtime.main.
			continue;
		}
		if((byte*)d->argp < cinfo.stk || cinfo.base <= (byte*)d->argp)
			break; // a defer for the next segment
		fn = d->fn;
		if(fn == nil) // See issue 8047
			continue;
		f = runtime·findfunc((uintptr)fn->fn);
		if(f == nil) {
			if(StackDebug >= 1)
				runtime·printf("copystack: no func for deferred pc %p\n", fn->fn);
			return -1;
		}

		// Check to make sure we have an args pointer map for the defer's args.
		// We only need the args map, but we check
		// for the locals map also, because when the locals map
		// isn't provided it means the ptr map came from C and
		// C (particularly, cgo) lies to us.  See issue 7695.
		stackmap = runtime·funcdata(f, FUNCDATA_ArgsPointerMaps);
		if(stackmap == nil || stackmap->n <= 0) {
			if(StackDebug >= 1)
				runtime·printf("copystack: no arg info for deferred %s\n", runtime·funcname(f));
			return -1;
		}
		stackmap = runtime·funcdata(f, FUNCDATA_LocalsPointerMaps);
		if(stackmap == nil || stackmap->n <= 0) {
			if(StackDebug >= 1)
				runtime·printf("copystack: no local info for deferred %s\n", runtime·funcname(f));
			return -1;
		}

		if(cinfo.stk <= (byte*)fn && (byte*)fn < cinfo.base) {
			// FuncVal is on the stack.  Again, its copyableness
			// was established during stack walking.
			continue;
		}
		// The FuncVal may have pointers in it, but fortunately for us
		// the compiler won't put pointers into the stack in a
		// heap-allocated FuncVal.
		// One day if we do need to check this, we'll need maps of the
		// pointerness of the closure args.  The only place we have that map
		// right now is in the gc program for the FuncVal.  Ugh.
	}

	return cinfo.frames;
}

typedef struct AdjustInfo AdjustInfo;
struct AdjustInfo {
	byte *oldstk;	// bottom address of segment
	byte *oldbase;	// top address of segment (after Stktop)
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
	
	minp = adjinfo->oldstk;
	maxp = adjinfo->oldbase;
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
	if(f->entry == (uintptr)runtime·main ||
		f->entry == (uintptr)runtime·switchtoM)
		return true;
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
		stackmap = runtime·funcdata(f, FUNCDATA_ArgsPointerMaps);
		if(stackmap == nil) {
			runtime·printf("size %d\n", (int32)frame->arglen);
			runtime·throw("no arg info");
		}
		bv = runtime·stackmapdata(stackmap, pcdata);
		if(StackDebug >= 3)
			runtime·printf("      args\n");
		adjustpointers((byte**)frame->argp, &bv, adjinfo, nil);
	}
	return true;
}

static void
adjustctxt(G *gp, AdjustInfo *adjinfo)
{
	if(adjinfo->oldstk <= (byte*)gp->sched.ctxt && (byte*)gp->sched.ctxt < adjinfo->oldbase)
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
		if(adjinfo->oldstk <= (byte*)d && (byte*)d < adjinfo->oldbase) {
			// The Defer record is on the stack.  Its fields will
			// get adjusted appropriately.
			// This only happens for runtime.main now, but a compiler
			// optimization could do more of this.
			*dp = (Defer*)((byte*)d + adjinfo->delta);
			continue;
		}
		if((byte*)d->argp < adjinfo->oldstk || adjinfo->oldbase <= (byte*)d->argp)
			break; // a defer for the next segment
		fn = d->fn;
		if(fn == nil) {
			// Defer of nil function.  It will panic when run, and there
			// aren't any args to adjust.  See issue 8047.
			d->argp += adjinfo->delta;
			continue;
		}
		f = runtime·findfunc((uintptr)fn->fn);
		if(f == nil)
			runtime·throw("can't adjust unknown defer");
		if(StackDebug >= 4)
			runtime·printf("  checking defer %s\n", runtime·funcname(f));
		// Defer's FuncVal might be on the stack
		if(adjinfo->oldstk <= (byte*)fn && (byte*)fn < adjinfo->oldbase) {
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
		d->argp += adjinfo->delta;
	}
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
		if(adjinfo->oldstk <= e && e < adjinfo->oldbase)
			s->elem = e + adjinfo->delta;
	}
}

// Copies the top stack segment of gp to a new stack segment of a
// different size.  The top segment must contain nframes frames.
static void
copystack(G *gp, uintptr nframes, uintptr newsize)
{
	byte *oldstk, *oldbase, *newstk, *newbase;
	uintptr oldsize, used;
	AdjustInfo adjinfo;
	Stktop *oldtop, *newtop;
	uint32 oldstatus;
	bool (*cb)(Stkframe*, void*);

	if(gp->syscallstack != 0)
		runtime·throw("can't handle stack copy in syscall yet");
	oldstk = (byte*)gp->stackguard - StackGuard;
	if(gp->stackbase == 0)
		runtime·throw("nil stackbase");
	oldbase = (byte*)gp->stackbase + sizeof(Stktop);
	oldsize = oldbase - oldstk;
	used = oldbase - (byte*)gp->sched.sp;
	oldtop = (Stktop*)gp->stackbase;

	// allocate new stack
	newstk = runtime·stackalloc(gp, newsize);
	newbase = newstk + newsize;
	newtop = (Stktop*)(newbase - sizeof(Stktop));

	if(StackDebug >= 1)
		runtime·printf("copystack gp=%p [%p %p]/%d -> [%p %p]/%d\n", gp, oldstk, oldbase, (int32)oldsize, newstk, newbase, (int32)newsize);
	USED(oldsize);
	
	// adjust pointers in the to-be-copied frames
	adjinfo.oldstk = oldstk;
	adjinfo.oldbase = oldbase;
	adjinfo.delta = newbase - oldbase;
	cb = adjustframe;
	runtime·gentraceback(~(uintptr)0, ~(uintptr)0, 0, gp, 0, nil, nframes, &cb, &adjinfo, false);
	
	// adjust other miscellaneous things that have pointers into stacks.
	adjustctxt(gp, &adjinfo);
	adjustdefers(gp, &adjinfo);
	adjustsudogs(gp, &adjinfo);
	
	// copy the stack (including Stktop) to the new location
	runtime·memmove(newbase - used, oldbase - used, used);
	oldstatus = runtime·readgstatus(gp);
	oldstatus &= ~Gscan;
	if (oldstatus == Gwaiting || oldstatus == Grunnable)
		runtime·casgstatus(gp, oldstatus, Gcopystack); // oldstatus is Gwaiting or Grunnable
	else
		runtime·throw("copystack: bad status, not Gwaiting or Grunnable");
	// Swap out old stack for new one
	gp->stackbase = (uintptr)newtop;
	gp->stackguard = (uintptr)newstk + StackGuard;
	gp->stackguard0 = (uintptr)newstk + StackGuard; // NOTE: might clobber a preempt request
	if(gp->stack0 == (uintptr)oldstk)
		gp->stack0 = (uintptr)newstk;
	gp->sched.sp = (uintptr)(newbase - used);

	runtime·casgstatus(gp, Gcopystack, oldstatus); // oldstatus is Gwaiting or Grunnable

	// free old stack
	runtime·stackfree(gp, oldstk, oldtop);
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

// Called from runtime·newstackcall or from runtime·morestack when a new
// stack segment is needed.  Allocate a new stack big enough for
// m->moreframesize bytes, copy m->moreargsize bytes to the new frame,
// and then act as though runtime·lessstack called the function at
// m->morepc.
//
// g->atomicstatus will be Grunning, Gsyscall or Gscanrunning, Gscansyscall upon entry. 
// If the GC is trying to stop this g then it will set preemptscan to true.
void
runtime·newstack(void)
{
	int32 framesize, argsize, oldstatus, oldsize, newsize, nframes;
	Stktop *top, *oldtop;
	byte *stk, *oldstk, *oldbase;
	uintptr sp;
	uintptr *src, *dst, *dstend;
	G *gp;
	Gobuf label, morebuf;
	void *moreargp;
	bool newstackcall;

	if(g->m->forkstackguard)
		runtime·throw("split stack after fork");
	if(g->m->morebuf.g != g->m->curg) {
		runtime·printf("runtime: newstack called from g=%p\n"
			"\tm=%p m->curg=%p m->g0=%p m->gsignal=%p\n",
			g->m->morebuf.g, g->m, g->m->curg, g->m->g0, g->m->gsignal);
		runtime·throw("runtime: wrong goroutine in newstack");
	}

	// The goroutine must be executing in order to call newstack, so the possible states are
	// Grunning and Gsyscall (and, due to GC, also Gscanrunning and Gscansyscall).	

	// gp->status is usually Grunning, but it could be Gsyscall if a stack overflow
	// happens during a function call inside entersyscall.
	gp = g->m->curg;
	oldstatus = runtime·readgstatus(gp) & ~Gscan;
	framesize = g->m->moreframesize;
	argsize = g->m->moreargsize;
	moreargp = g->m->moreargp;
	g->m->moreargp = nil;
	morebuf = g->m->morebuf;
	g->m->morebuf.pc = (uintptr)nil;
	g->m->morebuf.lr = (uintptr)nil;
	g->m->morebuf.sp = (uintptr)nil;

	runtime·casgstatus(gp, oldstatus, Gwaiting); // oldstatus is not in a Gscan status
	gp->waitreason = runtime·gostringnocopy((byte*)"stack growth");
	newstackcall = framesize==1;
	if(newstackcall)
		framesize = 0;

	// For newstackcall the context already points to beginning of runtime·newstackcall.
	if(!newstackcall)
		runtime·rewindmorestack(&gp->sched);

	if(gp->stackbase == 0)
		runtime·throw("nil stackbase");
	sp = gp->sched.sp;
	if(thechar == '6' || thechar == '8') {
		// The call to morestack cost a word.
		sp -= sizeof(uintreg);
	}
	if(StackDebug >= 1 || sp < gp->stackguard - StackGuard) {
		runtime·printf("runtime: newstack framesize=%p argsize=%p sp=%p stack=[%p, %p]\n"
			"\tmorebuf={pc:%p sp:%p lr:%p}\n"
			"\tsched={pc:%p sp:%p lr:%p ctxt:%p}\n",
			(uintptr)framesize, (uintptr)argsize, sp, gp->stackguard - StackGuard, gp->stackbase,
			g->m->morebuf.pc, g->m->morebuf.sp, g->m->morebuf.lr,
			gp->sched.pc, gp->sched.sp, gp->sched.lr, gp->sched.ctxt);
	}
	if(sp < gp->stackguard - StackGuard) {
		runtime·printf("runtime: gp=%p, gp->status=%d, oldstatus=%d\n ", (void*)gp, runtime·readgstatus(gp), oldstatus);
		runtime·printf("runtime: split stack overflow: %p < %p\n", sp, gp->stackguard - StackGuard);
		runtime·throw("runtime: split stack overflow");
	}

	if(argsize % sizeof(uintptr) != 0) {
		runtime·printf("runtime: stack growth with misaligned argsize %d\n", argsize);
		runtime·throw("runtime: stack growth argsize");
	}

	if(gp->stackguard0 == (uintptr)StackPreempt) {
		if(gp == g->m->g0)
			runtime·throw("runtime: preempt g0");
		if(oldstatus == Grunning && g->m->p == nil && g->m->locks == 0)
			runtime·throw("runtime: g is running but p is not");
		if(oldstatus == Gsyscall && g->m->locks == 0)
			runtime·throw("runtime: stack growth during syscall");

		// Be conservative about where we preempt.
		// We are interested in preempting user Go code, not runtime code.
		if(oldstatus != Grunning || g->m->locks || g->m->mallocing || g->m->gcing || g->m->p->status != Prunning) {
			// Let the goroutine keep running for now.
			// gp->preempt is set, so it will be preempted next time.
			gp->stackguard0 = gp->stackguard;
			runtime·casgstatus(gp, Gwaiting, oldstatus); // oldstatus is Gsyscall or Grunning
			runtime·gogo(&gp->sched);	// never return
		}
		// Act like goroutine called runtime.Gosched.
		runtime·casgstatus(gp, Gwaiting, oldstatus); // oldstatus is Gsyscall or Grunning
		runtime·gosched_m(gp);	// never return
	}

	// If every frame on the top segment is copyable, allocate a bigger segment
	// and move the segment instead of allocating a new segment.
	if(runtime·copystack) {
		if(!runtime·precisestack)
			runtime·throw("can't copy stacks without precise stacks");
		nframes = copyabletopsegment(gp);
		if(nframes != -1) {
			oldstk = (byte*)gp->stackguard - StackGuard;
			oldbase = (byte*)gp->stackbase + sizeof(Stktop);
			oldsize = oldbase - oldstk;
			newsize = oldsize * 2;
			// Note that the concurrent GC might be scanning the stack as we try to replace it.
			// copystack takes care of the appropriate coordination with the stack scanner.
			copystack(gp, nframes, newsize);
			if(StackDebug >= 1)
				runtime·printf("stack grow done\n");
			if(gp->stacksize > runtime·maxstacksize) {
				runtime·printf("runtime: goroutine stack exceeds %D-byte limit\n", (uint64)runtime·maxstacksize);
				runtime·throw("stack overflow");
			}
			runtime·casgstatus(gp, Gwaiting, oldstatus); // oldstatus is Gsyscall or Grunning
			runtime·gogo(&gp->sched);
		}
		// TODO: if stack is uncopyable because we're in C code, patch return value at
		// end of C code to trigger a copy as soon as C code exits.  That way, we'll
		// have stack available if we get this deep again.
	}

	// allocate new segment.
	framesize += argsize;
	framesize += StackExtra;	// room for more functions, Stktop.
	if(framesize < StackMin)
		framesize = StackMin;
	framesize += StackSystem;
	framesize = runtime·round2(framesize);
	stk = runtime·stackalloc(gp, framesize);
	if(gp->stacksize > runtime·maxstacksize) {
		runtime·printf("runtime: goroutine stack exceeds %D-byte limit\n", (uint64)runtime·maxstacksize);
		runtime·throw("stack overflow");
	}
	top = (Stktop*)(stk+framesize-sizeof(*top));

	if(StackDebug >= 1) {
		runtime·printf("\t-> new stack gp=%p [%p, %p]\n", gp, stk, top);
	}

	top->stackbase = gp->stackbase;
	top->stackguard = gp->stackguard;
	top->gobuf = morebuf;
	top->argp = moreargp;
	top->argsize = argsize;

	// copy flag from panic
	top->panic = gp->ispanic;
	gp->ispanic = false;
	
	// if this isn't a panic, maybe we're splitting the stack for a panic.
	// if we're splitting in the top frame, propagate the panic flag
	// forward so that recover will know we're in a panic.
	oldtop = (Stktop*)top->stackbase;
	if(oldtop != nil && oldtop->panic && top->argp == (byte*)oldtop - oldtop->argsize - gp->panicwrap)
		top->panic = true;

	top->panicwrap = gp->panicwrap;
	gp->panicwrap = 0;

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
	label.g = g->m->curg;
	if(newstackcall)
		runtime·gostartcallfn(&label, (FuncVal*)g->m->cret);
	else {
		runtime·gostartcall(&label, (void(*)(void))gp->sched.pc, gp->sched.ctxt);
		gp->sched.ctxt = nil;
	}
	runtime·casgstatus(gp, Gwaiting, oldstatus); // oldstatus is Grunning or Gsyscall
	runtime·gogo(&label);

	*(int32*)345 = 123;	// never return
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
	int32 nframes;
	byte *oldstk, *oldbase;
	uintptr used, oldsize, newsize;

	if(!runtime·copystack)
		return;
	if(runtime·readgstatus(gp) == Gdead)
		return;
	if(gp->stackbase == 0)
		runtime·throw("stackbase == 0");
	//return; // TODO: why does this happen?
	oldstk = (byte*)gp->stackguard - StackGuard;
	oldbase = (byte*)gp->stackbase + sizeof(Stktop);
	oldsize = oldbase - oldstk;
	newsize = oldsize / 2;
	if(newsize < FixedStack)
		return; // don't shrink below the minimum-sized stack
	used = oldbase - (byte*)gp->sched.sp;
	if(used >= oldsize / 4)
		return; // still using at least 1/4 of the segment.

	if(gp->syscallstack != (uintptr)nil) // TODO: can we handle this case?
		return;
#ifdef GOOS_windows
	if(gp->m != nil && gp->m->libcallsp != 0)
		return;
#endif
	if(StackDebug > 0)
		runtime·printf("shrinking stack %D->%D\n", (uint64)oldsize, (uint64)newsize);
	nframes = copyabletopsegment(gp);
	if(nframes == -1)
		return;
	copystack(gp, nframes, newsize);
}
