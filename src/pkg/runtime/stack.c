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
		n = (StackCacheNode*)runtime·SysAlloc(FixedStack*StackCacheBatch, &mstats.stacks_sys);
		if(n == nil)
			runtime·throw("out of memory (stackcacherefill)");
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
runtime·stackalloc(G *gp, uint32 n)
{
	uint32 pos;
	void *v;
	bool malloced;
	Stktop *top;

	// Stackalloc must be called on scheduler stack, so that we
	// never try to grow the stack during the code that stackalloc runs.
	// Doing so would cause a deadlock (issue 1547).
	if(g != m->g0)
		runtime·throw("stackalloc not on scheduler stack");
	if((n & (n-1)) != 0)
		runtime·throw("stack size not a power of 2");
	if(StackDebug >= 1)
		runtime·printf("stackalloc %d\n", n);

	gp->stacksize += n;
	if(runtime·debug.efence || StackFromSystem)
		return runtime·SysAlloc(ROUND(n, PageSize), &mstats.stacks_sys);

	// Minimum-sized stacks are allocated with a fixed-size free-list allocator,
	// but if we need a stack of a bigger size, we fall back on malloc
	// (assuming that inside malloc all the stack frames are small,
	// so that we do not deadlock).
	malloced = true;
	if(n == FixedStack || m->mallocing) {
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
		malloced = false;
	} else
		v = runtime·mallocgc(n, 0, FlagNoProfiling|FlagNoGC|FlagNoZero|FlagNoInvokeGC);

	top = (Stktop*)((byte*)v+n-sizeof(Stktop));
	runtime·memclr((byte*)top, sizeof(*top));
	top->malloced = malloced;
	return v;
}

void
runtime·stackfree(G *gp, void *v, Stktop *top)
{
	uint32 pos;
	uintptr n;

	n = (uintptr)(top+1) - (uintptr)v;
	if(StackDebug >= 1)
		runtime·printf("stackfree %p %d\n", v, (int32)n);
	gp->stacksize -= n;
	if(runtime·debug.efence || StackFromSystem) {
		if(runtime·debug.efence || StackFaultOnFree)
			runtime·SysFault(v, n);
		else
			runtime·SysFree(v, n, &mstats.stacks_sys);
		return;
	}
	if(top->malloced) {
		runtime·free(v);
		return;
	}
	if(n != FixedStack)
		runtime·throw("stackfree: bad fixed size");
	if(m->stackcachecnt == StackCacheSize)
		stackcacherelease();
	pos = m->stackcachepos;
	m->stackcache[pos] = v;
	m->stackcachepos = (pos + 1) % StackCacheSize;
	m->stackcachecnt++;
	m->stackinuse--;
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

	if(StackDebug >= 1) {
		runtime·printf("runtime: oldstack gobuf={pc:%p sp:%p lr:%p} cret=%p argsize=%p\n",
			top->gobuf.pc, top->gobuf.sp, top->gobuf.lr, (uintptr)m->cret, (uintptr)argsize);
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
	gp->panicwrap = top->panicwrap;
	runtime·stackfree(gp, old, top);

	gp->status = oldstatus;
	runtime·gogo(&gp->sched);
}

uintptr runtime·maxstacksize = 1<<20; // enough until runtime.main sets it for real

static uint8*
mapnames[] = {
	(uint8*)"---",
	(uint8*)"ptr",
	(uint8*)"iface",
	(uint8*)"eface",
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
	if(frame->varp < cinfo->stk || frame->varp >= cinfo->base) {
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
	if(frame->varp != (byte*)frame->sp) { // not in prologue (and has at least one local or outarg)
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

	cinfo.stk = (byte*)gp->stackguard - StackGuard;
	cinfo.base = (byte*)gp->stackbase + sizeof(Stktop);
	cinfo.frames = 0;
	runtime·gentraceback(~(uintptr)0, ~(uintptr)0, 0, gp, 0, nil, 0x7fffffff, checkframecopy, &cinfo, false);
	if(StackDebug >= 1 && cinfo.frames != -1)
		runtime·printf("copystack: %d copyable frames\n", cinfo.frames);
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
		case BitsNoPointer:
			break;
		case BitsPointer:
			p = scanp[i];
			if(f != nil && (byte*)0 < p && p < (byte*)PageSize) {
				// Looks like a junk value in a pointer slot.
				// Live analysis wrong?
				runtime·printf("%p: %p %s\n", &scanp[i], p, runtime·funcname(f));
				runtime·throw("bad pointer!");
			}
			if(minp <= p && p < maxp) {
				if(StackDebug >= 3)
					runtime·printf("adjust ptr %p\n", p);
				scanp[i] = p + delta;
			}
			break;
		case BitsEface:
			t = (Type*)scanp[i];
			if(t != nil && (t->size > PtrSize || (t->kind & KindNoPointers) == 0)) {
				p = scanp[i+1];
				if(minp <= p && p < maxp) {
					if(StackDebug >= 3)
						runtime·printf("adjust eface %p\n", p);
					if(t->size > PtrSize) // currently we always allocate such objects on the heap
						runtime·throw("large interface value found on stack");
					scanp[i+1] = p + delta;
				}
			}
			break;
		case BitsIface:
			tab = (Itab*)scanp[i];
			if(tab != nil) {
				t = tab->type;
				if(t->size > PtrSize || (t->kind & KindNoPointers) == 0) {
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
	BitVector *bv;

	adjinfo = arg;
	f = frame->fn;
	if(StackDebug >= 2)
		runtime·printf("    adjusting %s frame=[%p,%p]\n", runtime·funcname(f), frame->sp, frame->fp);
	if(f->entry == (uintptr)runtime·main)
		return true;
	pcdata = runtime·pcdatavalue(f, PCDATA_StackMapIndex, frame->pc);
	if(pcdata == -1)
		pcdata = 0; // in prologue

	// adjust local pointers
	if(frame->varp != (byte*)frame->sp) {
		stackmap = runtime·funcdata(f, FUNCDATA_LocalsPointerMaps);
		if(stackmap == nil)
			runtime·throw("no locals info");
		if(stackmap->n <= 0)
			runtime·throw("locals size info only");
		bv = runtime·stackmapdata(stackmap, pcdata);
		if(StackDebug >= 3)
			runtime·printf("      locals\n");
		adjustpointers((byte**)frame->varp - bv->n / BitsPerPointer, bv, adjinfo, f);
	}
	// adjust inargs and outargs
	if(frame->arglen != 0) {
		stackmap = runtime·funcdata(f, FUNCDATA_ArgsPointerMaps);
		if(stackmap == nil)
			runtime·throw("no arg info");
		bv = runtime·stackmapdata(stackmap, pcdata);
		if(StackDebug >= 3)
			runtime·printf("      args\n");
		adjustpointers((byte**)frame->argp, bv, adjinfo, nil);
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
	BitVector *bv;

	for(dp = &gp->defer, d = *dp; d != nil; dp = &d->link, d = *dp) {
		if(adjinfo->oldstk <= (byte*)d && (byte*)d < adjinfo->oldbase) {
			// The Defer record is on the stack.  Its fields will
			// get adjusted appropriately.
			// This only happens for runtime.main now, but a compiler
			// optimization could do more of this.
			*dp = (Defer*)((byte*)d + adjinfo->delta);
			continue;
		}
		if(d->argp < adjinfo->oldstk || adjinfo->oldbase <= d->argp)
			break; // a defer for the next segment
		f = runtime·findfunc((uintptr)d->fn->fn);
		if(f == nil) {
			runtime·printf("runtime: bad defer %p %d %d %p %p\n", d->fn->fn, d->siz, d->special, d->argp, d->pc);
			runtime·printf("caller %s\n", runtime·funcname(runtime·findfunc((uintptr)d->pc)));
			runtime·throw("can't adjust unknown defer");
		}
		if(StackDebug >= 4)
			runtime·printf("  checking defer %s\n", runtime·funcname(f));
		// Defer's FuncVal might be on the stack
		fn = d->fn;
		if(adjinfo->oldstk <= (byte*)fn && (byte*)fn < adjinfo->oldbase) {
			if(StackDebug >= 3)
				runtime·printf("    adjust defer fn %s\n", runtime·funcname(f));
			d->fn = (FuncVal*)((byte*)fn + adjinfo->delta);
		} else {
			// deferred function's closure args might point into the stack.
			if(StackDebug >= 3)
				runtime·printf("    adjust deferred args for %s\n", runtime·funcname(f));
			stackmap = runtime·funcdata(f, FUNCDATA_ArgsPointerMaps);
			if(stackmap == nil)
				runtime·throw("runtime: deferred function has no arg ptr map");
			bv = runtime·stackmapdata(stackmap, 0);
			adjustpointers(d->args, bv, adjinfo, f);
		}
		d->argp += adjinfo->delta;
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
	bool malloced;

	if(gp->syscallstack != 0)
		runtime·throw("can't handle stack copy in syscall yet");
	oldstk = (byte*)gp->stackguard - StackGuard;
	oldbase = (byte*)gp->stackbase + sizeof(Stktop);
	oldsize = oldbase - oldstk;
	used = oldbase - (byte*)gp->sched.sp;
	oldtop = (Stktop*)gp->stackbase;

	// allocate new stack
	newstk = runtime·stackalloc(gp, newsize);
	newbase = newstk + newsize;
	newtop = (Stktop*)(newbase - sizeof(Stktop));
	malloced = newtop->malloced;

	if(StackDebug >= 1)
		runtime·printf("copystack [%p %p]/%d -> [%p %p]/%d\n", oldstk, oldbase, (int32)oldsize, newstk, newbase, (int32)newsize);
	USED(oldsize);
	
	// adjust pointers in the to-be-copied frames
	adjinfo.oldstk = oldstk;
	adjinfo.oldbase = oldbase;
	adjinfo.delta = newbase - oldbase;
	runtime·gentraceback(~(uintptr)0, ~(uintptr)0, 0, gp, 0, nil, nframes, adjustframe, &adjinfo, false);
	
	// adjust other miscellaneous things that have pointers into stacks.
	adjustctxt(gp, &adjinfo);
	adjustdefers(gp, &adjinfo);
	
	// copy the stack (including Stktop) to the new location
	runtime·memmove(newbase - used, oldbase - used, used);
	newtop->malloced = malloced;
	
	// Swap out old stack for new one
	gp->stackbase = (uintptr)newtop;
	gp->stackguard = (uintptr)newstk + StackGuard;
	gp->stackguard0 = (uintptr)newstk + StackGuard; // NOTE: might clobber a preempt request
	if(gp->stack0 == (uintptr)oldstk)
		gp->stack0 = (uintptr)newstk;
	gp->sched.sp = (uintptr)(newbase - used);

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
void
runtime·newstack(void)
{
	int32 framesize, argsize, oldstatus, oldsize, newsize, nframes;
	Stktop *top, *oldtop;
	byte *stk, *oldstk, *oldbase;
	uintptr sp;
	uintptr *src, *dst, *dstend;
	G *gp;
	Gobuf label;
	bool newstackcall;

	if(m->forkstackguard)
		runtime·throw("split stack after fork");
	if(m->morebuf.g != m->curg) {
		runtime·printf("runtime: newstack called from g=%p\n"
			"\tm=%p m->curg=%p m->g0=%p m->gsignal=%p\n",
			m->morebuf.g, m, m->curg, m->g0, m->gsignal);
		runtime·throw("runtime: wrong goroutine in newstack");
	}

	// gp->status is usually Grunning, but it could be Gsyscall if a stack split
	// happens during a function call inside entersyscall.
	gp = m->curg;
	oldstatus = gp->status;

	framesize = m->moreframesize;
	argsize = m->moreargsize;
	gp->status = Gwaiting;
	gp->waitreason = "stack split";
	newstackcall = framesize==1;
	if(newstackcall)
		framesize = 0;

	// For newstackcall the context already points to beginning of runtime·newstackcall.
	if(!newstackcall)
		runtime·rewindmorestack(&gp->sched);

	sp = gp->sched.sp;
	if(thechar == '6' || thechar == '8') {
		// The call to morestack cost a word.
		sp -= sizeof(uintptr);
	}
	if(StackDebug >= 1 || sp < gp->stackguard - StackGuard) {
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
		if(oldstatus == Grunning && m->p == nil && m->locks == 0)
			runtime·throw("runtime: g is running but p is not");
		if(oldstatus == Gsyscall && m->locks == 0)
			runtime·throw("runtime: stack split during syscall");
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
			copystack(gp, nframes, newsize);
			if(StackDebug >= 1)
				runtime·printf("stack grow done\n");
			if(gp->stacksize > runtime·maxstacksize) {
				runtime·printf("runtime: goroutine stack exceeds %D-byte limit\n", (uint64)runtime·maxstacksize);
				runtime·throw("stack overflow");
			}
			gp->status = oldstatus;
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
		runtime·printf("\t-> new stack [%p, %p]\n", stk, top);
	}

	top->stackbase = gp->stackbase;
	top->stackguard = gp->stackguard;
	top->gobuf = m->morebuf;
	top->argp = m->moreargp;
	top->argsize = argsize;
	m->moreargp = nil;
	m->morebuf.pc = (uintptr)nil;
	m->morebuf.lr = (uintptr)nil;
	m->morebuf.sp = (uintptr)nil;

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
	label.g = m->curg;
	if(newstackcall)
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

// Maybe shrink the stack being used by gp.
// Called at garbage collection time.
void
runtime·shrinkstack(G *gp)
{
	int32 nframes;
	byte *oldstk, *oldbase;
	uintptr used, oldsize, newsize;
	MSpan *span;

	if(!runtime·copystack)
		return;
	oldstk = (byte*)gp->stackguard - StackGuard;
	oldbase = (byte*)gp->stackbase + sizeof(Stktop);
	oldsize = oldbase - oldstk;
	newsize = oldsize / 2;
	if(newsize < FixedStack)
		return; // don't shrink below the minimum-sized stack
	used = oldbase - (byte*)gp->sched.sp;
	if(used >= oldsize / 4)
		return; // still using at least 1/4 of the segment.

	// To shrink to less than 1/2 a page, we need to copy.
	if(newsize < PageSize/2) {
		if(gp->syscallstack != (uintptr)nil) // TODO: can we handle this case?
			return;
#ifdef GOOS_windows
		if(gp->m != nil && gp->m->libcallsp != 0)
			return;
#endif
		nframes = copyabletopsegment(gp);
		if(nframes == -1)
			return;
		copystack(gp, nframes, newsize);
		return;
	}

	// To shrink a stack of one page size or more, we can shrink it
	// without copying.  Just deallocate the lower half.
	span = runtime·MHeap_LookupMaybe(&runtime·mheap, oldstk);
	if(span == nil)
		return; // stack allocated outside heap.  Can't shrink it.  Can happen if stack is allocated while inside malloc.  TODO: shrink by copying?
	if(span->elemsize != oldsize)
		runtime·throw("span element size doesn't match stack size");
	if((uintptr)oldstk != span->start << PageShift)
		runtime·throw("stack not at start of span");

	if(StackDebug)
		runtime·printf("shrinking stack in place %p %X->%X\n", oldstk, oldsize, newsize);

	// new stack guard for smaller stack
	gp->stackguard = (uintptr)oldstk + newsize + StackGuard;
	gp->stackguard0 = (uintptr)oldstk + newsize + StackGuard;
	if(gp->stack0 == (uintptr)oldstk)
		gp->stack0 = (uintptr)oldstk + newsize;
	gp->stacksize -= oldsize - newsize;

	// Free bottom half of the stack.
	if(runtime·debug.efence || StackFromSystem) {
		if(runtime·debug.efence || StackFaultOnFree)
			runtime·SysFault(oldstk, newsize);
		else
			runtime·SysFree(oldstk, newsize, &mstats.stacks_sys);
		return;
	}
	// First, we trick malloc into thinking
	// we allocated the stack as two separate half-size allocs.  Then the
	// free() call does the rest of the work for us.
	if(oldsize == PageSize) {
		// convert span of 1 PageSize object to a span of 2
		// PageSize/2 objects.
		span->ref = 2;
		span->sizeclass = runtime·SizeToClass(PageSize/2);
		span->elemsize = PageSize/2;
	} else {
		// convert span of n>1 pages into two spans of n/2 pages each.
		runtime·MHeap_SplitSpan(&runtime·mheap, span);
	}
	runtime·free(oldstk);
}
