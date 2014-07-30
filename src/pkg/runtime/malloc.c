// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// See malloc.h for overview.
//
// TODO(rsc): double-check stats.

#include "runtime.h"
#include "arch_GOARCH.h"
#include "malloc.h"
#include "type.h"
#include "typekind.h"
#include "race.h"
#include "stack.h"
#include "../../cmd/ld/textflag.h"

// Mark mheap as 'no pointers', it does not contain interesting pointers but occupies ~45K.
#pragma dataflag NOPTR
MHeap runtime·mheap;
#pragma dataflag NOPTR
MStats runtime·memstats;

void* runtime·cmallocgc(uintptr size, Type *typ, uint32 flag, void **ret);

void*
runtime·mallocgc(uintptr size, Type *typ, uint32 flag)
{
	void *ret;

	// Call into the Go version of mallocgc.
	// TODO: maybe someday we can get rid of this.  It is
	// probably the only location where we run Go code on the M stack.
	runtime·cmallocgc(size, typ, flag, &ret);
	return ret;
}

void*
runtime·malloc(uintptr size)
{
	return runtime·mallocgc(size, nil, FlagNoInvokeGC);
}

// Free the object whose base pointer is v.
void
runtime·free(void *v)
{
	int32 sizeclass;
	MSpan *s;
	MCache *c;
	uintptr size;

	if(v == nil)
		return;
	
	// If you change this also change mgc0.c:/^sweep,
	// which has a copy of the guts of free.

	if(g->m->mallocing)
		runtime·throw("malloc/free - deadlock");
	g->m->mallocing = 1;

	if(!runtime·mlookup(v, nil, nil, &s)) {
		runtime·printf("free %p: not an allocated block\n", v);
		runtime·throw("free runtime·mlookup");
	}
	size = s->elemsize;
	sizeclass = s->sizeclass;
	// Objects that are smaller than TinySize can be allocated using tiny alloc,
	// if then such object is combined with an object with finalizer, we will crash.
	if(size < TinySize)
		runtime·throw("freeing too small block");

	if(runtime·debug.allocfreetrace)
		runtime·tracefree(v, size);

	// Ensure that the span is swept.
	// If we free into an unswept span, we will corrupt GC bitmaps.
	runtime·MSpan_EnsureSwept(s);

	if(s->specials != nil)
		runtime·freeallspecials(s, v, size);

	c = g->m->mcache;
	if(sizeclass == 0) {
		// Large object.
		s->needzero = 1;
		// Must mark v freed before calling unmarkspan and MHeap_Free:
		// they might coalesce v into other spans and change the bitmap further.
		runtime·markfreed(v);
		runtime·unmarkspan(v, s->npages<<PageShift);
		// NOTE(rsc,dvyukov): The original implementation of efence
		// in CL 22060046 used SysFree instead of SysFault, so that
		// the operating system would eventually give the memory
		// back to us again, so that an efence program could run
		// longer without running out of memory. Unfortunately,
		// calling SysFree here without any kind of adjustment of the
		// heap data structures means that when the memory does
		// come back to us, we have the wrong metadata for it, either in
		// the MSpan structures or in the garbage collection bitmap.
		// Using SysFault here means that the program will run out of
		// memory fairly quickly in efence mode, but at least it won't
		// have mysterious crashes due to confused memory reuse.
		// It should be possible to switch back to SysFree if we also 
		// implement and then call some kind of MHeap_DeleteSpan.
		if(runtime·debug.efence) {
			s->limit = nil;	// prevent mlookup from finding this span
			runtime·SysFault((void*)(s->start<<PageShift), size);
		} else
			runtime·MHeap_Free(&runtime·mheap, s, 1);
		c->local_nlargefree++;
		c->local_largefree += size;
	} else {
		// Small object.
		if(size > 2*sizeof(uintptr))
			((uintptr*)v)[1] = (uintptr)0xfeedfeedfeedfeedll;	// mark as "needs to be zeroed"
		else if(size > sizeof(uintptr))
			((uintptr*)v)[1] = 0;
		// Must mark v freed before calling MCache_Free:
		// it might coalesce v and other blocks into a bigger span
		// and change the bitmap further.
		c->local_nsmallfree[sizeclass]++;
		c->local_cachealloc -= size;
		if(c->alloc[sizeclass] == s) {
			// We own the span, so we can just add v to the freelist
			runtime·markfreed(v);
			((MLink*)v)->next = s->freelist;
			s->freelist = v;
			s->ref--;
		} else {
			// Someone else owns this span.  Add to free queue.
			runtime·MCache_Free(c, v, sizeclass, size);
		}
	}
	g->m->mallocing = 0;
}

int32
runtime·mlookup(void *v, byte **base, uintptr *size, MSpan **sp)
{
	uintptr n, i;
	byte *p;
	MSpan *s;

	g->m->mcache->local_nlookup++;
	if (sizeof(void*) == 4 && g->m->mcache->local_nlookup >= (1<<30)) {
		// purge cache stats to prevent overflow
		runtime·lock(&runtime·mheap);
		runtime·purgecachedstats(g->m->mcache);
		runtime·unlock(&runtime·mheap);
	}

	s = runtime·MHeap_LookupMaybe(&runtime·mheap, v);
	if(sp)
		*sp = s;
	if(s == nil) {
		if(base)
			*base = nil;
		if(size)
			*size = 0;
		return 0;
	}

	p = (byte*)((uintptr)s->start<<PageShift);
	if(s->sizeclass == 0) {
		// Large object.
		if(base)
			*base = p;
		if(size)
			*size = s->npages<<PageShift;
		return 1;
	}

	n = s->elemsize;
	if(base) {
		i = ((byte*)v - p)/n;
		*base = p + i*n;
	}
	if(size)
		*size = n;

	return 1;
}

void
runtime·purgecachedstats(MCache *c)
{
	MHeap *h;
	int32 i;

	// Protected by either heap or GC lock.
	h = &runtime·mheap;
	mstats.heap_alloc += c->local_cachealloc;
	c->local_cachealloc = 0;
	mstats.nlookup += c->local_nlookup;
	c->local_nlookup = 0;
	h->largefree += c->local_largefree;
	c->local_largefree = 0;
	h->nlargefree += c->local_nlargefree;
	c->local_nlargefree = 0;
	for(i=0; i<nelem(c->local_nsmallfree); i++) {
		h->nsmallfree[i] += c->local_nsmallfree[i];
		c->local_nsmallfree[i] = 0;
	}
}

// Size of the trailing by_size array differs between Go and C,
// NumSizeClasses was changed, but we can not change Go struct because of backward compatibility.
// sizeof_C_MStats is what C thinks about size of Go struct.
uintptr runtime·sizeof_C_MStats = sizeof(MStats) - (NumSizeClasses - 61) * sizeof(mstats.by_size[0]);

#define MaxArena32 (2U<<30)

// For use by Go.  It can't be a constant in Go, unfortunately,
// because it depends on the OS.
uintptr runtime·maxMem = MaxMem;

void
runtime·mallocinit(void)
{
	byte *p, *p1;
	uintptr arena_size, bitmap_size, spans_size, p_size;
	extern byte end[];
	uintptr limit;
	uint64 i;
	bool reserved;

	p = nil;
	p_size = 0;
	arena_size = 0;
	bitmap_size = 0;
	spans_size = 0;
	reserved = false;

	// for 64-bit build
	USED(p);
	USED(p_size);
	USED(arena_size);
	USED(bitmap_size);
	USED(spans_size);

	runtime·InitSizes();

	if(runtime·class_to_size[TinySizeClass] != TinySize)
		runtime·throw("bad TinySizeClass");

	// limit = runtime·memlimit();
	// See https://code.google.com/p/go/issues/detail?id=5049
	// TODO(rsc): Fix after 1.1.
	limit = 0;

	// Set up the allocation arena, a contiguous area of memory where
	// allocated data will be found.  The arena begins with a bitmap large
	// enough to hold 4 bits per allocated word.
	if(sizeof(void*) == 8 && (limit == 0 || limit > (1<<30))) {
		// On a 64-bit machine, allocate from a single contiguous reservation.
		// 128 GB (MaxMem) should be big enough for now.
		//
		// The code will work with the reservation at any address, but ask
		// SysReserve to use 0x0000XXc000000000 if possible (XX=00...7f).
		// Allocating a 128 GB region takes away 37 bits, and the amd64
		// doesn't let us choose the top 17 bits, so that leaves the 11 bits
		// in the middle of 0x00c0 for us to choose.  Choosing 0x00c0 means
		// that the valid memory addresses will begin 0x00c0, 0x00c1, ..., 0x00df.
		// In little-endian, that's c0 00, c1 00, ..., df 00. None of those are valid
		// UTF-8 sequences, and they are otherwise as far away from 
		// ff (likely a common byte) as possible.  If that fails, we try other 0xXXc0
		// addresses.  An earlier attempt to use 0x11f8 caused out of memory errors
		// on OS X during thread allocations.  0x00c0 causes conflicts with
		// AddressSanitizer which reserves all memory up to 0x0100.
		// These choices are both for debuggability and to reduce the
		// odds of the conservative garbage collector not collecting memory
		// because some non-pointer block of memory had a bit pattern
		// that matched a memory address.
		//
		// Actually we reserve 136 GB (because the bitmap ends up being 8 GB)
		// but it hardly matters: e0 00 is not valid UTF-8 either.
		//
		// If this fails we fall back to the 32 bit memory mechanism
		arena_size = MaxMem;
		bitmap_size = arena_size / (sizeof(void*)*8/4);
		spans_size = arena_size / PageSize * sizeof(runtime·mheap.spans[0]);
		spans_size = ROUND(spans_size, PageSize);
		for(i = 0; i <= 0x7f; i++) {
			p = (void*)(i<<40 | 0x00c0ULL<<32);
			p_size = bitmap_size + spans_size + arena_size + PageSize;
			p = runtime·SysReserve(p, p_size, &reserved);
			if(p != nil)
				break;
		}
	}
	if (p == nil) {
		// On a 32-bit machine, we can't typically get away
		// with a giant virtual address space reservation.
		// Instead we map the memory information bitmap
		// immediately after the data segment, large enough
		// to handle another 2GB of mappings (256 MB),
		// along with a reservation for another 512 MB of memory.
		// When that gets used up, we'll start asking the kernel
		// for any memory anywhere and hope it's in the 2GB
		// following the bitmap (presumably the executable begins
		// near the bottom of memory, so we'll have to use up
		// most of memory before the kernel resorts to giving out
		// memory before the beginning of the text segment).
		//
		// Alternatively we could reserve 512 MB bitmap, enough
		// for 4GB of mappings, and then accept any memory the
		// kernel threw at us, but normally that's a waste of 512 MB
		// of address space, which is probably too much in a 32-bit world.
		bitmap_size = MaxArena32 / (sizeof(void*)*8/4);
		arena_size = 512<<20;
		spans_size = MaxArena32 / PageSize * sizeof(runtime·mheap.spans[0]);
		if(limit > 0 && arena_size+bitmap_size+spans_size > limit) {
			bitmap_size = (limit / 9) & ~((1<<PageShift) - 1);
			arena_size = bitmap_size * 8;
			spans_size = arena_size / PageSize * sizeof(runtime·mheap.spans[0]);
		}
		spans_size = ROUND(spans_size, PageSize);

		// SysReserve treats the address we ask for, end, as a hint,
		// not as an absolute requirement.  If we ask for the end
		// of the data segment but the operating system requires
		// a little more space before we can start allocating, it will
		// give out a slightly higher pointer.  Except QEMU, which
		// is buggy, as usual: it won't adjust the pointer upward.
		// So adjust it upward a little bit ourselves: 1/4 MB to get
		// away from the running binary image and then round up
		// to a MB boundary.
		p = (byte*)ROUND((uintptr)end + (1<<18), 1<<20);
		p_size = bitmap_size + spans_size + arena_size + PageSize;
		p = runtime·SysReserve(p, p_size, &reserved);
		if(p == nil)
			runtime·throw("runtime: cannot reserve arena virtual address space");
	}

	// PageSize can be larger than OS definition of page size,
	// so SysReserve can give us a PageSize-unaligned pointer.
	// To overcome this we ask for PageSize more and round up the pointer.
	p1 = (byte*)ROUND((uintptr)p, PageSize);

	runtime·mheap.spans = (MSpan**)p1;
	runtime·mheap.bitmap = p1 + spans_size;
	runtime·mheap.arena_start = p1 + spans_size + bitmap_size;
	runtime·mheap.arena_used = runtime·mheap.arena_start;
	runtime·mheap.arena_end = p + p_size;
	runtime·mheap.arena_reserved = reserved;

	if(((uintptr)runtime·mheap.arena_start & (PageSize-1)) != 0)
		runtime·throw("misrounded allocation in mallocinit");

	// Initialize the rest of the allocator.	
	runtime·MHeap_Init(&runtime·mheap);
	g->m->mcache = runtime·allocmcache();

	// See if it works.
	runtime·free(runtime·malloc(TinySize));
}

void*
runtime·MHeap_SysAlloc(MHeap *h, uintptr n)
{
	byte *p, *p_end;
	uintptr p_size;
	bool reserved;

	if(n > h->arena_end - h->arena_used) {
		// We are in 32-bit mode, maybe we didn't use all possible address space yet.
		// Reserve some more space.
		byte *new_end;

		p_size = ROUND(n + PageSize, 256<<20);
		new_end = h->arena_end + p_size;
		if(new_end <= h->arena_start + MaxArena32) {
			// TODO: It would be bad if part of the arena
			// is reserved and part is not.
			p = runtime·SysReserve(h->arena_end, p_size, &reserved);
			if(p == h->arena_end) {
				h->arena_end = new_end;
				h->arena_reserved = reserved;
			}
			else if(p+p_size <= h->arena_start + MaxArena32) {
				// Keep everything page-aligned.
				// Our pages are bigger than hardware pages.
				h->arena_end = p+p_size;
				h->arena_used = p + (-(uintptr)p&(PageSize-1));
				h->arena_reserved = reserved;
			} else {
				uint64 stat;
				stat = 0;
				runtime·SysFree(p, p_size, &stat);
			}
		}
	}
	if(n <= h->arena_end - h->arena_used) {
		// Keep taking from our reservation.
		p = h->arena_used;
		runtime·SysMap(p, n, h->arena_reserved, &mstats.heap_sys);
		h->arena_used += n;
		runtime·MHeap_MapBits(h);
		runtime·MHeap_MapSpans(h);
		if(raceenabled)
			runtime·racemapshadow(p, n);
		
		if(((uintptr)p & (PageSize-1)) != 0)
			runtime·throw("misrounded allocation in MHeap_SysAlloc");
		return p;
	}
	
	// If using 64-bit, our reservation is all we have.
	if(h->arena_end - h->arena_start >= MaxArena32)
		return nil;

	// On 32-bit, once the reservation is gone we can
	// try to get memory at a location chosen by the OS
	// and hope that it is in the range we allocated bitmap for.
	p_size = ROUND(n, PageSize) + PageSize;
	p = runtime·SysAlloc(p_size, &mstats.heap_sys);
	if(p == nil)
		return nil;

	if(p < h->arena_start || p+p_size - h->arena_start >= MaxArena32) {
		runtime·printf("runtime: memory allocated by OS (%p) not in usable range [%p,%p)\n",
			p, h->arena_start, h->arena_start+MaxArena32);
		runtime·SysFree(p, p_size, &mstats.heap_sys);
		return nil;
	}
	
	p_end = p + p_size;
	p += -(uintptr)p & (PageSize-1);
	if(p+n > h->arena_used) {
		h->arena_used = p+n;
		if(p_end > h->arena_end)
			h->arena_end = p_end;
		runtime·MHeap_MapBits(h);
		runtime·MHeap_MapSpans(h);
		if(raceenabled)
			runtime·racemapshadow(p, n);
	}
	
	if(((uintptr)p & (PageSize-1)) != 0)
		runtime·throw("misrounded allocation in MHeap_SysAlloc");
	return p;
}

static struct
{
	Lock;
	byte*	pos;
	byte*	end;
} persistent;

enum
{
	PersistentAllocChunk	= 256<<10,
	PersistentAllocMaxBlock	= 64<<10,  // VM reservation granularity is 64K on windows
};

// Wrapper around SysAlloc that can allocate small chunks.
// There is no associated free operation.
// Intended for things like function/type/debug-related persistent data.
// If align is 0, uses default align (currently 8).
void*
runtime·persistentalloc(uintptr size, uintptr align, uint64 *stat)
{
	byte *p;

	if(align != 0) {
		if(align&(align-1))
			runtime·throw("persistentalloc: align is not a power of 2");
		if(align > PageSize)
			runtime·throw("persistentalloc: align is too large");
	} else
		align = 8;
	if(size >= PersistentAllocMaxBlock)
		return runtime·SysAlloc(size, stat);
	runtime·lock(&persistent);
	persistent.pos = (byte*)ROUND((uintptr)persistent.pos, align);
	if(persistent.pos + size > persistent.end) {
		persistent.pos = runtime·SysAlloc(PersistentAllocChunk, &mstats.other_sys);
		if(persistent.pos == nil) {
			runtime·unlock(&persistent);
			runtime·throw("runtime: cannot allocate memory");
		}
		persistent.end = persistent.pos + PersistentAllocChunk;
	}
	p = persistent.pos;
	persistent.pos += size;
	runtime·unlock(&persistent);
	if(stat != &mstats.other_sys) {
		// reaccount the allocation against provided stat
		runtime·xadd64(stat, size);
		runtime·xadd64(&mstats.other_sys, -(uint64)size);
	}
	return p;
}

// Runtime stubs.

void*
runtime·mal(uintptr n)
{
	return runtime·mallocgc(n, nil, 0);
}

static void*
cnew(Type *typ, intgo n)
{
	if(n < 0 || (typ->size > 0 && n > MaxMem/typ->size))
		runtime·panicstring("runtime: allocation size out of range");
	return runtime·mallocgc(typ->size*n, typ, typ->kind&KindNoPointers ? FlagNoScan : 0);
}

// same as runtime·new, but callable from C
void*
runtime·cnew(Type *typ)
{
	return cnew(typ, 1);
}

void*
runtime·cnewarray(Type *typ, intgo n)
{
	return cnew(typ, n);
}

static void
setFinalizer(Eface obj, Eface finalizer)
{
	byte *base;
	uintptr size;
	FuncType *ft;
	int32 i;
	uintptr nret;
	Type *t;
	Type *fint;
	PtrType *ot;
	Iface iface;

	if(obj.type == nil) {
		runtime·printf("runtime.SetFinalizer: first argument is nil interface\n");
		goto throw;
	}
	if((obj.type->kind&KindMask) != KindPtr) {
		runtime·printf("runtime.SetFinalizer: first argument is %S, not pointer\n", *obj.type->string);
		goto throw;
	}
	ot = (PtrType*)obj.type;
	// As an implementation detail we do not run finalizers for zero-sized objects,
	// because we use &runtime·zerobase for all such allocations.
	if(ot->elem != nil && ot->elem->size == 0)
		return;
	// The following check is required for cases when a user passes a pointer to composite literal,
	// but compiler makes it a pointer to global. For example:
	//	var Foo = &Object{}
	//	func main() {
	//		runtime.SetFinalizer(Foo, nil)
	//	}
	// See issue 7656.
	if((byte*)obj.data < runtime·mheap.arena_start || runtime·mheap.arena_used <= (byte*)obj.data)
		return;
	if(!runtime·mlookup(obj.data, &base, &size, nil) || obj.data != base) {
		// As an implementation detail we allow to set finalizers for an inner byte
		// of an object if it could come from tiny alloc (see mallocgc for details).
		if(ot->elem == nil || (ot->elem->kind&KindNoPointers) == 0 || ot->elem->size >= TinySize) {
			runtime·printf("runtime.SetFinalizer: pointer not at beginning of allocated block (%p)\n", obj.data);
			goto throw;
		}
	}
	if(finalizer.type != nil) {
		runtime·createfing();
		if(finalizer.type->kind != KindFunc)
			goto badfunc;
		ft = (FuncType*)finalizer.type;
		if(ft->dotdotdot || ft->in.len != 1)
			goto badfunc;
		fint = *(Type**)ft->in.array;
		if(fint == obj.type) {
			// ok - same type
		} else if(fint->kind == KindPtr && (fint->x == nil || fint->x->name == nil || obj.type->x == nil || obj.type->x->name == nil) && ((PtrType*)fint)->elem == ((PtrType*)obj.type)->elem) {
			// ok - not same type, but both pointers,
			// one or the other is unnamed, and same element type, so assignable.
		} else if(fint->kind == KindInterface && ((InterfaceType*)fint)->mhdr.len == 0) {
			// ok - satisfies empty interface
		} else if(fint->kind == KindInterface && runtime·ifaceE2I2((InterfaceType*)fint, obj, &iface)) {
			// ok - satisfies non-empty interface
		} else
			goto badfunc;

		// compute size needed for return parameters
		nret = 0;
		for(i=0; i<ft->out.len; i++) {
			t = ((Type**)ft->out.array)[i];
			nret = ROUND(nret, t->align) + t->size;
		}
		nret = ROUND(nret, sizeof(void*));
		ot = (PtrType*)obj.type;
		if(!runtime·addfinalizer(obj.data, finalizer.data, nret, fint, ot)) {
			runtime·printf("runtime.SetFinalizer: finalizer already set\n");
			goto throw;
		}
	} else {
		// NOTE: asking to remove a finalizer when there currently isn't one set is OK.
		runtime·removefinalizer(obj.data);
	}
	return;

badfunc:
	runtime·printf("runtime.SetFinalizer: cannot pass %S to finalizer %S\n", *obj.type->string, *finalizer.type->string);
throw:
	runtime·throw("runtime.SetFinalizer");
}

void
runtime·setFinalizer(void)
{
	Eface obj, finalizer;

	obj.type = g->m->ptrarg[0];
	obj.data = g->m->ptrarg[1];
	finalizer.type = g->m->ptrarg[2];
	finalizer.data = g->m->ptrarg[3];
	g->m->ptrarg[0] = nil;
	g->m->ptrarg[1] = nil;
	g->m->ptrarg[2] = nil;
	g->m->ptrarg[3] = nil;
	setFinalizer(obj, finalizer);
}

// mcallable cache refill
void 
runtime·mcacheRefill(void)
{
	runtime·MCache_Refill(g->m->mcache, (int32)g->m->scalararg[0]);
}

void
runtime·largeAlloc(void)
{
	uintptr npages, size;
	MSpan *s;
	void *v;
	int32 flag;

	//runtime·printf("largeAlloc size=%D\n", g->m->scalararg[0]);
	// Allocate directly from heap.
	size = g->m->scalararg[0];
	flag = (int32)g->m->scalararg[1];
	if(size + PageSize < size)
		runtime·throw("out of memory");
	npages = size >> PageShift;
	if((size & PageMask) != 0)
		npages++;
	s = runtime·MHeap_Alloc(&runtime·mheap, npages, 0, 1, !(flag & FlagNoZero));
	if(s == nil)
		runtime·throw("out of memory");
	s->limit = (byte*)(s->start<<PageShift) + size;
	v = (void*)(s->start << PageShift);
	// setup for mark sweep
	runtime·markspan(v, 0, 0, true);
	g->m->ptrarg[0] = s;
}
