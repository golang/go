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
#include "textflag.h"

// Mark mheap as 'no pointers', it does not contain interesting pointers but occupies ~45K.
#pragma dataflag NOPTR
MHeap runtime·mheap;
#pragma dataflag NOPTR
MStats runtime·memstats;

int32
runtime·mlookup(void *v, byte **base, uintptr *size, MSpan **sp)
{
	uintptr n, i;
	byte *p;
	MSpan *s;

	g->m->mcache->local_nlookup++;
	if (sizeof(void*) == 4 && g->m->mcache->local_nlookup >= (1<<30)) {
		// purge cache stats to prevent overflow
		runtime·lock(&runtime·mheap.lock);
		runtime·purgecachedstats(g->m->mcache);
		runtime·unlock(&runtime·mheap.lock);
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

#pragma textflag NOSPLIT
void
runtime·purgecachedstats(MCache *c)
{
	MHeap *h;
	int32 i;

	// Protected by either heap or GC lock.
	h = &runtime·mheap;
	mstats.heap_alloc += c->local_cachealloc;
	c->local_cachealloc = 0;
	mstats.tinyallocs += c->local_tinyallocs;
	c->local_tinyallocs = 0;
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
// and all data after by_size is local to C, not exported to Go.
// NumSizeClasses was changed, but we can not change Go struct because of backward compatibility.
// sizeof_C_MStats is what C thinks about size of Go struct.
uintptr runtime·sizeof_C_MStats = offsetof(MStats, by_size[61]);

#define MaxArena32 (2U<<30)

// For use by Go. If it were a C enum it would be made available automatically,
// but the value of MaxMem is too large for enum.
uintptr runtime·maxmem = MaxMem;

void
runtime·mallocinit(void)
{
	byte *p, *p1;
	uintptr arena_size, bitmap_size, spans_size, p_size;
	extern byte runtime·end[];
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
		p = (byte*)ROUND((uintptr)runtime·end + (1<<18), 1<<20);
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
	p = runtime·sysAlloc(p_size, &mstats.heap_sys);
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

void
runtime·setFinalizer_m(void)
{
	FuncVal *fn;
	void *arg;
	uintptr nret;
	Type *fint;
	PtrType *ot;

	fn = g->m->ptrarg[0];
	arg = g->m->ptrarg[1];
	nret = g->m->scalararg[0];
	fint = g->m->ptrarg[2];
	ot = g->m->ptrarg[3];
	g->m->ptrarg[0] = nil;
	g->m->ptrarg[1] = nil;
	g->m->ptrarg[2] = nil;
	g->m->ptrarg[3] = nil;

	g->m->scalararg[0] = runtime·addfinalizer(arg, fn, nret, fint, ot);
}

void
runtime·removeFinalizer_m(void)
{
	void *p;

	p = g->m->ptrarg[0];
	g->m->ptrarg[0] = nil;
	runtime·removefinalizer(p);
}

// mcallable cache refill
void 
runtime·mcacheRefill_m(void)
{
	runtime·MCache_Refill(g->m->mcache, (int32)g->m->scalararg[0]);
}

void
runtime·largeAlloc_m(void)
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
