// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// See malloc.h for overview.
//
// TODO(rsc): double-check stats.
// TODO(rsc): solve "stack overflow during malloc" problem.

#include "runtime.h"
#include "malloc.h"

MHeap mheap;
MStats mstats;

// Allocate an object of at least size bytes.
// Small objects are allocated from the per-thread cache's free lists.
// Large objects (> 32 kB) are allocated straight from the heap.
void*
malloc(uintptr size)
{
	int32 sizeclass;
	MCache *c;
	uintptr npages;
	MSpan *s;
	void *v;

	if(size == 0)
		size = 1;

	if(size <= MaxSmallSize) {
		// Allocate from mcache free lists.
		sizeclass = SizeToClass(size);
		size = class_to_size[sizeclass];
		c = m->mcache;
		v = MCache_Alloc(c, sizeclass, size);
		if(v == nil)
			return nil;
		mstats.alloc += size;
		return v;
	}

	// TODO(rsc): Report tracebacks for very large allocations.

	// Allocate directly from heap.
	npages = size >> PageShift;
	if((size & PageMask) != 0)
		npages++;
	s = MHeap_Alloc(&mheap, npages, 0);
	if(s == nil)
		return nil;
	mstats.alloc += npages<<PageShift;
	return (void*)(s->start << PageShift);
}

// Free the object whose base pointer is v.
void
free(void *v)
{
	int32 sizeclass, size;
	uintptr page, tmp;
	MSpan *s;
	MCache *c;

	// Find size class for v.
	page = (uintptr)v >> PageShift;
	sizeclass = MHeapMapCache_GET(&mheap.mapcache, page, tmp);
	if(sizeclass == 0) {
		// Missed in cache.
		s = MHeap_Lookup(&mheap, page);
		if(s == nil)
			throw("free - invalid pointer");
		sizeclass = s->sizeclass;
		if(sizeclass == 0) {
			// Large object.
			mstats.alloc -= s->npages<<PageShift;
			sys·memclr(v, s->npages<<PageShift);
			MHeap_Free(&mheap, s);
			return;
		}
		MHeapMapCache_SET(&mheap.mapcache, page, sizeclass);
	}

	// Small object.
	c = m->mcache;
	size = class_to_size[sizeclass];
	sys·memclr(v, size);
	mstats.alloc -= size;
	MCache_Free(c, v, sizeclass, size);
}

MCache*
allocmcache(void)
{
	return FixAlloc_Alloc(&mheap.cachealloc);
}

void
mallocinit(void)
{
	InitSizes();
	MHeap_Init(&mheap, SysAlloc);
	m->mcache = allocmcache();

	// See if it works.
	free(malloc(1));
}

// TODO(rsc): Move elsewhere.
enum
{
	NHUNK		= 20<<20,

	PROT_NONE	= 0x00,
	PROT_READ	= 0x01,
	PROT_WRITE	= 0x02,
	PROT_EXEC	= 0x04,

	MAP_FILE	= 0x0000,
	MAP_SHARED	= 0x0001,
	MAP_PRIVATE	= 0x0002,
	MAP_FIXED	= 0x0010,
	MAP_ANON	= 0x1000,	// not on Linux - TODO(rsc)
};

void*
SysAlloc(uintptr n)
{
	mstats.sys += n;
	return sys·mmap(nil, n, PROT_READ|PROT_WRITE, MAP_ANON|MAP_PRIVATE, 0, 0);
}

void
SysUnused(void *v, uintptr n)
{
	// TODO(rsc): call madvise MADV_DONTNEED
}

void
SysFree(void *v, uintptr n)
{
	USED(v);
	USED(n);
	// TODO(rsc): call munmap
}


// Go function stubs.

void
malloc·Alloc(uintptr n, byte *p)
{
	p = malloc(n);
	FLUSH(&p);
}

void
malloc·Free(byte *p)
{
	free(p);
}

void
malloc·GetStats(MStats *s)
{
	s = &mstats;
	FLUSH(&s);
}

