// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// General C malloc/free, but intended for Go.
// Same design as tcmalloc:
// see https://www/eng/designdocs/tcmalloc/tcmalloc.html

// TODO:
//	* Central free lists.
//	* Thread cache stealing.
//	* Return memory to the OS.
//	* Memory footprint during testrandom is too big.
//	* Need to coalesce adjacent free spans.
//
//	*** Some way to avoid the ``malloc overflows the stack
//	    during the stack overflow malloc'' problem.

#include "malloc.h"

typedef struct Span Span;
typedef struct Central Central;

// A Span contains metadata about a range of pages.
enum {
	SpanInUse = 0,	// span has been handed out by allocator
	SpanFree = 1,	// span is in central free list
};
struct Span
{
	Span *next;	// in free lists
	byte *base;	// first byte in span
	uintptr length;	// number of pages in span
	int32 cl;
	int32 state;	// state (enum above)
//	int ref;	// reference count if state == SpanInUse (for GC)
//	void *type;	// object type if state == SpanInUse (for GC)
};

// The Central cache contains a list of free spans,
// as well as free lists of small blocks.
struct Central
{
	Lock;
	Span *free[256];
	Span *large;	// free spans >= MaxPage pages
};

static Central central;
static PageMap spanmap;

// Insert a new span into the map.
static void
insertspan(Span *s)
{
	int32 i;
	uintptr base;

	// TODO: This is likely too slow for large spans.
	base = (uintptr)s->base >> PageShift;
	for(i=0; i<s->length; i++)
		pminsert(&spanmap, base+i, s);
}

// Record that a span has gotten smaller.
static void
shrinkspan(Span *s, int32 newlength)
{
	int32 i;
	uintptr base;

	// TODO: This is unnecessary, because an insertspan is next.
	base = (uintptr)s->base >> PageShift;
	for(i=newlength; i<s->length; i++)
		pminsert(&spanmap, base+i, nil);

	s->length = newlength;
}

// Find the span for a given pointer.
static Span*
spanofptr(void *v)
{
	return pmlookup(&spanmap, (uintptr)v >> PageShift);
}

static void freespan(Span*);

// Allocate a span of at least n pages.
static Span*
allocspan(int32 npage)
{
	Span *s, **l, *s1;
	int32 allocnpage, i;

	// Look in the n-page free lists for big enough n.
	for(i=npage; i<nelem(central.free); i++) {
		s = central.free[i];
		if(s != nil) {
			central.free[i] = s->next;
			goto havespan;
		}
	}

	// Look in the large list, which has large runs of pages.
	for(l=&central.large; (s=*l) != nil; l=&s->next) {
		if(s->length >= npage) {
			*l = s->next;
			s->next = nil;
//if(s->length > npage) printf("Chop span %D for %d\n", s->length, npage);
			goto havespan;
		}
	}

	// Otherwise we need more memory.
	// TODO: Could try to release this lock while asking for memory.
	s = trivalloc(sizeof *s);
	allocnpage = npage;
	if(allocnpage < (1<<20>>PageShift))	// TODO: Tune
		allocnpage = (1<<20>>PageShift);
	s->length = allocnpage;
//printf("New span %d for %d\n", allocnpage, npage);
	s->base = trivalloc(allocnpage<<PageShift);
	insertspan(s);

havespan:
	// If span is bigger than needed, redistribute the remainder.
	if(s->length > npage) {
		s1 = trivalloc(sizeof *s);
		s1->base = s->base + (npage << PageShift);
		s1->length = s->length - npage;
		shrinkspan(s, npage);
		insertspan(s1);
		freespan(s1);
	}
	s->state = SpanInUse;
	return s;
}

// Free a span.
static void
freespan(Span *s)
{
	Span **l;
	Span *ss;

	s->state = SpanFree;
	if(s->length < nelem(central.free)) {
		s->next = central.free[s->length];
		central.free[s->length] = s;
	} else {
		// Keep central.large sorted in
		// increasing size for best-fit allocation.
		for(l = &central.large; (ss=*l) != nil; l=&ss->next)
			if(ss->length >= s->length)
				break;
		s->next = *l;
		*l = s;
	}
}

// Small objects are kept on per-size free lists in the M.
// There are SmallFreeClasses (defined in runtime.h) different lists.
static int32 classtosize[SmallFreeClasses] = {
	/*
	seq 8 8 127 | sed 's/$/,/' | fmt
	seq 128 16 255 | sed 's/$/,/' | fmt
	seq 256 32 511 | sed 's/$/,/' | fmt
	seq 512 64 1023 | sed 's/$/,/' | fmt
	seq 1024 128 2047 | sed 's/$/,/' | fmt
	seq 2048 256 32768 | sed 's/$/,/' | fmt
	*/
	8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120,
	128, 144, 160, 176, 192, 208, 224, 240,
	256, 288, 320, 352, 384, 416, 448, 480,
	512, 576, 640, 704, 768, 832, 896, 960,
	1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920,
	2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4352, 4608,
	4864, 5120, 5376, 5632, 5888, 6144, 6400, 6656, 6912, 7168, 7424,
	7680, 7936, 8192, 8448, 8704, 8960, 9216, 9472, 9728, 9984, 10240,
	10496, 10752, 11008, 11264, 11520, 11776, 12032, 12288, 12544,
	12800, 13056, 13312, 13568, 13824, 14080, 14336, 14592, 14848,
	15104, 15360, 15616, 15872, 16128, 16384, 16640, 16896, 17152,
	17408, 17664, 17920, 18176, 18432, 18688, 18944, 19200, 19456,
	19712, 19968, 20224, 20480, 20736, 20992, 21248, 21504, 21760,
	22016, 22272, 22528, 22784, 23040, 23296, 23552, 23808, 24064,
	24320, 24576, 24832, 25088, 25344, 25600, 25856, 26112, 26368,
	26624, 26880, 27136, 27392, 27648, 27904, 28160, 28416, 28672,
	28928, 29184, 29440, 29696, 29952, 30208, 30464, 30720, 30976,
	31232, 31488, 31744, 32000, 32256, 32512, 32768,
};
enum {
	LargeSize = 32768
};

// Trigger compile error if nelem(classtosize) != SmallFreeClasses.
static int32 zzz1[SmallFreeClasses-nelem(classtosize)+1];
static int32 zzz2[nelem(classtosize)-SmallFreeClasses+1];

static int32
sizetoclass(int32 siz)
{
	if(siz <= 0)
		return 0;
	if(siz <= 128)
		return (siz-1) >> 3;
	if(siz <= 256)
		return ((siz-1) >> 4) + 8;
	if(siz <= 512)
		return ((siz-1) >> 5) + 16;
	if(siz <= 1024)
		return ((siz-1) >> 6) + 24;
	if(siz <= 2048)
		return ((siz-1) >> 7) + 32;
	if(siz <= 32768)
		return ((siz-1) >> 8) + 40;
	throw("sizetoclass - invalid size");
	return -1;
}

void
allocator·testsizetoclass(void)
{
	int32 i, n;

	n = 0;
	for(i=0; i<nelem(classtosize); i++) {
		for(; n <= classtosize[i]; n++) {
			if(sizetoclass(n) != i) {
				printf("sizetoclass %d = %d want %d\n", n, sizetoclass(n), i);
				throw("testsizetoclass");
			}
		}
	}
	if (n != 32768+1) {
		printf("testsizetoclass stopped at %d\n", n);
		throw("testsizetoclass");
	}
}

// Grab a bunch of objects of size class cl off the central free list.
// Set *pn to the number of objects returned.
static void*
centralgrab(int32 cl, int32 *pn)
{
	byte *p;
	Span *s;
	int32 chunk, i, n, siz;

	// For now there is no central free list.
	// Fall back to allocating a new span
	// and chopping it up.
	chunk = classtosize[cl] * 1024;
	if(chunk > 1<<20) {
		chunk = 1<<20;
	}
	chunk = (chunk+PageMask) & ~PageMask;
	s = allocspan(chunk>>PageShift);
//printf("New class %d\n", cl);
	s->state = SpanInUse;
	s->cl = cl;
	siz = classtosize[cl];
	n = chunk/siz;
	p = s->base;
//printf("centralgrab cl=%d siz=%d n=%d\n", cl, siz, n);
	for(i=0; i<n-1; i++) {
		*(void**)p = p+siz;
		p += siz;
	}
	*pn = n;
	return s->base;
}

// Allocate a small object of size class cl.
void*
allocsmall(int32 cl)
{
	void **p;
	int32 n;

	if(cl < 0 || cl >= SmallFreeClasses)
		throw("allocsmall - invalid class");

	// try m-local cache.
	p = m->freelist[cl];
	if(p == nil) {
		// otherwise grab some blocks from central cache.
		lock(&central);
//printf("centralgrab for %d\n", cl);
		p = centralgrab(cl, &n);
		// TODO: update local counters using n
		unlock(&central);
	}

//printf("alloc from cl %d\n", cl);
	// advance linked list.
	m->freelist[cl] = *p;

	// Blocks on free list are zeroed except for
	// the linked list pointer that we just used.  Zero it.
	*p = 0;

	return p;
}

// Allocate large object of np pages.
void*
alloclarge(int32 np)
{
	Span *s;

	lock(&central);
//printf("Alloc span %d\n", np);
	s = allocspan(np);
	unlock(&central);
	s->state = SpanInUse;
	s->cl = -1;
	return s->base;
}

// Allocate object of n bytes.
void*
alloc(int32 n)
{
	int32 cl, np;

	if(n < LargeSize) {
		cl = sizetoclass(n);
		if(cl < 0 || cl >= SmallFreeClasses) {
			printf("%d -> %d\n", n, cl);
			throw("alloc - logic error");
		}
		allocator·allocated += classtosize[cl];
		return allocsmall(cl);
	}

	// count number of pages; careful about overflow for big n.
	np = (n>>PageShift) + (((n&PageMask)+PageMask)>>PageShift);
	allocator·allocated += (uint64)np<<PageShift;
	return alloclarge(np);
}

void
allocator·malloc(int32 n, byte *out)
{
	out = alloc(n);
	FLUSH(&out);
}

// Free object with base pointer v.
void
free(void *v)
{
	void **p;
	Span *s;
	int32 siz, off;

	s = spanofptr(v);
	if(s->state != SpanInUse)
		throw("free - invalid pointer1");

	// Big object should be s->base.
	if(s->cl < 0) {
		if(v != s->base)
			throw("free - invalid pointer2");
		// TODO: For large spans, maybe just return the
		// memory to the operating system and let it zero it.
		sys·memclr(s->base, s->length << PageShift);
//printf("Free big %D\n", s->length);
		allocator·allocated -= s->length << PageShift;
		lock(&central);
		freespan(s);
		unlock(&central);
		return;
	}

	// Small object should be aligned properly.
	siz = classtosize[s->cl];
	off = (byte*)v - (byte*)s->base;
	if(off%siz)
		throw("free - invalid pointer3");

	// Zero and add to free list.
	sys·memclr(v, siz);
	allocator·allocated -= siz;
	p = v;
	*p = m->freelist[s->cl];
	m->freelist[s->cl] = p;
//printf("Free siz %d cl %d\n", siz, s->cl);
}

void
allocator·free(byte *v)
{
	free(v);
}

void
allocator·memset(byte *v, int32 c, int32 n)
{
	int32 i;

	for(i=0; i<n; i++)
		v[i] = c;
}

