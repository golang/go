// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Per-P malloc cache for small objects.
//
// See malloc.h for an overview.

#include "runtime.h"
#include "arch_GOARCH.h"
#include "malloc.h"

extern volatile intgo runtime·MemProfileRate;

// dummy MSpan that contains no free objects.
MSpan runtime·emptymspan;

MCache*
runtime·allocmcache(void)
{
	intgo rate;
	MCache *c;
	int32 i;

	runtime·lock(&runtime·mheap.lock);
	c = runtime·FixAlloc_Alloc(&runtime·mheap.cachealloc);
	runtime·unlock(&runtime·mheap.lock);
	runtime·memclr((byte*)c, sizeof(*c));
	for(i = 0; i < NumSizeClasses; i++)
		c->alloc[i] = &runtime·emptymspan;

	// Set first allocation sample size.
	rate = runtime·MemProfileRate;
	if(rate > 0x3fffffff)	// make 2*rate not overflow
		rate = 0x3fffffff;
	if(rate != 0)
		c->next_sample = runtime·fastrand1() % (2*rate);

	return c;
}

static void
freemcache(MCache *c)
{
	runtime·MCache_ReleaseAll(c);
	runtime·stackcache_clear(c);
	runtime·gcworkbuffree(c->gcworkbuf);
	runtime·lock(&runtime·mheap.lock);
	runtime·purgecachedstats(c);
	runtime·FixAlloc_Free(&runtime·mheap.cachealloc, c);
	runtime·unlock(&runtime·mheap.lock);
}

static void
freemcache_m(void)
{
	MCache *c;

	c = g->m->ptrarg[0];
	g->m->ptrarg[0] = nil;
	freemcache(c);
}

void
runtime·freemcache(MCache *c)
{
	void (*fn)(void);

	g->m->ptrarg[0] = c;
	fn = freemcache_m;
	runtime·onM(&fn);
}

// Gets a span that has a free object in it and assigns it
// to be the cached span for the given sizeclass.  Returns this span.
MSpan*
runtime·MCache_Refill(MCache *c, int32 sizeclass)
{
	MSpan *s;

	g->m->locks++;
	// Return the current cached span to the central lists.
	s = c->alloc[sizeclass];
	if(s->freelist != nil)
		runtime·throw("refill on a nonempty span");
	if(s != &runtime·emptymspan)
		s->incache = false;

	// Get a new cached span from the central lists.
	s = runtime·MCentral_CacheSpan(&runtime·mheap.central[sizeclass].mcentral);
	if(s == nil)
		runtime·throw("out of memory");
	if(s->freelist == nil) {
		runtime·printf("%d %d\n", s->ref, (int32)((s->npages << PageShift) / s->elemsize));
		runtime·throw("empty span");
	}
	c->alloc[sizeclass] = s;
	g->m->locks--;
	return s;
}

void
runtime·MCache_ReleaseAll(MCache *c)
{
	int32 i;
	MSpan *s;

	for(i=0; i<NumSizeClasses; i++) {
		s = c->alloc[i];
		if(s != &runtime·emptymspan) {
			runtime·MCentral_UncacheSpan(&runtime·mheap.central[i].mcentral, s);
			c->alloc[i] = &runtime·emptymspan;
		}
	}
}
