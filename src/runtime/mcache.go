// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Per-P malloc cache for small objects.
//
// See malloc.h for an overview.

package runtime

import "unsafe"

// dummy MSpan that contains no free objects.
var emptymspan mspan

func allocmcache() *mcache {
	lock(&mheap_.lock)
	c := (*mcache)(fixAlloc_Alloc(&mheap_.cachealloc))
	unlock(&mheap_.lock)
	memclr(unsafe.Pointer(c), unsafe.Sizeof(*c))
	for i := 0; i < _NumSizeClasses; i++ {
		c.alloc[i] = &emptymspan
	}

	// Set first allocation sample size.
	rate := MemProfileRate
	if rate > 0x3fffffff { // make 2*rate not overflow
		rate = 0x3fffffff
	}
	if rate != 0 {
		c.next_sample = int32(int(fastrand1()) % (2 * rate))
	}

	return c
}

func freemcache(c *mcache) {
	systemstack(func() {
		mCache_ReleaseAll(c)
		stackcache_clear(c)

		// NOTE(rsc,rlh): If gcworkbuffree comes back, we need to coordinate
		// with the stealing of gcworkbufs during garbage collection to avoid
		// a race where the workbuf is double-freed.
		// gcworkbuffree(c.gcworkbuf)

		lock(&mheap_.lock)
		purgecachedstats(c)
		fixAlloc_Free(&mheap_.cachealloc, unsafe.Pointer(c))
		unlock(&mheap_.lock)
	})
}

// Gets a span that has a free object in it and assigns it
// to be the cached span for the given sizeclass.  Returns this span.
func mCache_Refill(c *mcache, sizeclass int32) *mspan {
	_g_ := getg()

	_g_.m.locks++
	// Return the current cached span to the central lists.
	s := c.alloc[sizeclass]
	if s.freelist.ptr() != nil {
		gothrow("refill on a nonempty span")
	}
	if s != &emptymspan {
		s.incache = false
	}

	// Get a new cached span from the central lists.
	s = mCentral_CacheSpan(&mheap_.central[sizeclass].mcentral)
	if s == nil {
		gothrow("out of memory")
	}
	if s.freelist.ptr() == nil {
		println(s.ref, (s.npages<<_PageShift)/s.elemsize)
		gothrow("empty span")
	}
	c.alloc[sizeclass] = s
	_g_.m.locks--
	return s
}

func mCache_ReleaseAll(c *mcache) {
	for i := 0; i < _NumSizeClasses; i++ {
		s := c.alloc[i]
		if s != &emptymspan {
			mCentral_UncacheSpan(&mheap_.central[i].mcentral, s)
			c.alloc[i] = &emptymspan
		}
	}
}
